"""Long-running episode collector service.

Spawns AsyncVectorEnv workers ONCE per env_name at startup, then services
collect() requests over ZMQ. This eliminates the per-iteration startup cost
that the GRPO trainer otherwise pays each iteration:
  - re-importing robocasa/robosuite (~5-10s wall time per worker)
  - spawning AsyncVectorEnv subprocess workers (~1-2s)
  - building MuJoCo models via gym.make (~5-10s per worker)

Multi-task support: pass --env-names with multiple names; the server
pre-initializes one EpisodeCollector per env. Per-task max_episode_steps is
passed as a parallel list (one value per env). Per-iteration RPC dispatches
to the right collector by env_name.

Architecture:
- Trainer (main .venv) → CollectorClient → ZMQ → CollectorServer (robocasa venv)
- CollectorServer holds a dict {env_name: EpisodeCollector}, dispatches per request.
- Each EpisodeCollector owns its own AsyncVectorEnv (group_size workers).

Usage:
    # Terminal 1 (sim venv) — start the long-running collector server.
    # The trainer's in-process model server (started by train_grpo.py at port
    # 5555) serves this collector via --policy-server-port 5555. We do NOT
    # start a separate model server here — that would conflict with the
    # trainer's in-process bind on port 5555.
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \\
        scripts/grpo/collector_server.py \\
        --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
                    robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \\
        --max-episode-steps 720 480 \\
        --group-size 5 --n-action-steps 8 \\
        --policy-server-host 127.0.0.1 --policy-server-port 5555 \\
        --listen-port 5556

    # Terminal 2 (main venv) — trainer connects via CollectorClient, and
    # ALSO hosts the in-process model server on --server-port (5555).
    uv run python scripts/grpo/train_grpo.py \\
        --collector-server-host 127.0.0.1 --collector-server-port 5556

    # Optional standalone debug: to test collector_server WITHOUT the
    # trainer, first start scripts/grpo/grpo_server.py (NOT
    # gr00t/eval/run_gr00t_server.py — that one doesn't install the GRPO
    # noise/raw_action capture hooks the collector relies on).

Operational notes:
- Each EpisodeCollector spawns group_size MuJoCo subprocesses; total worker
  count is len(env_names) × group_size. Sized to fit on one machine — 8
  tasks × 5 workers = 40 sims is comfortable on 64+ GB RAM.
- Memory creep over many iterations is real (small leaks in MuJoCo/robosuite
  caches). Restart the server every ~50-100 iterations as a hygiene measure;
  the trainer's --collector-server-host flag handles connect-fail by simply
  retrying on the next iteration once the server is back.
- The per-iteration verification command (--debug-fast-forward via
  collect_episodes.py CLI) is unaffected — it doesn't go through this server.
"""

import argparse
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import msgpack
import zmq

from collect_episodes import EpisodeCollector, save_episodes


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FatalCollectorError(RuntimeError):
    """Server reported an error that won't change on retry.

    Currently raised for ValueError-class server-side errors — primarily
    `env_name not pre-initialized` mismatches between trainer config and
    server boot args. The trainer's _collect_via_server re-raises this
    immediately rather than burning its 3-iteration retry budget.
    """


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


# How often the run loop wakes from blocking recv() to check the running
# flag (set by SIGTERM handler). Trades a tiny bit of CPU for signal
# responsiveness.
_RECV_POLL_MS = 1000


class CollectorServer:
    """ZMQ REP server that holds one EpisodeCollector per env_name.

    Endpoints:
      - ping  → {"status": "ok", "envs": [...], "group_size": int,
                 "n_action_steps": int, "env_max_steps": {env: int}}
        The trainer pings at __init__ to validate that its config matches
        the server's bake-time args (n_action_steps / group_size / per-env
        max_episode_steps). Any mismatch fails fast with the exact restart
        command needed.
      - collect (data: env_name, output_dir, base_seed, num_groups,
        success_weight, fast_forward_steps, fast_forward_pct)
        → {"n_episodes", "n_successes", "elapsed_s", "env_name"}
      - kill  → {"status": "ok"} then exits the run loop
    """

    def __init__(
        self,
        env_names: list[str],
        max_episode_steps: list[int],
        group_size: int,
        n_action_steps: int,
        policy_server_host: str,
        policy_server_port: int,
        listen_port: int,
        debug_fast_forward: bool = False,
    ):
        if len(env_names) != len(max_episode_steps):
            raise ValueError(
                f"--env-names ({len(env_names)}) and --max-episode-steps "
                f"({len(max_episode_steps)}) must have the same length."
            )

        # Stored for ping()-based validation by the trainer.
        self.group_size = group_size
        self.n_action_steps = n_action_steps
        self.env_max_steps: dict[str, int] = dict(zip(env_names, max_episode_steps))
        self.debug_fast_forward = debug_fast_forward
        self.listen_port = listen_port

        # Pre-initialize one collector per env. Each one spawns its own
        # AsyncVectorEnv (group_size workers) — the entire startup cost is
        # paid once here, not per iteration.
        self.collectors: dict[str, EpisodeCollector] = {}
        for env_name, max_steps in zip(env_names, max_episode_steps):
            print(f"\n[collector_server] Initializing collector for {env_name} (max_steps={max_steps})...")
            self.collectors[env_name] = EpisodeCollector(
                env_name=env_name,
                group_size=group_size,
                max_episode_steps=max_steps,
                n_action_steps=n_action_steps,
                server_host=policy_server_host,
                server_port=policy_server_port,
                debug_fast_forward=debug_fast_forward,
                output_dir="/tmp",  # overridden per-request
            )

        # ZMQ REP socket. RCVTIMEO lets the run loop wake periodically so
        # the SIGTERM handler can request a clean shutdown.
        self.ctx = zmq.Context()
        self.sock = self._make_rep_socket()
        self.running = True

        print(f"\n[collector_server] {len(self.collectors)} collector(s) ready.")
        print(f"[collector_server] Listening on tcp://*:{listen_port}")
        sys.stdout.flush()

    def _make_rep_socket(self) -> zmq.Socket:
        """Create + bind a fresh REP socket with the right options.

        Used by __init__ and by the error path that has to rebuild the
        socket if a reply-send fails (REQ/REP state machine is unrecoverable
        once the reply phase fails mid-flight).
        """
        sock = self.ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, _RECV_POLL_MS)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://*:{self.listen_port}")
        return sock

    def run(self):
        while self.running:
            try:
                msg = self.sock.recv()
            except zmq.error.Again:
                # RCVTIMEO expired — loop back to check self.running.
                continue
            except zmq.error.ContextTerminated:
                break

            try:
                request = msgpack.unpackb(msg, raw=False)
                endpoint = request.get("endpoint", "")

                if endpoint == "ping":
                    self.sock.send(msgpack.packb({
                        "status": "ok",
                        "envs": list(self.collectors),
                        "group_size": self.group_size,
                        "n_action_steps": self.n_action_steps,
                        "env_max_steps": self.env_max_steps,
                        "debug_fast_forward": self.debug_fast_forward,
                    }))
                elif endpoint == "kill":
                    self.sock.send(msgpack.packb({"status": "ok"}))
                    self.running = False
                elif endpoint == "collect":
                    result = self._handle_collect(request.get("data", {}))
                    self.sock.send(msgpack.packb(result))
                else:
                    self.sock.send(msgpack.packb({
                        "error": f"unknown endpoint: {endpoint!r}",
                        "fatal": True,
                    }))
            except Exception as e:
                # Per-request error: log, reply with {"error": ..., "fatal": ...},
                # then continue. ValueError-class errors are config mismatches
                # that won't fix themselves on retry — tag them fatal so the
                # trainer aborts immediately. Other exceptions are treated as
                # transient (server bug, transient OOM, etc.) and the trainer
                # may retry within its consecutive-failure budget.
                fatal = isinstance(e, ValueError)
                if fatal:
                    print(
                        f"[collector_server] config error (fatal): "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                else:
                    traceback.print_exc()

                error_payload = msgpack.packb({
                    "error": f"{type(e).__name__}: {e}",
                    "fatal": fatal,
                })
                try:
                    self.sock.send(error_payload)
                except zmq.error.ZMQError as send_err:
                    # Reply-send failed → REP socket state-machine is dead.
                    # Rebuild before the next recv() so the server stays alive.
                    print(
                        f"[collector_server] WARN: error-reply send failed "
                        f"({send_err!r}); rebuilding REP socket on port "
                        f"{self.listen_port}",
                        file=sys.stderr,
                    )
                    try:
                        self.sock.close(linger=0)
                    except Exception:
                        pass
                    self.sock = self._make_rep_socket()

    def _handle_collect(self, data: dict) -> dict:
        env_name = data["env_name"]
        if env_name not in self.collectors:
            raise ValueError(
                f"env_name {env_name!r} not pre-initialized; server was started "
                f"with: {list(self.collectors)}"
            )

        collector = self.collectors[env_name]

        # Update output_dir for this request. EpisodeCollector reads
        # self.output_dir inside _verify_branch_point (debug montage path);
        # save_episodes is passed output_dir explicitly below. Mutating self
        # is safe here because REQ/REP is synchronous — at most one request
        # in flight at a time.
        output_dir = Path(data["output_dir"])
        collector.output_dir = output_dir

        t0 = time.time()
        # max_groups is optional in the payload — older trainers won't send
        # it. If absent, EpisodeCollector.collect() defaults to num_groups
        # (which disables dynamic mode).
        collect_kwargs = dict(
            num_groups=int(data["num_groups"]),
            base_seed=int(data["base_seed"]),
            success_weight=float(data.get("success_weight", 1.0)),
            fast_forward_steps=int(data.get("fast_forward_steps", 0)),
            fast_forward_pct=float(data.get("fast_forward_pct", 0.5)),
            min_successful_groups=int(data.get("min_successful_groups", 0)),
        )
        if "max_groups" in data and data["max_groups"] is not None:
            collect_kwargs["max_groups"] = int(data["max_groups"])
        # init_state_npz_path is optional and forwarded only when present so a
        # mismatched-version trainer that doesn't send it still works.
        if "init_state_npz_path" in data and data["init_state_npz_path"] is not None:
            collect_kwargs["init_state_npz_path"] = str(data["init_state_npz_path"])
        episodes = collector.collect(**collect_kwargs)
        save_episodes(episodes, str(output_dir))
        elapsed = time.time() - t0

        return {
            "n_episodes": len(episodes),
            "n_successes": int(sum(bool(e["success"]) for e in episodes)),
            "elapsed_s": round(elapsed, 2),
            "env_name": env_name,
        }

    def shutdown(self):
        print("\n[collector_server] Shutting down...")
        for env_name, collector in self.collectors.items():
            try:
                collector.close()
            except Exception as e:
                print(f"  WARN: failed to close collector for {env_name}: {e}")
        try:
            self.sock.close(linger=0)
        except Exception:
            pass
        try:
            self.ctx.term()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class CollectorClient:
    """Thin ZMQ REQ client for CollectorServer.

    Used by train_grpo.py instead of subprocess.Popen("python collect_episodes.py").
    Raises FatalCollectorError on server-side config errors (fail fast); all
    other server errors raise RuntimeError for the trainer's retry path.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        timeout_ms: int = 2_100_000,  # 35 min: longer than any plausible group
        ping_timeout_ms: int = 30_000,  # 30 sec — fail fast on connect/server-down, but tolerate slow server warmup (AsyncVectorEnv worker spawn can take ~10-30s on first ping)
    ):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.ping_timeout_ms = ping_timeout_ms
        self.context = zmq.Context()
        self.socket: zmq.Socket | None = None
        self._init_socket()

    def _init_socket(self):
        # Close the prior socket before swapping it out, otherwise every
        # timeout/ZMQError reconnect leaks an FD + zmq i/o thread (the FD
        # cap eventually trips on long-lived clients).
        if self.socket is not None:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> dict:
        # Use a short timeout for ping. The trainer pings at __init__, so a
        # dead/unreachable server should fail fast (10 sec) rather than
        # hanging for `timeout_ms` (which scales with max_groups and can be
        # >70 min for default config). Override the socket's RCVTIMEO for
        # this single call, then restore the long timeout after.
        return self._call(
            "ping",
            requires_input=False,
            timeout_ms_override=self.ping_timeout_ms,
        )

    def collect(
        self,
        env_name: str,
        output_dir: str,
        base_seed: int,
        num_groups: int,
        success_weight: float = 1.0,
        fast_forward_steps: int = 0,
        fast_forward_pct: float = 0.5,
        min_successful_groups: int = 0,
        max_groups: int | None = None,
        init_state_npz_path: str | None = None,
    ) -> dict:
        payload = {
            "env_name": env_name,
            "output_dir": str(output_dir),
            "base_seed": int(base_seed),
            "num_groups": int(num_groups),
            "success_weight": float(success_weight),
            "fast_forward_steps": int(fast_forward_steps),
            "fast_forward_pct": float(fast_forward_pct),
            "min_successful_groups": int(min_successful_groups),
        }
        # Only include max_groups when set so older servers (which don't know
        # this key) keep the EpisodeCollector default behaviour.
        if max_groups is not None:
            payload["max_groups"] = int(max_groups)
        # Same pattern for init_state_npz_path — optional, only sent when used.
        # Older servers without this key will silently ignore it (the field is
        # in msgpack payload but never looked up).
        if init_state_npz_path is not None:
            payload["init_state_npz_path"] = str(init_state_npz_path)
        return self._call("collect", payload)

    def kill(self) -> dict:
        return self._call("kill", requires_input=False)

    def _call(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True,
        timeout_ms_override: int | None = None,
    ) -> Any:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        # Optionally override RCVTIMEO for this single call (e.g., short
        # timeout for ping). Restored in `finally` so subsequent calls keep
        # the long timeout. If the socket gets rebuilt by _init_socket on a
        # timeout/error path, the rebuilt socket already has self.timeout_ms.
        if timeout_ms_override is not None:
            self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms_override)
        try:
            try:
                self.socket.send(msgpack.packb(request))
                reply = self.socket.recv()
            except zmq.error.Again:
                # Timeout: REQ/REP socket state is unrecoverable — must rebuild
                # before the next call, otherwise next send() raises EFSM.
                self._init_socket()
                effective_timeout = (
                    timeout_ms_override
                    if timeout_ms_override is not None
                    else self.timeout_ms
                )
                raise TimeoutError(
                    f"Collector server did not reply within "
                    f"{effective_timeout / 1000:.0f}s "
                    f"(endpoint={endpoint!r})"
                )
            except zmq.error.ZMQError:
                self._init_socket()
                raise
        finally:
            # Restore the long timeout if we overrode it and the socket is
            # still alive. If _init_socket ran above, this is a harmless
            # no-op (the new socket already has self.timeout_ms set).
            if timeout_ms_override is not None and self.socket is not None:
                try:
                    self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                except zmq.error.ZMQError:
                    pass

        response = msgpack.unpackb(reply, raw=False)
        if isinstance(response, dict) and "error" in response:
            err = response["error"]
            if response.get("fatal"):
                # Won't change on retry — let the trainer abort immediately
                # rather than burn its consecutive-failure budget.
                raise FatalCollectorError(err)
            raise RuntimeError(f"Collector server error: {err}")
        return response

    def close(self):
        try:
            if self.socket is not None:
                self.socket.close(linger=0)
        except Exception:
            pass
        try:
            self.context.term()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-running episode collector server (per-iter startup cost = 0).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-names", nargs="+", required=True,
        help="One or more gymnasium env IDs. One EpisodeCollector is pre-initialized per env.",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, nargs="+", required=True,
        help="One value per --env-names entry (matched 1:1).",
    )
    parser.add_argument(
        "--group-size", type=int, default=5,
        help="Rollouts per group (G). Same for all envs. MUST match trainer config.",
    )
    parser.add_argument(
        "--n-action-steps", type=int, default=8,
        help="Substeps per action chunk. MUST match trainer config.",
    )
    parser.add_argument(
        "--policy-server-host", type=str, default="127.0.0.1",
        help="GR00T model server hostname (the trainer's in-process server, OR "
        "scripts/grpo/grpo_server.py if running standalone).",
    )
    parser.add_argument(
        "--policy-server-port", type=int, default=5555,
        help="GR00T model server port.",
    )
    parser.add_argument(
        "--listen-port", type=int, default=5556,
        help="Port this collector server listens on for trainer requests.",
    )
    parser.add_argument(
        "--debug-fast-forward", action="store_true",
        help="Enable verification-image rendering for FF (writes to per-request output_dir/debug_ff/).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    server = CollectorServer(
        env_names=args.env_names,
        max_episode_steps=args.max_episode_steps,
        group_size=args.group_size,
        n_action_steps=args.n_action_steps,
        policy_server_host=args.policy_server_host,
        policy_server_port=args.policy_server_port,
        listen_port=args.listen_port,
        debug_fast_forward=args.debug_fast_forward,
    )

    # SIGTERM handler: just flip the running flag; the run loop polls every
    # _RECV_POLL_MS and will exit cleanly through the finally: shutdown()
    # below (which closes the AsyncVectorEnv worker subprocesses). Without
    # this, kill -TERM would orphan G×N MuJoCo workers as zombies.
    # SIGINT is left untouched so KeyboardInterrupt still works.
    def _on_term(signum, frame):
        print(
            f"\n[collector_server] Received signal {signum} — shutting down",
            file=sys.stderr,
        )
        server.running = False

    signal.signal(signal.SIGTERM, _on_term)

    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[collector_server] KeyboardInterrupt — shutting down")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
