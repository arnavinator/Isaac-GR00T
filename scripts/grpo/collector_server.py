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
    # Terminal 1 (model venv) — start the GR00T model server
    uv run python gr00t/eval/run_gr00t_server.py \\
        --model-path nvidia/GR00T-N1.6-3B \\
        --embodiment-tag ROBOCASA_PANDA_OMRON \\
        --use-sim-policy-wrapper

    # Terminal 2 (sim venv) — start the long-running collector server
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \\
        scripts/grpo/collector_server.py \\
        --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
                    robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \\
        --max-episode-steps 720 480 \\
        --group-size 5 --n-action-steps 8 \\
        --policy-server-host 127.0.0.1 --policy-server-port 5555 \\
        --listen-port 5556

    # Terminal 3 (main venv) — trainer connects via CollectorClient
    # (set GrpoConfig.collector_server_host / collector_server_port).

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
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import msgpack
import zmq

from collect_episodes import EpisodeCollector, save_episodes


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class CollectorServer:
    """ZMQ REP server that holds one EpisodeCollector per env_name.

    Endpoints:
      - ping  → {"status": "ok", "envs": [...]}
      - collect (data: env_name, output_dir, base_seed, num_groups,
        success_weight, fast_forward_steps, fast_forward_pct)
        → {"n_episodes", "n_successes", "elapsed_s"}
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
        seed: int = 42,
        debug_fast_forward: bool = False,
    ):
        if len(env_names) != len(max_episode_steps):
            raise ValueError(
                f"--env-names ({len(env_names)}) and --max-episode-steps "
                f"({len(max_episode_steps)}) must have the same length."
            )

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
                seed=seed,
                debug_fast_forward=debug_fast_forward,
                output_dir="/tmp",  # overridden per-request
            )

        # ZMQ REP socket
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(f"tcp://*:{listen_port}")
        self.listen_port = listen_port
        self.running = True

        print(f"\n[collector_server] {len(self.collectors)} collector(s) ready.")
        print(f"[collector_server] Listening on tcp://*:{listen_port}")
        sys.stdout.flush()

    def run(self):
        while self.running:
            try:
                msg = self.sock.recv()
            except zmq.error.ContextTerminated:
                break
            try:
                request = msgpack.unpackb(msg, raw=False)
                endpoint = request.get("endpoint", "")

                if endpoint == "ping":
                    self.sock.send(
                        msgpack.packb({"status": "ok", "envs": list(self.collectors)})
                    )
                elif endpoint == "kill":
                    self.sock.send(msgpack.packb({"status": "ok"}))
                    self.running = False
                elif endpoint == "collect":
                    result = self._handle_collect(request.get("data", {}))
                    self.sock.send(msgpack.packb(result))
                else:
                    self.sock.send(
                        msgpack.packb({"error": f"unknown endpoint: {endpoint!r}"})
                    )
            except Exception as e:
                # Per-request error: log and reply with {"error": ...} so the
                # client can decide whether to retry or fail. Don't crash the
                # whole server (which would void the startup cost savings).
                traceback.print_exc()
                try:
                    self.sock.send(msgpack.packb({"error": f"{type(e).__name__}: {e}"}))
                except zmq.error.ZMQError:
                    pass

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
        episodes = collector.collect(
            num_groups=int(data["num_groups"]),
            base_seed=int(data["base_seed"]),
            success_weight=float(data.get("success_weight", 1.0)),
            fast_forward_steps=int(data.get("fast_forward_steps", 0)),
            fast_forward_pct=float(data.get("fast_forward_pct", 0.5)),
        )
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
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        timeout_ms: int = 1_800_000,  # 30 min: longer than any plausible group
    ):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> dict:
        return self._call("ping", requires_input=False)

    def collect(
        self,
        env_name: str,
        output_dir: str,
        base_seed: int,
        num_groups: int,
        success_weight: float = 1.0,
        fast_forward_steps: int = 0,
        fast_forward_pct: float = 0.5,
    ) -> dict:
        return self._call("collect", {
            "env_name": env_name,
            "output_dir": str(output_dir),
            "base_seed": int(base_seed),
            "num_groups": int(num_groups),
            "success_weight": float(success_weight),
            "fast_forward_steps": int(fast_forward_steps),
            "fast_forward_pct": float(fast_forward_pct),
        })

    def kill(self) -> dict:
        return self._call("kill", requires_input=False)

    def _call(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        try:
            self.socket.send(msgpack.packb(request))
            reply = self.socket.recv()
        except zmq.error.Again:
            # Timeout: socket state is unrecoverable for REQ/REP — must
            # rebuild it before the next call, otherwise the next send()
            # raises EFSM.
            self._init_socket()
            raise TimeoutError(
                f"Collector server did not reply within {self.timeout_ms / 1000:.0f}s "
                f"(endpoint={endpoint!r})"
            )
        except zmq.error.ZMQError:
            self._init_socket()
            raise

        response = msgpack.unpackb(reply, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Collector server error: {response['error']}")
        return response

    def close(self):
        try:
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
        help="Rollouts per group (G). Same for all envs.",
    )
    parser.add_argument(
        "--n-action-steps", type=int, default=8,
        help="Substeps per action chunk.",
    )
    parser.add_argument(
        "--policy-server-host", type=str, default="127.0.0.1",
        help="GR00T model server (run_gr00t_server.py / grpo_server.py) hostname.",
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
        "--seed", type=int, default=42,
        help="Initial seed (overridden per-request via base_seed).",
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
        seed=args.seed,
        debug_fast_forward=args.debug_fast_forward,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[collector_server] KeyboardInterrupt — shutting down")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
