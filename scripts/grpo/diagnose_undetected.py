"""Classify the failing episodes the post-reopen detector found no cycle in.

Usage: python diagnose_undetected.py <episode_dir> [--close 0.035] [--open 0.055] [--dwell 3]
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.environ.get("GRPO_DIR", "scripts/grpo"))
from gripper_release import gripper_widths, close_cross_indices, reopen_onset_index

ap = argparse.ArgumentParser()
ap.add_argument("episode_dir")
ap.add_argument("--close", type=float, default=0.035)
ap.add_argument("--open", dest="open_", type=float, default=0.055)
ap.add_argument("--dwell", type=int, default=3)
ap.add_argument("--show-all", action="store_true")
a = ap.parse_args()

rows = []
for p in sorted(glob.glob(os.path.join(a.episode_dir, "episode_*.npz"))):
    d = np.load(p, allow_pickle=True)
    nc = int(d["num_chunks"])
    w = gripper_widths([{"gripper_qpos": d[f"state_gripper_qpos_{i}"]} for i in range(nc)])
    rows.append((os.path.basename(p), bool(d["success"]), nc, w))

fails = [r for r in rows if not r[1]]
undet = []
for nm, _, nc, w in fails:
    if close_cross_indices(w, a.close, a.open_, a.dwell) is None:
        undet.append((nm, nc, w))

print(f"{len(fails)} failing episodes, {len(fails)-len(undet)} detected, {len(undet)} NOT detected\n")
for nm, nc, w in undet:
    seen_open = bool((w > a.open_).any())
    # longest run of consecutive sub-close_below samples, and where it starts
    best_len = best_at = 0; cur = 0
    for i, x in enumerate(w):
        if x < a.close:
            cur += 1
            if cur > best_len: best_len, best_at = cur, i - cur + 1
        else:
            cur = 0
    closed_at_end = w[-1] < a.close
    # would a shorter dwell, or a laxer band, find a cycle?
    fix = []
    for dw in (1, 2):
        if dw < a.dwell and close_cross_indices(w, a.close, a.open_, dw) is not None:
            fix.append(f"dwell={dw}"); break
    for cb in (0.040, 0.045, 0.050):
        if close_cross_indices(w, cb, a.open_, a.dwell) is not None:
            fix.append(f"close={cb}"); break
    for oa in (0.050, 0.045, 0.040):
        if oa > a.close and close_cross_indices(w, a.close, oa, a.dwell) is not None:
            fix.append(f"open={oa}"); break

    if best_len == 0:
        cause = "NEVER CLOSED (approach never reached a grasp)"
    elif closed_at_end:
        cause = f"CLOSED AT END, held from ~{best_at} (grasped, never placed)"
    elif best_len < a.dwell:
        cause = f"CLOSE TOO BRIEF ({best_len} < dwell {a.dwell}) at ~{best_at}"
    elif not seen_open:
        cause = "NEVER OBSERVED OPEN (started closed?)"
    else:
        cause = f"CLOSED at ~{best_at} but reopen never exceeded open={a.open_}"
    print(f"  {nm}  nc={nc:3d}  w[min={w.min():.4f} max={w.max():.4f} last={w[-1]:.4f}]  "
          f"longest_closed_run={best_len:2d}  {cause}"
          + (f"   [would detect with {', '.join(fix)}]" if fix else ""))
