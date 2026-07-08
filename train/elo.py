"""Automatic Elo milestones: watch the checkpoint directory and pit
every Nth checkpoint against fixed anchors, appending to elo.log.

    uv run python -m train.elo --dir checkpoints \
        --anchor checkpoints/run1/iter_0059.pt --every 25

Runs alongside training (it shares the GPU briefly per milestone).
Already-evaluated iterations are skipped by parsing elo.log, so the
watcher can be restarted freely.
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch

from goboard import Stone

from train.arena import load_evaluator, play_match
from train.net import default_device


def evaluated_iters(log: Path) -> set[int]:
    if not log.exists():
        return set()
    return {int(m) for m in re.findall(r"^iter (\d+)", log.read_text(),
                                       re.MULTILINE)}


def match(ckpt_a: Path, ckpt_b: Path, games: int, simulations: int,
          komi: float, device, seed: int) -> int:
    rng = np.random.default_rng(seed)
    eval_a, config = load_evaluator(ckpt_a, device)
    eval_b, _ = load_evaluator(ckpt_b, device)
    wins = 0
    for game in range(games):
        a_is_black = game % 2 == 0
        evaluators = {
            Stone.BLACK: eval_a if a_is_black else eval_b,
            Stone.WHITE: eval_b if a_is_black else eval_a,
        }
        margin = play_match(evaluators, config["board_size"], komi,
                            simulations, temperature_moves=6, rng=rng)
        wins += (margin > 0) == a_is_black
    return wins


def elo_diff(wins: int, games: int) -> str:
    if wins == 0 or wins == games:
        return ">=+400" if wins else "<=-400"
    return f"{400 * np.log10(wins / (games - wins)):+.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--anchor", type=Path,
                        default=Path("checkpoints/run1/iter_0059.pt"))
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--komi", type=float, default=7.0)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    log = args.dir / "elo.log"
    device = default_device()
    while True:
        done = evaluated_iters(log)
        pending = []
        for path in args.dir.glob("iter_*.pt"):
            iteration = int(path.stem.split("_")[1])
            if iteration % args.every == 0 and iteration not in done:
                pending.append((iteration, path))
        for iteration, path in sorted(pending):
            start = time.time()
            wins = match(path, args.anchor, args.games, args.simulations,
                         args.komi, device, seed=iteration)
            line = (f"iter {iteration} vs {args.anchor.stem}: "
                    f"{wins}/{args.games} (elo {elo_diff(wins, args.games)}) "
                    f"[{time.time() - start:.0f}s]")
            with open(log, "a") as f:
                f.write(line + "\n")
            print(line, flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
