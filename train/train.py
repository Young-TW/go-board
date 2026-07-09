"""AlphaZero training loop: self-play -> replay buffer -> SGD.

Run from the repo root, e.g.:
    uv run python -m train.train --board-size 9 --iterations 50
"""

import argparse
import os
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import goboard
from goboard import Board, Stone

from train.batchplay import play_games
from train.net import PolicyValueNet, default_device, load_checkpoint
from train.selfplay import NetEvaluator, apply_random_symmetries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iter", type=int, default=128)
    parser.add_argument("--simulations", type=int, default=480,
                        help="full search budget per move")
    parser.add_argument("--cheap-simulations", type=int, default=96)
    parser.add_argument("--full-search-prob", type=float, default=0.25)
    parser.add_argument("--temperature-moves", type=int, default=8)
    parser.add_argument("--leaves-per-game", type=int, default=4)
    parser.add_argument("--noise-fraction", type=float, default=0.25)
    parser.add_argument("--resign-threshold", type=float, default=0.95,
                        help="side resigns below -threshold; >=1 disables")
    parser.add_argument("--no-resign-fraction", type=float, default=0.1)
    parser.add_argument("--dynamic-komi", action="store_true",
                        help="track the average scored margin with an EMA "
                             "and use it as the self-play base komi, so "
                             "one-sided phases keep producing 50/50 "
                             "outcome labels")
    parser.add_argument("--min-pass-moves", type=int, default=0,
                        help="self-play searches exclude pass before this "
                             "move number (guards the early pass-out "
                             "equilibrium)")
    parser.add_argument("--komi-jitter", type=float, default=0.0,
                        help="randomize each self-play game's komi by "
                             "+/- this many points (half-point steps)")
    parser.add_argument("--resign-fp-limit", type=float, default=0.05,
                        help="enable resignation only while the measured "
                             "false-positive rate stays below this")
    parser.add_argument("--parallel-games", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true",
                        help="disable torch.compile for self-play inference")
    parser.add_argument("--workers", type=int, default=0,
                        help="self-play worker processes, one per GPU "
                             "(0 = single-process on the training GPU)")
    parser.add_argument("--buffer-size", type=int, default=600_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps-per-iter", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min-factor", type=float, default=0.05,
                        help="cosine-decay floor as a fraction of --lr")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


STOP_REQUESTED = False


def _request_stop(signum, frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"signal {signum}: will save and exit after this iteration "
          "(SIGSTOP/SIGCONT pause instantly instead)", flush=True)


def save_buffer(buffer, path: Path) -> None:
    np.savez(path,
             planes=np.stack([entry[0] for entry in buffer]),
             pi=np.stack([entry[1] for entry in buffer]),
             z=np.array([entry[2] for entry in buffer], dtype=np.float32),
             w_pi=np.array([entry[3] for entry in buffer],
                           dtype=np.float32),
             ownership=np.stack([entry[4] for entry in buffer]),
             score=np.array([entry[5] for entry in buffer],
                            dtype=np.float32),
             w_own=np.array([entry[6] for entry in buffer],
                            dtype=np.float32))


def load_buffer(path: Path, buffer, board_size: int) -> bool:
    data = np.load(path)
    if data["planes"].shape[-1] != board_size:
        print(f"ignoring {path}: board size mismatch", flush=True)
        return False
    if data["planes"].shape[1] != goboard.FEATURE_PLANES:
        print(f"ignoring {path}: feature plane count mismatch",
              flush=True)
        return False
    if "w_pi" not in data or "w_own" not in data:
        print(f"ignoring {path}: old buffer format", flush=True)
        return False
    for entry in zip(data["planes"], data["pi"], data["z"], data["w_pi"],
                     data["ownership"], data["score"], data["w_own"]):
        buffer.append((entry[0], entry[1], float(entry[2]),
                       float(entry[3]), entry[4], float(entry[5]),
                       float(entry[6])))
    return True


OWNERSHIP_WEIGHT = 0.15
SCORE_WEIGHT = 0.05


def mark_spectate_training(path: Path, iteration: int) -> None:
    """Replace the spectate header so the watcher shows the training
    phase instead of a frozen final position."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    board = lines[1:] if len(lines) > 1 else []
    header = f"training iteration {iteration} (self-play paused)"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join([header, *board]) + "\n")
    os.replace(tmp, path)


def train_steps(net, buffer, optimizer, device, batch_size, steps,
                rng) -> tuple[float, float, float]:
    net.train()
    # Random indexing into a deque is O(n); snapshot it once.
    buffer = list(buffer)
    policy_losses = []
    value_losses = []
    ownership_losses = []
    for _ in range(steps):
        indices = rng.choice(len(buffer), size=batch_size)
        raw_planes = np.stack([buffer[i][0]
                               for i in indices])[:, :net.in_planes]
        raw_pi = np.stack([buffer[i][1] for i in indices])
        raw_own = np.stack([buffer[i][4] for i in indices])
        # Train-time augmentation: the buffer holds each position
        # once; every draw gets an independent dihedral transform.
        sym_planes, sym_pi, sym_own = apply_random_symmetries(
            raw_planes, raw_pi, raw_own, rng)
        planes = torch.from_numpy(
            np.ascontiguousarray(sym_planes)).to(device)
        target_pi = torch.from_numpy(
            np.ascontiguousarray(sym_pi)).to(device)
        target_z = torch.tensor(
            [buffer[i][2] for i in indices], dtype=torch.float32,
            device=device)

        w_pi = torch.tensor([buffer[i][3] for i in indices],
                            dtype=torch.float32, device=device)
        target_own = torch.from_numpy(
            np.ascontiguousarray(sym_own)).to(device)
        target_score = torch.tensor(
            [buffer[i][5] for i in indices], dtype=torch.float32,
            device=device) / net.board_size
        w_own = torch.tensor([buffer[i][6] for i in indices],
                             dtype=torch.float32, device=device)

        # bf16 keeps fp32 master weights and needs no grad scaler.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits, value, ownership, score = net(planes)
            # Playout cap randomization: cheap-search moves carry no
            # policy target, so their weight is zero. Resigned games
            # have no final position, so their ownership/score weight
            # is zero.
            per_sample = -(target_pi * F.log_softmax(logits, dim=1)).sum(1)
            policy_loss = (per_sample * w_pi).sum() / \
                w_pi.sum().clamp(min=1.0)
            value_loss = F.mse_loss(value, target_z)
            w_own_total = w_own.sum().clamp(min=1.0)
            ownership_loss = ((ownership - target_own).pow(2).mean(dim=1)
                              * w_own).sum() / w_own_total
            score_loss = ((score - target_score).pow(2)
                          * w_own).sum() / w_own_total
            loss = (policy_loss + value_loss
                    + OWNERSHIP_WEIGHT * ownership_loss
                    + SCORE_WEIGHT * score_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        ownership_losses.append(ownership_loss.item())
    return (float(np.mean(policy_losses)), float(np.mean(value_losses)),
            float(np.mean(ownership_losses)))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else default_device()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    state = None
    start_iter = 0
    upgraded_planes = False
    in_planes = goboard.FEATURE_PLANES
    if args.resume is not None:
        state = load_checkpoint(args.resume)
        start_iter = state["iteration"] + 1
        ckpt_planes = state["config"].get("in_planes", 3)
        if ckpt_planes != in_planes:
            upgraded_planes = True
            # Upgrade across feature-plane growth: copy the stem
            # kernels for the old planes and zero-init the new ones,
            # so the upgraded net starts out functionally identical.
            old = state["model"]["stem.0.weight"]
            new = torch.zeros(old.shape[0], in_planes, *old.shape[2:])
            new[:, :ckpt_planes] = old
            state["model"]["stem.0.weight"] = new
            print(f"upgraded stem from {ckpt_planes} to {in_planes} "
                  "feature planes (new planes zero-initialized)",
                  flush=True)
    net = PolicyValueNet(args.board_size, args.channels, args.blocks,
                         in_planes)
    if state is not None:
        try:
            net.load_state_dict(state["model"])
        except RuntimeError:
            # Warm start across architecture growth: shared parts load,
            # new heads keep their fresh initialization.
            missing, _ = net.load_state_dict(state["model"], strict=False)
            print(f"warm start: {len(missing)} new parameters "
                  "initialized fresh", flush=True)
    net.to(device)
    # Create the optimizer only once the model is on its device, so a
    # resumed optimizer state is cast onto the parameters' device too.
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    if state is not None:
        # A plane upgrade reshapes the stem parameter; the saved
        # optimizer moments no longer match it (load_state_dict does
        # not validate shapes — it crashes at the first step).
        if upgraded_planes:
            print("optimizer state reset after feature plane upgrade",
                  flush=True)
        else:
            try:
                optimizer.load_state_dict(state["optimizer"])
            except (RuntimeError, ValueError):
                print("optimizer state incompatible after architecture "
                      "change; starting it fresh", flush=True)
        print(f"resumed from {args.resume} at iteration {start_iter}",
              flush=True)

    buffer: deque = deque(maxlen=args.buffer_size)
    # Resignation activates only once the measured false-positive rate
    # over a rolling window is low: an immature value net + resignation
    # is a self-reinforcing death spiral (learned the hard way, twice).
    calibration_window: deque = deque(maxlen=20)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    buffer_path = args.checkpoint_dir / "buffer.npz"
    if args.resume is not None and buffer_path.exists():
        if load_buffer(buffer_path, buffer, args.board_size):
            print(f"restored replay buffer ({len(buffer)} samples)",
                  flush=True)
    evaluator = NetEvaluator(net, device, compile=not args.no_compile)
    workers = None
    if args.workers > 0:
        from train.workers import SelfPlayWorkers
        workers = SelfPlayWorkers(
            args.workers,
            dict(board_size=args.board_size, channels=args.channels,
                 blocks=args.blocks, in_planes=net.in_planes))

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    checkpoint = args.resume
    resign_active = False
    # Dynamic komi curriculum state; restored on resume.
    komi_ema = args.komi
    if state is not None and "komi_ema" in state:
        komi_ema = state["komi_ema"]
    for iteration in range(start_iter, args.iterations):
        # Cosine decay from --lr to its floor across the whole run;
        # recomputed from the iteration number, so it is resume-safe.
        progress = iteration / max(1, args.iterations - 1)
        # float(): a numpy scalar here leaks into the optimizer state
        # and poisons the checkpoint for weights_only torch.load.
        lr = float(args.lr * (args.lr_min_factor
                              + (1 - args.lr_min_factor)
                              * 0.5 * (1 + np.cos(np.pi * progress))))
        for group in optimizer.param_groups:
            group["lr"] = lr

        start = time.time()
        net.eval()  # train_steps leaves the net in train mode
        komi_base = args.komi
        if args.dynamic_komi:
            # Round to a half point; never below the real komi.
            komi_base = max(args.komi, round(komi_ema * 2) / 2)
        play_kwargs = dict(
            board_size=args.board_size, komi=komi_base,
            simulations=args.simulations,
            cheap_simulations=args.cheap_simulations,
            full_search_prob=args.full_search_prob,
            temperature_moves=args.temperature_moves,
            leaves_per_game=args.leaves_per_game,
            parallel=args.parallel_games,
            noise_fraction=args.noise_fraction,
            resign_threshold=(args.resign_threshold if resign_active
                              else 2.0),
            no_resign_fraction=args.no_resign_fraction,
            komi_jitter=args.komi_jitter,
            min_pass_moves=args.min_pass_moves,
            spectate_path=args.checkpoint_dir / "spectate.txt")
        if workers is not None:
            game_samples, margins, stats = workers.play(
                net, args.games_per_iter, play_kwargs, rng)
        else:
            game_samples, margins, stats = play_games(
                evaluator.evaluate_planes, args.games_per_iter,
                rng=rng, **play_kwargs)
        calibration_window.append(
            (stats["resign_false_positives"],
             stats["resign_calibration_games"]))
        fp = sum(f for f, _ in calibration_window)
        observed = sum(g for _, g in calibration_window)
        # The fp gate alone cannot catch a self-consistent delusion
        # (the would-resigner really does lose when both sides believe
        # it), so also require the value net to be sane about the one
        # position with a known answer: the empty board is close to
        # even, and both collapses showed |v(empty)| ~ 0.9.
        empty = Board(size=args.board_size, komi=args.komi)
        net.eval()
        v_empty = max(abs(evaluator(empty, Stone.BLACK)[1]),
                      abs(evaluator(empty, Stone.WHITE)[1]))
        resign_active = (observed >= 30
                         and fp / observed < args.resign_fp_limit
                         and v_empty < 0.6)
        black_wins = sum(margin > 0 for margin in margins)
        moves = sum(len(samples) for samples in game_samples)
        # Track the raw board margin from scored games only (margins
        # are komi-adjusted; add this iteration's komi back). The
        # median is by definition the komi that splits outcomes 50/50;
        # the mean lagged badly when the leader's margin kept growing
        # (observed: black climbed back to 78% wins while a mean-EMA
        # chased 10+ points behind).
        scored = [margin + komi_base for margin in margins
                  if abs(margin) < 10000.0]
        if args.dynamic_komi and scored:
            komi_ema = 0.5 * komi_ema + 0.5 * float(np.median(scored))
        for samples in game_samples:
            for sample in samples:
                buffer.append((sample.planes, sample.pi, sample.z,
                               sample.train_pi, sample.ownership,
                               sample.score, sample.w_own))
        selfplay_time = time.time() - start

        mark_spectate_training(args.checkpoint_dir / "spectate.txt",
                               iteration)
        start = time.time()
        policy_loss, value_loss, ownership_loss = train_steps(
            net, buffer, optimizer, device, args.batch_size,
            args.steps_per_iter, rng)
        train_time = time.time() - start

        checkpoint = args.checkpoint_dir / f"iter_{iteration:04d}.pt"
        torch.save({
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "komi_ema": float(komi_ema),
            "config": vars(args) | {"checkpoint_dir": str(args.checkpoint_dir),
                                    "resume": None,
                                    "in_planes": net.in_planes,
                                    "arch": net.ARCH},
        }, checkpoint)

        line = (f"iter {iteration:4d} | "
                f"buffer {len(buffer):7d} | "
                f"B wins {black_wins}/{args.games_per_iter} | "
                f"avg moves {moves / args.games_per_iter:5.1f} | "
                f"p-loss {policy_loss:.4f} | v-loss {value_loss:.4f} | "
                f"o-loss {ownership_loss:.4f} | "
                f"resign {'on' if resign_active else 'off'} "
                f"fp {stats['resign_false_positives']}"
                f"/{stats['resign_calibration_games']} | "
                f"cache {stats['eval_cache_hits'] / max(1, stats['eval_cache_lookups']):.0%} | "
                f"komi {komi_base:.1f} | lr {lr:.1e} | "
                f"selfplay {selfplay_time:5.1f}s train {train_time:5.1f}s")
        print(line, flush=True)
        # Restart-proof history: stdout redirection truncates on every
        # relaunch, which wipes the spectator's loss curves.
        with open(args.checkpoint_dir / "history.log", "a") as history:
            history.write(line + "\n")

        if STOP_REQUESTED:
            break

    if workers is not None:
        workers.close()
    save_buffer(buffer, buffer_path)
    print(f"saved replay buffer ({len(buffer)} samples); resume with "
          f"--resume {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
