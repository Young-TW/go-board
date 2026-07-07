"""AlphaZero training loop: self-play -> replay buffer -> SGD.

Run from the repo root, e.g.:
    uv run python -m train.train --board-size 9 --iterations 50
"""

import argparse
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import goboard

from train.batchplay import play_games
from train.net import PolicyValueNet, default_device
from train.selfplay import NetEvaluator, symmetries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games-per-iter", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=480,
                        help="full search budget per move")
    parser.add_argument("--cheap-simulations", type=int, default=96)
    parser.add_argument("--full-search-prob", type=float, default=0.25)
    parser.add_argument("--temperature-moves", type=int, default=8)
    parser.add_argument("--leaves-per-game", type=int, default=4)
    parser.add_argument("--noise-fraction", type=float, default=0.25)
    parser.add_argument("--parallel-games", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true",
                        help="disable torch.compile for self-play inference")
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps-per-iter", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
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
                            dtype=np.float32))


def load_buffer(path: Path, buffer, board_size: int) -> bool:
    data = np.load(path)
    if data["planes"].shape[-1] != board_size:
        print(f"ignoring {path}: board size mismatch", flush=True)
        return False
    if "w_pi" not in data or "ownership" not in data:
        print(f"ignoring {path}: old buffer format", flush=True)
        return False
    for entry in zip(data["planes"], data["pi"], data["z"], data["w_pi"],
                     data["ownership"], data["score"]):
        buffer.append((entry[0], entry[1], float(entry[2]),
                       float(entry[3]), entry[4], float(entry[5])))
    return True


OWNERSHIP_WEIGHT = 0.15
SCORE_WEIGHT = 0.05


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
        planes = torch.from_numpy(
            np.stack([buffer[i][0]
                      for i in indices])[:, :net.in_planes]).to(device)
        target_pi = torch.from_numpy(
            np.stack([buffer[i][1] for i in indices])).to(device)
        target_z = torch.tensor(
            [buffer[i][2] for i in indices], dtype=torch.float32,
            device=device)

        w_pi = torch.tensor([buffer[i][3] for i in indices],
                            dtype=torch.float32, device=device)
        target_own = torch.from_numpy(
            np.stack([buffer[i][4] for i in indices])).to(device)
        target_score = torch.tensor(
            [buffer[i][5] for i in indices], dtype=torch.float32,
            device=device) / net.board_size

        logits, value, ownership, score = net(planes)
        # Playout cap randomization: cheap-search moves carry no
        # policy target, so their weight is zero.
        per_sample = -(target_pi * F.log_softmax(logits, dim=1)).sum(1)
        policy_loss = (per_sample * w_pi).sum() / w_pi.sum().clamp(min=1.0)
        value_loss = F.mse_loss(value, target_z)
        ownership_loss = F.mse_loss(ownership, target_own)
        score_loss = F.mse_loss(score, target_score)
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
    in_planes = goboard.FEATURE_PLANES
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        in_planes = state["config"].get("in_planes", 3)
        start_iter = state["iteration"] + 1
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
        try:
            optimizer.load_state_dict(state["optimizer"])
        except (RuntimeError, ValueError):
            print("optimizer state incompatible after architecture "
                  "change; starting it fresh", flush=True)
        print(f"resumed from {args.resume} at iteration {start_iter}",
              flush=True)

    buffer: deque = deque(maxlen=args.buffer_size)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    buffer_path = args.checkpoint_dir / "buffer.npz"
    if args.resume is not None and buffer_path.exists():
        if load_buffer(buffer_path, buffer, args.board_size):
            print(f"restored replay buffer ({len(buffer)} samples)",
                  flush=True)
    evaluator = NetEvaluator(net, device, compile=not args.no_compile)

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    checkpoint = args.resume
    for iteration in range(start_iter, args.iterations):
        start = time.time()
        net.eval()  # train_steps leaves the net in train mode
        game_samples, margins = play_games(
            evaluator.evaluate_planes, args.games_per_iter,
            board_size=args.board_size, komi=args.komi,
            simulations=args.simulations,
            cheap_simulations=args.cheap_simulations,
            full_search_prob=args.full_search_prob,
            temperature_moves=args.temperature_moves,
            leaves_per_game=args.leaves_per_game,
            parallel=args.parallel_games,
            noise_fraction=args.noise_fraction, rng=rng,
            spectate_path=args.checkpoint_dir / "spectate.txt")
        black_wins = sum(margin > 0 for margin in margins)
        moves = sum(len(samples) for samples in game_samples)
        for samples in game_samples:
            for sample in samples:
                for planes, pi, ownership in symmetries(
                        sample.planes, sample.pi, args.board_size,
                        sample.ownership):
                    buffer.append((planes, pi, sample.z, sample.train_pi,
                                   ownership, sample.score))
        selfplay_time = time.time() - start

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
            "config": vars(args) | {"checkpoint_dir": str(args.checkpoint_dir),
                                    "resume": None,
                                    "in_planes": net.in_planes},
        }, checkpoint)

        print(f"iter {iteration:4d} | "
              f"buffer {len(buffer):7d} | "
              f"B wins {black_wins}/{args.games_per_iter} | "
              f"avg moves {moves / args.games_per_iter:5.1f} | "
              f"p-loss {policy_loss:.4f} | v-loss {value_loss:.4f} | "
              f"o-loss {ownership_loss:.4f} | "
              f"selfplay {selfplay_time:5.1f}s train {train_time:5.1f}s",
              flush=True)

        if STOP_REQUESTED:
            break

    save_buffer(buffer, buffer_path)
    print(f"saved replay buffer ({len(buffer)} samples); resume with "
          f"--resume {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
