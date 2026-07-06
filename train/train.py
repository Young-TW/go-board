"""AlphaZero training loop: self-play -> replay buffer -> SGD.

Run from the repo root, e.g.:
    uv run python -m train.train --board-size 9 --iterations 50
"""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--temperature-moves", type=int, default=8)
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


def train_steps(net, buffer, optimizer, device, batch_size, steps,
                rng) -> tuple[float, float]:
    net.train()
    # Random indexing into a deque is O(n); snapshot it once.
    buffer = list(buffer)
    policy_losses = []
    value_losses = []
    for _ in range(steps):
        indices = rng.choice(len(buffer), size=batch_size)
        planes = torch.from_numpy(
            np.stack([buffer[i][0] for i in indices])).to(device)
        target_pi = torch.from_numpy(
            np.stack([buffer[i][1] for i in indices])).to(device)
        target_z = torch.tensor(
            [buffer[i][2] for i in indices], dtype=torch.float32,
            device=device)

        logits, value = net(planes)
        policy_loss = -(target_pi * F.log_softmax(logits, dim=1)).sum(1).mean()
        value_loss = F.mse_loss(value, target_z)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else default_device()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    net = PolicyValueNet(args.board_size, args.channels, args.blocks)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    start_iter = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iteration"] + 1
        print(f"resumed from {args.resume} at iteration {start_iter}")
    net.to(device)

    buffer: deque = deque(maxlen=args.buffer_size)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(start_iter, args.iterations):
        start = time.time()
        evaluator = NetEvaluator(net, device)
        game_samples, margins = play_games(
            evaluator.evaluate_batch, args.games_per_iter,
            board_size=args.board_size, komi=args.komi,
            simulations=args.simulations,
            temperature_moves=args.temperature_moves, rng=rng)
        black_wins = sum(margin > 0 for margin in margins)
        moves = sum(len(samples) for samples in game_samples)
        for samples in game_samples:
            for sample in samples:
                for planes, pi in symmetries(sample.planes, sample.pi,
                                             args.board_size):
                    buffer.append((planes, pi, sample.z))
        selfplay_time = time.time() - start

        start = time.time()
        policy_loss, value_loss = train_steps(
            net, buffer, optimizer, device, args.batch_size,
            args.steps_per_iter, rng)
        train_time = time.time() - start

        checkpoint = args.checkpoint_dir / f"iter_{iteration:04d}.pt"
        torch.save({
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "config": vars(args) | {"checkpoint_dir": str(args.checkpoint_dir),
                                    "resume": None},
        }, checkpoint)

        print(f"iter {iteration:4d} | "
              f"buffer {len(buffer):7d} | "
              f"B wins {black_wins}/{args.games_per_iter} | "
              f"avg moves {moves / args.games_per_iter:5.1f} | "
              f"p-loss {policy_loss:.4f} | v-loss {value_loss:.4f} | "
              f"selfplay {selfplay_time:5.1f}s train {train_time:5.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
