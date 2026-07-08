"""Pit two checkpoints against each other to measure progress.

    uv run python -m train.arena checkpoints/iter_0019.pt \
        checkpoints/iter_0059.pt --games 20

Colors alternate between games; a little early-move temperature varies
the openings so games are not all identical.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import goboard
from goboard import Board, Stone

from train.mcts import MCTS, apply_move
from train.net import PolicyValueNet, default_device
from train.selfplay import NetEvaluator
from train.sgf import margin_to_result, save_sgf


def load_evaluator(path: Path, device) -> tuple[NetEvaluator, dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state["config"]
    net = PolicyValueNet(config["board_size"], config["channels"],
                         config["blocks"],
                         in_planes=config.get("in_planes", 3))
    try:
        net.load_state_dict(state["model"])
    except RuntimeError:
        # Checkpoints predating the auxiliary heads: shared parts load,
        # the unused new heads keep their fresh initialization.
        net.load_state_dict(state["model"], strict=False)
    return NetEvaluator(net, device), config


def play_match(evaluators: dict, board_size: int, komi: float,
               simulations: int, temperature_moves: int,
               rng: np.random.Generator,
               moves_out: list | None = None) -> float:
    """One game; evaluators maps Stone.BLACK/WHITE to an evaluator.
    Returns the black margin; appends (color, point|None) records to
    moves_out when given (for SGF export)."""
    searchers = {color: MCTS(evaluate, rng=rng)
                 for color, evaluate in evaluators.items()}
    board = Board(size=board_size, komi=komi)
    to_play = Stone.BLACK
    max_moves = board_size * board_size * 2
    for move_count in range(max_moves):
        if board.is_terminal():
            break
        mcts = searchers[to_play]
        root = mcts.run(board, to_play, simulations, add_noise=False)
        temperature = 0.25 if move_count < temperature_moves else 0.0
        move = mcts.select_move(root, board_size, temperature)
        if moves_out is not None:
            point = (None if move == board_size * board_size
                     else (move % board_size, move // board_size))
            moves_out.append((to_play, point))
        apply_move(board, move, to_play)
        to_play = goboard.opponent(to_play)
    return board.score()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpt_a", type=Path)
    parser.add_argument("ckpt_b", type=Path)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument("--temperature-moves", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sgf-dir", type=Path, default=None,
                        help="save each game as an SGF file here")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else default_device()
    rng = np.random.default_rng(args.seed)
    eval_a, config = load_evaluator(args.ckpt_a, device)
    eval_b, _ = load_evaluator(args.ckpt_b, device)
    board_size = config["board_size"]

    wins_a = 0
    for game in range(args.games):
        a_is_black = game % 2 == 0
        evaluators = {
            Stone.BLACK: eval_a if a_is_black else eval_b,
            Stone.WHITE: eval_b if a_is_black else eval_a,
        }
        moves: list = []
        black_margin = play_match(evaluators, board_size, args.komi,
                                  args.simulations, args.temperature_moves,
                                  rng, moves_out=moves)
        if args.sgf_dir is not None:
            black = (args.ckpt_a if a_is_black else args.ckpt_b).stem
            white = (args.ckpt_b if a_is_black else args.ckpt_a).stem
            save_sgf(args.sgf_dir / f"game_{game + 1:03d}.sgf", moves,
                     board_size, args.komi, margin_to_result(black_margin),
                     black=black, white=white)
        a_won = (black_margin > 0) == a_is_black
        wins_a += a_won
        print(f"game {game + 1:3d}: {args.ckpt_a.name} as "
              f"{'B' if a_is_black else 'W'} "
              f"{'wins' if a_won else 'loses'} "
              f"(B{'+' if black_margin > 0 else ''}{black_margin})",
              flush=True)

    win_rate = wins_a / args.games
    print(f"\n{args.ckpt_a.name}: {wins_a}/{args.games} "
          f"({win_rate:.0%}) vs {args.ckpt_b.name}")
    if 0 < win_rate < 1:
        elo = 400 * np.log10(win_rate / (1 - win_rate))
        print(f"Elo difference: {elo:+.0f}")


if __name__ == "__main__":
    main()
