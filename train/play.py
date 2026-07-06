"""Play against a trained checkpoint in the terminal.

    uv run python -m train.play checkpoints/iter_0059.pt --human black

Enter moves as "x y" (1-based from the top-left), "pass", or "quit".
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import goboard
from goboard import Board, Stone

from train.arena import load_evaluator
from train.mcts import MCTS, apply_move


def read_human_move(board: Board, color: Stone) -> int | None:
    """Returns a flattened move, size*size for pass, or None to quit."""
    size = board.size
    while True:
        try:
            raw = input(f"{'black' if color == Stone.BLACK else 'white'}> ")
        except EOFError:
            return None
        token = raw.strip().lower()
        if token == "quit":
            return None
        if token == "pass":
            return size * size
        parts = token.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            x, y = int(parts[0]) - 1, int(parts[1]) - 1
            if board.is_legal(x, y, color):
                return y * size + x
        print("illegal move; enter \"x y\", \"pass\", or \"quit\"")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--human", choices=["black", "white"],
                        default="black")
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    evaluate, config = load_evaluator(args.checkpoint, device)
    board_size = config["board_size"]
    human = Stone.BLACK if args.human == "black" else Stone.WHITE
    mcts = MCTS(evaluate, rng=np.random.default_rng())

    board = Board(size=board_size, komi=args.komi)
    to_play = Stone.BLACK
    print(board)
    while not board.is_terminal():
        if to_play == human:
            move = read_human_move(board, to_play)
            if move is None:
                return
        else:
            root = mcts.run(board, to_play, args.simulations)
            move = mcts.select_move(root, board_size)
            if move == board_size * board_size:
                print("engine passes")
            else:
                print(f"engine plays {move % board_size + 1} "
                      f"{move // board_size + 1} "
                      f"(value {-root.value():+.2f} for you)")
        apply_move(board, move, to_play)
        to_play = goboard.opponent(to_play)
        print(board)

    margin = board.score()
    print(f"result: {'B+' if margin > 0 else 'W+'}{abs(margin)} "
          f"(Tromp-Taylor)")


if __name__ == "__main__":
    main()
