"""Smoke test for the goboard Python bindings.

Run with the built module on PYTHONPATH, e.g.:
    PYTHONPATH=build python python/test_goboard.py
"""

import numpy as np

import goboard
from goboard import Board, Stone


def main() -> None:
    board = Board(size=9, komi=7.5)
    assert board.size == 9

    # Basic play and capture: white (0,0) loses its last liberty.
    assert board.play(0, 0, Stone.WHITE)
    assert board.play(1, 0, Stone.BLACK)
    assert board.play(0, 1, Stone.BLACK)
    assert board.at(0, 0) == Stone.EMPTY

    # Illegal moves are rejected without changing the board.
    assert not board.play(1, 0, Stone.WHITE)
    assert not board.is_legal(1, 0, Stone.WHITE)

    # Two black stones occupied; (0,0) is suicide for white.
    assert len(board.legal_moves(Stone.WHITE)) == 9 * 9 - 3
    assert len(board.legal_moves(Stone.BLACK)) == 9 * 9 - 2

    planes = board.features(Stone.BLACK)
    assert planes.shape == (goboard.FEATURE_PLANES, 9, 9)
    assert planes.dtype == np.float32
    assert planes[0, 0, 1] == 1.0  # black stone at x=1, y=0
    assert planes[2].min() == 1.0  # black to play
    assert board.features(Stone.WHITE)[2].max() == 0.0

    # Pass twice to end the game; black owns the whole board.
    board.pass_()
    board.pass_()
    assert board.is_terminal()
    assert board.score() == 81 - 7.5

    assert goboard.opponent(Stone.BLACK) == Stone.WHITE
    assert "X" in str(board)

    print("goboard bindings OK")


if __name__ == "__main__":
    main()
