"""Curses TUI for playing against a trained checkpoint.

    uv run python -m train.tui checkpoints/iter_0059.pt --human black

Move the cursor with the arrow keys or hjkl (or click with the mouse),
place a stone with Enter/Space, "p" to pass, "q" to quit.
"""

import argparse
import curses
from pathlib import Path

import numpy as np
import torch

import goboard
from goboard import Board, Stone

from train.arena import load_evaluator
from train.mcts import MCTS, apply_move
from train.sgf import margin_to_result, save_sgf

MARGIN_Y = 2   # rows above the board (column labels)
MARGIN_X = 4   # columns left of the board (row labels)
CELL_W = 2     # horizontal spacing between intersections

STONE_CHARS = {Stone.EMPTY: "·", Stone.BLACK: "●",
               Stone.WHITE: "○"}


def screen_to_board(my: int, mx: int, size: int) -> tuple[int, int] | None:
    """Map a mouse click to the nearest intersection, if any."""
    y = my - MARGIN_Y
    x = round((mx - MARGIN_X) / CELL_W)
    if 0 <= x < size and 0 <= y < size:
        return x, y
    return None


def stone_name(color: Stone) -> str:
    return STONE_CHARS[color] + (" black" if color == Stone.BLACK
                                 else " white")


class Game:
    def __init__(self, evaluate, board_size: int, komi: float, human: Stone,
                 simulations: int):
        self.board = Board(size=board_size, komi=komi)
        self.human = human
        self.simulations = simulations
        self.mcts = MCTS(evaluate, rng=np.random.default_rng())
        self.to_play = Stone.BLACK
        self.cursor = (board_size // 2, board_size // 2)
        self.last_move: tuple[int, int] | None = None
        self.value: float | None = None
        self.message = ""
        self.moves: list = []

    def apply(self, move: int) -> None:
        size = self.board.size
        self.moves.append((self.to_play, None if move == size * size
                           else (move % size, move // size)))
        apply_move(self.board, move, self.to_play)
        self.last_move = (None if move == size * size
                          else (move % size, move // size))
        self.to_play = goboard.opponent(self.to_play)
        self.message = ""

    def engine_move(self) -> None:
        root = self.mcts.run(self.board, self.to_play, self.simulations)
        self.value = -root.value()  # from the human's perspective
        self.apply(mcts_move := self.mcts.select_move(root, self.board.size))
        if mcts_move == self.board.size ** 2:
            self.message = "engine passes"


def draw(stdscr, game: Game, status: str) -> None:
    stdscr.erase()
    board = game.board
    size = board.size

    for x in range(size):
        stdscr.addstr(MARGIN_Y - 1, MARGIN_X + x * CELL_W, str((x + 1) % 10))
    for y in range(size):
        stdscr.addstr(MARGIN_Y + y, 1, f"{y + 1:2d}")
        for x in range(size):
            attr = curses.A_BOLD if board.at(x, y) == Stone.BLACK else 0
            if game.last_move == (x, y):
                attr |= curses.A_REVERSE
            stdscr.addstr(MARGIN_Y + y, MARGIN_X + x * CELL_W,
                          STONE_CHARS[board.at(x, y)], attr)

    info_x = MARGIN_X + size * CELL_W + 3
    lines = [
        "go-board",
        "",
        f"you:  {stone_name(game.human)}",
        f"turn: {stone_name(game.to_play)}"
        + (" (you)" if game.to_play == game.human else " (engine)"),
        ("" if game.value is None
         else f"engine value for you: {game.value:+.2f}"),
        "",
        "arrows/hjkl: move cursor",
        "enter/space/click: play",
        "p: pass   q: quit",
        "",
        status or game.message,
    ]
    for i, line in enumerate(lines):
        try:
            stdscr.addstr(MARGIN_Y + i, info_x, line)
        except curses.error:
            pass  # terminal too small; draw what fits

    stdscr.move(MARGIN_Y + game.cursor[1], MARGIN_X + game.cursor[0] * CELL_W)
    stdscr.refresh()


def read_human_move(stdscr, game: Game) -> int | None:
    """Returns a flattened move, size*size for pass, or None to quit."""
    size = game.board.size
    while True:
        draw(stdscr, game, "your move")
        key = stdscr.getch()
        cx, cy = game.cursor
        if key == ord("q"):
            return None
        if key == ord("p"):
            return size * size
        if key in (curses.KEY_LEFT, ord("h")):
            game.cursor = (max(cx - 1, 0), cy)
        elif key in (curses.KEY_RIGHT, ord("l")):
            game.cursor = (min(cx + 1, size - 1), cy)
        elif key in (curses.KEY_UP, ord("k")):
            game.cursor = (cx, max(cy - 1, 0))
        elif key in (curses.KEY_DOWN, ord("j")):
            game.cursor = (cx, min(cy + 1, size - 1))
        elif key == curses.KEY_MOUSE:
            try:
                _, mx, my, _, _ = curses.getmouse()
            except curses.error:
                continue
            point = screen_to_board(my, mx, size)
            if point is None:
                continue
            game.cursor = point
            if game.board.is_legal(*point, game.human):
                return point[1] * size + point[0]
            game.message = "illegal move"
        elif key in (curses.KEY_ENTER, ord("\n"), ord(" ")):
            if game.board.is_legal(cx, cy, game.human):
                return cy * size + cx
            game.message = "illegal move"


def run(stdscr, game: Game) -> None:
    curses.curs_set(1)
    curses.mousemask(curses.ALL_MOUSE_EVENTS)
    stdscr.keypad(True)

    while not game.board.is_terminal():
        if game.to_play == game.human:
            move = read_human_move(stdscr, game)
            if move is None:
                return
            game.apply(move)
        else:
            draw(stdscr, game, "engine thinking...")
            game.engine_move()

    margin = game.board.score()
    result = f"{'B+' if margin > 0 else 'W+'}{abs(margin)}"
    draw(stdscr, game, f"game over: {result} - press any key")
    stdscr.getch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--human", choices=["black", "white"],
                        default="black")
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sgf", type=Path, default=None,
                        help="save the game record here")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    evaluate, config = load_evaluator(args.checkpoint, device)
    human = Stone.BLACK if args.human == "black" else Stone.WHITE
    game = Game(evaluate, config["board_size"], args.komi, human,
                args.simulations)
    curses.wrapper(run, game)
    if args.sgf is not None and game.moves:
        result = (margin_to_result(game.board.score())
                  if game.board.is_terminal() else "Void")
        names = {human: "human",
                 goboard.opponent(human): args.checkpoint.stem}
        save_sgf(args.sgf, game.moves, game.board.size, args.komi, result,
                 black=names[Stone.BLACK], white=names[Stone.WHITE])
        print(f"saved {args.sgf}")


if __name__ == "__main__":
    main()
