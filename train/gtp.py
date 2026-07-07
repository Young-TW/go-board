"""GTP (Go Text Protocol) engine wrapping a trained checkpoint.

    uv run python -m train.gtp checkpoints/iter_0100.pt --simulations 512

Speaks enough GTP for Sabaki/GoGui and engine matches (gogui-twogtp,
GNU Go). Board size is fixed by the checkpoint; other sizes are
rejected with "unacceptable size".
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import goboard
from goboard import Board, Stone

from train.arena import load_evaluator
from train.mcts import MCTS, apply_move

# GTP columns skip the letter I.
COLUMNS = "ABCDEFGHJKLMNOPQRST"


def vertex_to_xy(vertex: str, size: int) -> tuple[int, int] | None:
    """A1-style vertex to board (x, y); None for a pass."""
    vertex = vertex.strip().upper()
    if vertex == "PASS":
        return None
    x = COLUMNS.index(vertex[0])
    row = int(vertex[1:])
    y = size - row  # GTP rows count from the bottom
    if not (0 <= x < size and 0 <= y < size):
        raise ValueError(f"vertex off board: {vertex}")
    return x, y


def xy_to_vertex(x: int, y: int, size: int) -> str:
    return f"{COLUMNS[x]}{size - y}"


class GtpEngine:
    def __init__(self, checkpoint: Path, simulations: int,
                 resign_threshold: float, device: str | None):
        dev = torch.device(device) if device else None
        self.evaluate, config = load_evaluator(checkpoint, dev)
        self.size = config["board_size"]
        self.simulations = simulations
        self.resign_threshold = resign_threshold
        self.mcts = MCTS(self.evaluate, rng=np.random.default_rng())
        self.name = f"go-board {checkpoint.stem}"
        self.komi = 7.0
        self.moves: list[tuple[Stone, tuple[int, int] | None]] = []
        self.rebuild()

    def rebuild(self) -> None:
        self._built_komi = self.komi
        self.board = Board(size=self.size, komi=self.komi)
        for color, point in self.moves:
            if point is None:
                self.board.pass_()
            else:
                self.board.play(point[0], point[1], color)

    # --- command handlers ----------------------------------------

    def cmd_boardsize(self, args: list[str]) -> str:
        if int(args[0]) != self.size:
            raise ValueError("unacceptable size")
        return ""

    def cmd_clear_board(self, args: list[str]) -> str:
        self.moves = []
        self.rebuild()
        return ""

    def cmd_komi(self, args: list[str]) -> str:
        self.komi = float(args[0])
        if not self.moves:
            self.rebuild()
        return ""

    def cmd_play(self, args: list[str]) -> str:
        color = Stone.BLACK if args[0].upper().startswith("B") \
            else Stone.WHITE
        point = vertex_to_xy(args[1], self.size)
        if point is None:
            self.board.pass_()
        elif not self.board.play(point[0], point[1], color):
            raise ValueError("illegal move")
        self.moves.append((color, point))
        return ""

    def cmd_genmove(self, args: list[str]) -> str:
        color = Stone.BLACK if args[0].upper().startswith("B") \
            else Stone.WHITE
        root = self.mcts.run(self.board, color, self.simulations)
        if root.value() < -self.resign_threshold:
            return "resign"
        move = self.mcts.select_move(root, self.size, temperature=0.0)
        if move == self.size * self.size:
            self.board.pass_()
            self.moves.append((color, None))
            return "pass"
        x, y = move % self.size, move // self.size
        apply_move(self.board, move, color)
        self.moves.append((color, (x, y)))
        return xy_to_vertex(x, y, self.size)

    def cmd_undo(self, args: list[str]) -> str:
        if not self.moves:
            raise ValueError("cannot undo")
        self.moves.pop()
        self.rebuild()
        return ""

    def cmd_final_score(self, args: list[str]) -> str:
        # board.score() uses the komi it was built with; reapply the
        # current komi in case it changed mid-game.
        raw = self.board.score() + self._built_komi
        margin = raw - self.komi
        if margin == 0:
            return "0"
        return f"B+{margin}" if margin > 0 else f"W+{-margin}"

    def handle(self, command: str, args: list[str]) -> str:
        simple = {
            "protocol_version": lambda a: "2",
            "name": lambda a: self.name,
            "version": lambda a: "0.1",
            "known_command": lambda a: "true" if a and a[0] in COMMANDS
                                       else "false",
            "list_commands": lambda a: "\n".join(sorted(COMMANDS)),
            "showboard": lambda a: "\n" + str(self.board),
            "time_settings": lambda a: "",
            "time_left": lambda a: "",
        }
        if command in simple:
            return simple[command](args)
        return getattr(self, f"cmd_{command}")(args)


COMMANDS = {
    "protocol_version", "name", "version", "known_command",
    "list_commands", "boardsize", "clear_board", "komi", "play",
    "genmove", "undo", "final_score", "showboard", "quit",
    "time_settings", "time_left",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--resign-threshold", type=float, default=2.0,
                        help="genmove answers 'resign' below -threshold; "
                             ">=1 disables")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    engine = GtpEngine(args.checkpoint, args.simulations,
                       args.resign_threshold, args.device)
    for raw in sys.stdin:
        line = raw.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        ident = ""
        if parts[0].isdigit():
            ident = parts[0]
            parts = parts[1:]
        command, cmd_args = parts[0], parts[1:]
        if command == "quit":
            print(f"={ident}\n", flush=True)
            return
        try:
            if command not in COMMANDS:
                raise ValueError("unknown command")
            result = engine.handle(command, cmd_args)
            print(f"={ident} {result}".rstrip() + "\n", flush=True)
        except Exception as error:  # GTP requires an error response
            print(f"?{ident} {error}\n", flush=True)


if __name__ == "__main__":
    main()
