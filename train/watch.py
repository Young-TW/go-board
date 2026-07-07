"""Watch training self-play live in the terminal.

    uv run python -m train.watch --dir checkpoints

Renders the spectate file the training loop keeps updating (one of
the games currently being played) plus the tail of train.log.
Press q to quit. Read-only: it never touches the training process.
"""

import argparse
import curses
import time
from pathlib import Path

STONES = {"X": "●", "O": "○", ".": "·"}


def read_spectate(path: Path) -> tuple[str, list[str]]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return "waiting for spectate file...", []
    if not lines:
        return "spectate file empty", []
    header, board = lines[0], lines[1:]
    return header, board


def read_log_tail(path: Path, limit: int = 10) -> list[str]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return ["waiting for train.log..."]
    return lines[-limit:]


def run(stdscr, directory: Path, interval: float) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    spectate = directory / "spectate.txt"
    log = directory / "train.log"

    while True:
        header, board = read_spectate(spectate)
        stdscr.erase()

        def put(y: int, x: int, text: str, attr: int = 0) -> None:
            try:
                stdscr.addstr(y, x, text, attr)
            except curses.error:
                pass  # terminal too small; draw what fits

        put(0, 0, "go-board self-play spectator (q to quit)",
            curses.A_BOLD)
        put(1, 0, header)
        for y, row in enumerate(board):
            rendered = " ".join(STONES.get(c, c)
                                for c in row.replace(" ", ""))
            put(3 + y, 2, rendered)

        offset = 4 + len(board)
        put(offset, 0, "train.log", curses.A_BOLD)
        for i, line in enumerate(read_log_tail(log)):
            put(offset + 1 + i, 0, line[:curses.COLS - 1])
        stdscr.refresh()

        deadline = time.monotonic() + interval
        while time.monotonic() < deadline:
            if stdscr.getch() == ord("q"):
                return
            time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    curses.wrapper(run, args.dir, args.interval)


if __name__ == "__main__":
    main()
