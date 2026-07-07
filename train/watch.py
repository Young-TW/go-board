"""Watch training self-play live in the terminal.

    uv run python -m train.watch --dir checkpoints

Renders the spectate file the training loop keeps updating (one of
the games currently being played) plus the tail of train.log.
Press q to quit. Read-only: it never touches the training process.
"""

import argparse
import curses
import re
import time
from pathlib import Path

import numpy as np

STONES = {"X": "●", "O": "○", ".": "·"}

LOSS_RE = re.compile(r"p-loss (\d+\.\d+) \| v-loss (\d+\.\d+)")

# Braille dot bits by (dx, dy) within a 2x4-dot cell.
BRAILLE_BITS = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
                (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80}


def parse_losses(path: Path) -> tuple[list[float], list[float]]:
    try:
        text = path.read_text()
    except OSError:
        return [], []
    pairs = LOSS_RE.findall(text)
    return ([float(p) for p, _ in pairs], [float(v) for _, v in pairs])


def downsample(values: list[float], points: int) -> list[float]:
    if len(values) <= points:
        return values
    return [float(chunk.mean())
            for chunk in np.array_split(np.asarray(values), points)]


def braille_chart(values: list[float], cells_w: int,
                  cells_h: int) -> tuple[list[str], float, float]:
    """Render a line chart as braille rows; returns (rows, lo, hi)."""
    dots_w, dots_h = cells_w * 2, cells_h * 4
    series = downsample(values, dots_w)
    lo, hi = min(series), max(series)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    grid = [[0] * cells_w for _ in range(cells_h)]

    def set_dot(x: int, y: int) -> None:
        grid[y // 4][x // 2] |= BRAILLE_BITS[(x % 2, y % 4)]

    prev_y = None
    for x, value in enumerate(series):
        y = round((hi - value) / (hi - lo) * (dots_h - 1))
        if prev_y is None:
            set_dot(x, y)
        else:  # connect to the previous point so the line is continuous
            step = 1 if y >= prev_y else -1
            for yy in range(prev_y, y + step, step):
                set_dot(x, yy)
        prev_y = y
    rows = ["".join(chr(0x2800 | cell) for cell in row) for row in grid]
    return rows, lo, hi


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


def draw_loss_chart(put, values: list[float], name: str, top: int,
                    left: int, cells_w: int, cells_h: int,
                    color: int) -> None:
    put(top, left, f"{name} {values[-1]:.4f}", curses.A_BOLD)
    rows, lo, hi = braille_chart(values, cells_w, cells_h)
    put(top + 1, left, f"{hi:.3f}", curses.A_DIM)
    for i, row in enumerate(rows):
        put(top + 1 + i, left + 6, row, curses.color_pair(color))
    put(top + cells_h, left, f"{lo:.3f}", curses.A_DIM)


def run(stdscr, directory: Path, interval: float) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    spectate = directory / "spectate.txt"
    log = directory / "train.log"

    while True:
        header, board = read_spectate(spectate)
        p_losses, v_losses = parse_losses(log)
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

        # Loss curves to the right of the board: two separately scaled
        # charts (the series differ by an order of magnitude — never
        # share one axis). They start below the header line so long
        # headers never collide with the chart titles.
        chart_x = max(30, 2 * len(board) + 8)
        chart_w = (curses.COLS - chart_x - 8) // 1
        chart_h = 4
        p_top = 3
        v_top = p_top + chart_h + 2
        if p_losses and chart_w >= 16:
            draw_loss_chart(put, p_losses, "p-loss", p_top, chart_x,
                            chart_w - 6, chart_h, color=1)
            draw_loss_chart(put, v_losses, "v-loss", v_top, chart_x,
                            chart_w - 6, chart_h, color=2)

        offset = 2 + max(3 + len(board), v_top + chart_h + 1)
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
