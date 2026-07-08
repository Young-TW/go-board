"""Minimal SGF (FF[4]) writer for game records."""

from pathlib import Path

from goboard import Stone

Move = tuple[Stone, tuple[int, int] | None]  # None means pass


def _coord(point: tuple[int, int] | None) -> str:
    if point is None:
        return ""  # FF[4] pass
    x, y = point
    return f"{chr(ord('a') + x)}{chr(ord('a') + y)}"


def game_to_sgf(moves: list[Move], board_size: int, komi: float,
                result: str, black: str = "black",
                white: str = "white") -> str:
    header = (f";GM[1]FF[4]CA[UTF-8]AP[go-board]SZ[{board_size}]"
              f"KM[{komi}]RU[Tromp-Taylor]RE[{result}]"
              f"PB[{black}]PW[{white}]")
    body = "".join(
        f";{'B' if color == Stone.BLACK else 'W'}[{_coord(point)}]"
        for color, point in moves)
    return f"({header}{body})\n"


def margin_to_result(black_margin: float) -> str:
    if black_margin == 0:
        return "0"  # jigo
    side = "B" if black_margin > 0 else "W"
    return f"{side}+{abs(black_margin)}"


def save_sgf(path: Path, moves: list[Move], board_size: int, komi: float,
             result: str, black: str = "black",
             white: str = "white") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(game_to_sgf(moves, board_size, komi, result,
                                black, white))
