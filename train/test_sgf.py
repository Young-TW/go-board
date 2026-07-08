from goboard import Stone

from train.sgf import game_to_sgf, margin_to_result


def test_game_to_sgf():
    moves = [(Stone.BLACK, (4, 4)), (Stone.WHITE, (2, 2)),
             (Stone.BLACK, None)]  # pass
    sgf = game_to_sgf(moves, board_size=9, komi=7.0, result="B+3",
                      black="me", white="engine")
    assert sgf.startswith("(;GM[1]FF[4]")
    assert "SZ[9]" in sgf and "KM[7.0]" in sgf and "RE[B+3]" in sgf
    assert "PB[me]PW[engine]" in sgf
    assert ";B[ee];W[cc];B[]" in sgf
    assert sgf.rstrip().endswith(")")


def test_margin_to_result():
    assert margin_to_result(3.0) == "B+3.0"
    assert margin_to_result(-7.0) == "W+7.0"
    assert margin_to_result(0.0) == "0"
