from train.tui import CELL_W, MARGIN_X, MARGIN_Y, screen_to_board


def test_screen_to_board_maps_intersections():
    for size in (5, 9):
        for y in range(size):
            for x in range(size):
                my = MARGIN_Y + y
                mx = MARGIN_X + x * CELL_W
                assert screen_to_board(my, mx, size) == (x, y)
                # A click one cell to the right still snaps to the point.
                assert screen_to_board(my, mx + 1, size) in ((x, y),
                                                             (x + 1, y))


def test_screen_to_board_rejects_outside():
    assert screen_to_board(0, MARGIN_X, 9) is None
    assert screen_to_board(MARGIN_Y + 9, MARGIN_X, 9) is None
    assert screen_to_board(MARGIN_Y, MARGIN_X + 9 * CELL_W + 2, 9) is None
