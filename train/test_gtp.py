import pytest

from train.gtp import COLUMNS, vertex_to_xy, xy_to_vertex


def test_vertex_roundtrip():
    for size in (9, 19):
        for x in range(size):
            for y in range(size):
                vertex = xy_to_vertex(x, y, size)
                assert vertex_to_xy(vertex, size) == (x, y)


def test_vertex_conventions():
    # GTP: A1 is bottom-left, columns skip I.
    assert vertex_to_xy("A1", 9) == (0, 8)
    assert vertex_to_xy("J9", 9) == (8, 0)
    assert vertex_to_xy("E5", 9) == (4, 4)  # tengen
    assert "I" not in COLUMNS
    assert vertex_to_xy("pass", 9) is None


def test_vertex_off_board_rejected():
    with pytest.raises(ValueError):
        vertex_to_xy("A10", 9)
