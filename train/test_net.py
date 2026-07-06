import torch

from train.net import IN_PLANES, PolicyValueNet


def test_forward_shapes_and_ranges():
    net = PolicyValueNet(board_size=5, channels=8, blocks=1)
    x = torch.randn(4, IN_PLANES, 5, 5)
    logits, value = net(x)
    assert logits.shape == (4, 5 * 5 + 1)
    assert value.shape == (4,)
    assert value.abs().max() <= 1.0


def test_matches_goboard_features():
    from goboard import Board, Stone

    board = Board(size=5)
    board.play(2, 2, Stone.BLACK)
    net = PolicyValueNet(board_size=5, channels=8, blocks=1).eval()
    x = torch.from_numpy(board.features(Stone.WHITE)[None])
    logits, value = net(x)
    assert logits.shape == (1, 26)
