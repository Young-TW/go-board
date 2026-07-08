import torch

import goboard

from train.net import PolicyValueNet


def test_forward_shapes_and_ranges():
    net = PolicyValueNet(board_size=5, channels=8, blocks=1,
                         in_planes=goboard.FEATURE_PLANES)
    x = torch.randn(4, goboard.FEATURE_PLANES, 5, 5)
    logits, value, ownership, score = net(x)
    assert logits.shape == (4, 5 * 5 + 1)
    assert value.shape == (4,)
    assert value.abs().max() <= 1.0
    assert ownership.shape == (4, 25)
    assert ownership.abs().max() <= 1.0
    assert score.shape == (4,)


def test_one_net_plays_every_board_size():
    # v2 heads are convolutional/pooled: the same weights must accept
    # 9x9 and 19x19 inputs.
    net = PolicyValueNet(board_size=9, channels=8, blocks=1,
                         in_planes=goboard.FEATURE_PLANES).eval()
    for size in (5, 9, 19):
        logits, value, ownership, score = net(
            torch.randn(2, goboard.FEATURE_PLANES, size, size))
        assert logits.shape == (2, size * size + 1)
        assert ownership.shape == (2, size * size)
        assert value.shape == score.shape == (2,)


def test_legacy_three_plane_input():
    # Checkpoints from the v1 encoding still construct and run.
    from train.net import LegacyPolicyValueNet

    net = LegacyPolicyValueNet(board_size=5, channels=8, blocks=1)
    assert net.in_planes == 3
    logits, *_ = net(torch.randn(2, 3, 5, 5))
    assert logits.shape == (2, 26)


def test_matches_goboard_features():
    from goboard import Board, Stone

    board = Board(size=5)
    board.play(2, 2, Stone.BLACK)
    net = PolicyValueNet(board_size=5, channels=8, blocks=1,
                         in_planes=goboard.FEATURE_PLANES).eval()
    x = torch.from_numpy(board.features(Stone.WHITE)[None])
    logits, *_ = net(x)
    assert logits.shape == (1, 26)
