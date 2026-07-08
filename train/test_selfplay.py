import numpy as np
import torch

import goboard
from goboard import Stone

from train.net import PolicyValueNet
from train.selfplay import NetEvaluator, play_game, symmetries


def test_play_game_produces_consistent_samples():
    torch.manual_seed(0)
    net = PolicyValueNet(board_size=5, channels=8, blocks=1)
    evaluator = NetEvaluator(net, device=torch.device("cpu"))
    samples, black_margin = play_game(
        evaluator, board_size=5, simulations=8, temperature_moves=4,
        rng=np.random.default_rng(0))

    assert samples
    black_won = black_margin > 0
    for sample in samples:
        assert sample.planes.shape == (goboard.FEATURE_PLANES, 5, 5)
        assert sample.planes.dtype == np.float32
        assert sample.pi.shape == (26,)
        assert abs(sample.pi.sum() - 1.0) < 1e-5
        expected = 1.0 if black_won == (sample.to_play == Stone.BLACK) else -1.0
        assert sample.z == expected


def test_apply_random_symmetries_batch():
    from train.selfplay import apply_random_symmetries

    rng = np.random.default_rng(2)
    planes = rng.random((16, 3, 5, 5)).astype(np.float32)
    pi = rng.random((16, 26)).astype(np.float32)
    ownership = rng.random((16, 25)).astype(np.float32)

    out_planes, out_pi, out_own = apply_random_symmetries(
        planes, pi, ownership, rng)
    assert out_planes.shape == planes.shape
    assert out_pi.shape == pi.shape
    assert out_own.shape == ownership.shape
    for i in range(16):
        # Each sample keeps its multiset of values and its pass prob.
        assert np.allclose(sorted(out_pi[i, :-1]), sorted(pi[i, :-1]),
                           atol=1e-6)
        assert out_pi[i, -1] == pi[i, -1]
        assert np.allclose(sorted(out_own[i]), sorted(ownership[i]),
                           atol=1e-6)
    # With 16 samples over 8 transforms, at least one differs.
    assert not np.array_equal(out_planes, planes)


def test_symmetries_preserve_content():
    rng = np.random.default_rng(1)
    planes = rng.random((3, 5, 5)).astype(np.float32)
    pi = rng.random(26).astype(np.float32)
    pi /= pi.sum()

    variants = list(symmetries(planes, pi, board_size=5))
    assert len(variants) == 8
    # With an ownership map, it is transformed alongside the planes.
    ownership = rng.random(25).astype(np.float32)
    with_owner = list(symmetries(planes, pi, 5, ownership))
    assert len(with_owner) == 8
    for p, _, o in with_owner:
        assert o.shape == (25,)
        assert np.allclose(sorted(o), sorted(ownership), atol=1e-6)
    assert np.array_equal(with_owner[0][2], ownership)
    for p, transformed_pi in variants:
        assert p.shape == planes.shape
        assert p.flags["C_CONTIGUOUS"]
        assert transformed_pi.shape == pi.shape
        assert abs(transformed_pi.sum() - pi.sum()) < 1e-5
        assert transformed_pi[-1] == pi[-1]  # pass never moves
        # The multiset of board probabilities is preserved.
        assert np.allclose(sorted(transformed_pi[:-1]), sorted(pi[:-1]),
                           atol=1e-6)
    # Identity transform comes first.
    assert np.array_equal(variants[0][0], planes)
