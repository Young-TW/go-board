import numpy as np
import torch

import goboard
from goboard import Stone

from train.batchplay import play_games
from train.net import PolicyValueNet
from train.selfplay import NetEvaluator


def uniform_planes(planes):
    count = planes.shape[0]
    stride = planes.shape[-1] ** 2 + 1
    priors = np.full((count, stride), 1.0 / stride, dtype=np.float32)
    return priors, np.zeros(count, dtype=np.float32)


def test_play_games_uniform_evaluator():
    game_samples, margins = play_games(
        uniform_planes, n_games=3, board_size=5, simulations=12,
        temperature_moves=4, parallel=2, rng=np.random.default_rng(0))

    assert len(game_samples) == 3
    assert len(margins) == 3
    for samples, margin in zip(game_samples, margins):
        assert samples
        black_won = margin > 0
        for sample in samples:
            assert sample.planes.shape == (goboard.FEATURE_PLANES, 5, 5)
            assert sample.pi.shape == (26,)
            assert abs(sample.pi.sum() - 1.0) < 1e-4
            expected = (1.0 if black_won == (sample.to_play == Stone.BLACK)
                        else -1.0)
            assert sample.z == expected
    # Self-play alternates colors, starting with black.
    assert game_samples[0][0].to_play == Stone.BLACK
    assert game_samples[0][1].to_play == Stone.WHITE


def test_play_games_with_net_evaluator():
    torch.manual_seed(0)
    net = PolicyValueNet(board_size=5, channels=8, blocks=1,
                         in_planes=goboard.FEATURE_PLANES)
    evaluator = NetEvaluator(net, device=torch.device("cpu"))
    game_samples, margins = play_games(
        evaluator.evaluate_planes, n_games=2, board_size=5, simulations=8,
        temperature_moves=2, rng=np.random.default_rng(1))
    assert len(game_samples) == 2
    assert all(samples for samples in game_samples)
