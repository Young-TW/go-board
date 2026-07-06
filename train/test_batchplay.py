import numpy as np
import torch

from goboard import Stone

from train.batchplay import play_games
from train.net import PolicyValueNet
from train.selfplay import NetEvaluator


def uniform_batch(positions):
    n = positions[0][0].size ** 2 + 1
    priors = np.full((len(positions), n), 1.0 / n, dtype=np.float32)
    return priors, np.zeros(len(positions), dtype=np.float32)


def test_play_games_uniform_evaluator():
    game_samples, margins = play_games(
        uniform_batch, n_games=3, board_size=5, simulations=12,
        temperature_moves=4, rng=np.random.default_rng(0))

    assert len(game_samples) == 3
    assert len(margins) == 3
    for samples, margin in zip(game_samples, margins):
        assert samples
        black_won = margin > 0
        for sample in samples:
            assert sample.planes.shape == (3, 5, 5)
            assert sample.pi.shape == (26,)
            assert abs(sample.pi.sum() - 1.0) < 1e-5
            expected = (1.0 if black_won == (sample.to_play == Stone.BLACK)
                        else -1.0)
            assert sample.z == expected


def test_play_games_with_net_evaluator():
    torch.manual_seed(0)
    net = PolicyValueNet(board_size=5, channels=8, blocks=1)
    evaluator = NetEvaluator(net, device=torch.device("cpu"))
    game_samples, margins = play_games(
        evaluator.evaluate_batch, n_games=2, board_size=5, simulations=8,
        temperature_moves=2, rng=np.random.default_rng(1))
    assert len(game_samples) == 2
    assert all(samples for samples in game_samples)
