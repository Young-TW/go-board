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
    game_samples, margins, _ = play_games(
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


def test_spectate_file_is_written(tmp_path):
    spectate = tmp_path / "spectate.txt"
    play_games(uniform_planes, n_games=2, board_size=5, simulations=8,
               temperature_moves=2, rng=np.random.default_rng(0),
               spectate_path=spectate, spectate_every=1)
    text = spectate.read_text()
    header, *board = text.splitlines()
    assert header.startswith("moves=")
    assert "finished=" in header
    assert len(board) == 5


def test_ownership_and_score_targets():
    game_samples, margins, _ = play_games(
        uniform_planes, n_games=2, board_size=5, simulations=10,
        temperature_moves=2, rng=np.random.default_rng(5))
    komi = 7.5
    for samples, margin in zip(game_samples, margins):
        for sample in samples:
            sign = 1.0 if sample.to_play == Stone.BLACK else -1.0
            assert sample.ownership.shape == (25,)
            assert set(np.unique(sample.ownership)) <= {-1.0, 0.0, 1.0}
            # Ownership sums to the (komi-free) margin, from the
            # sample's perspective.
            assert sample.ownership.sum() == sign * (margin + komi)
            assert sample.score == sign * margin


def test_playout_cap_randomization_flags():
    game_samples, _, _ = play_games(
        uniform_planes, n_games=2, board_size=5, simulations=12,
        cheap_simulations=6, full_search_prob=0.5, temperature_moves=2,
        rng=np.random.default_rng(3))
    flags = [s.train_pi for game in game_samples for s in game]
    assert set(flags) <= {0.0, 1.0}
    assert 0.0 in flags and 1.0 in flags

    # prob=1 keeps the old behavior: every move trains the policy.
    game_samples, _, _ = play_games(
        uniform_planes, n_games=1, board_size=5, simulations=8,
        temperature_moves=2, rng=np.random.default_rng(4))
    assert all(s.train_pi == 1.0 for game in game_samples for s in game)


def test_eval_cache_hits_and_correctness():
    game_samples, _, stats = play_games(
        uniform_planes, n_games=4, board_size=5, simulations=12,
        parallel=4, rng=np.random.default_rng(0))
    assert stats["eval_cache_hits"] > 0
    assert stats["eval_cache_lookups"] > stats["eval_cache_hits"]
    # Games still complete with valid targets.
    for samples in game_samples:
        for sample in samples:
            assert abs(sample.pi.sum() - 1.0) < 1e-4


def black_always_losing(planes):
    """Value says black is lost; priors uniform."""
    priors, _ = uniform_planes(planes)
    # Plane 2 is the black-to-play indicator.
    black = planes[:, 2].max(axis=(1, 2)) > 0.5
    values = np.where(black, -0.99, 0.99).astype(np.float32)
    return priors, values


def test_resignation_ends_games_early():
    game_samples, margins, stats = play_games(
        black_always_losing, n_games=4, board_size=5, simulations=12,
        temperature_moves=0, resign_threshold=0.9, no_resign_fraction=0.0,
        rng=np.random.default_rng(6))
    full_length = [s for g in game_samples for s in g]
    assert all(margin == -10000.0 for margin in margins)  # white wins
    assert len(full_length) < 4 * 20  # far shorter than played-out games
    for sample in full_length:
        assert sample.w_own == 0.0  # no scored final position
        expected = 1.0 if sample.to_play == Stone.WHITE else -1.0
        assert sample.z == expected
    assert stats["resign_calibration_games"] == 0


def test_resign_calibration_counts():
    _, margins, stats = play_games(
        black_always_losing, n_games=3, board_size=5, simulations=12,
        temperature_moves=0, resign_threshold=0.9, no_resign_fraction=1.0,
        rng=np.random.default_rng(7))
    # All games are calibration games: they play to the end...
    assert all(abs(margin) < 10000.0 for margin in margins)
    assert stats["resign_calibration_games"] == 3


def test_play_games_with_net_evaluator():
    torch.manual_seed(0)
    net = PolicyValueNet(board_size=5, channels=8, blocks=1,
                         in_planes=goboard.FEATURE_PLANES)
    evaluator = NetEvaluator(net, device=torch.device("cpu"))
    game_samples, margins, _ = play_games(
        evaluator.evaluate_planes, n_games=2, board_size=5, simulations=8,
        temperature_moves=2, rng=np.random.default_rng(1))
    assert len(game_samples) == 2
    assert all(samples for samples in game_samples)
