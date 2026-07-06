import numpy as np
import torch

from goboard import Stone

from train.arena import play_match
from train.net import PolicyValueNet
from train.selfplay import NetEvaluator


def test_play_match_returns_score():
    torch.manual_seed(0)
    device = torch.device("cpu")
    eval_a = NetEvaluator(
        PolicyValueNet(board_size=5, channels=8, blocks=1), device)
    eval_b = NetEvaluator(
        PolicyValueNet(board_size=5, channels=8, blocks=1), device)
    margin = play_match(
        {Stone.BLACK: eval_a, Stone.WHITE: eval_b}, board_size=5,
        komi=7.5, simulations=8, temperature_moves=2,
        rng=np.random.default_rng(0))
    assert isinstance(margin, float)
    assert margin != 0.0  # komi 7.5 makes draws impossible
