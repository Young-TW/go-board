"""Self-play game generation: MCTS guided by the policy/value net."""

from dataclasses import dataclass, field

import numpy as np
import torch

import goboard
from goboard import Board, Stone

from train.mcts import MCTS, apply_move
from train.net import default_device


class NetEvaluator:
    """Adapts PolicyValueNet to the MCTS evaluate interface."""

    def __init__(self, net, device=None):
        self.device = device if device is not None else default_device()
        self.net = net.to(self.device).eval()

    def __call__(self, board: Board, to_play: Stone):
        planes = board.features(to_play)[None]  # add batch dim
        x = torch.from_numpy(planes).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
        probs = torch.softmax(logits[0], dim=0).cpu().numpy()
        return probs, float(value.item())


@dataclass
class Sample:
    planes: np.ndarray  # (3, size, size) float32
    pi: np.ndarray      # (size*size + 1,) MCTS visit distribution
    to_play: Stone
    z: float = field(default=0.0)  # final outcome for to_play, in {-1, 1}


def play_game(evaluate, board_size: int = 9, komi: float = 7.5,
              simulations: int = 128, temperature_moves: int = 8,
              rng: np.random.Generator | None = None):
    """Play one self-play game; returns (samples, black_margin)."""
    rng = rng if rng is not None else np.random.default_rng()
    mcts = MCTS(evaluate, rng=rng)
    board = Board(size=board_size, komi=komi)
    to_play = Stone.BLACK
    samples: list[Sample] = []

    # Safety net against pathological never-passing games early in
    # training; the position is Tromp-Taylor scoreable regardless.
    max_moves = board_size * board_size * 2

    for move_count in range(max_moves):
        if board.is_terminal():
            break
        root = mcts.run(board, to_play, simulations, add_noise=True)
        pi = mcts.policy(root, board_size)
        samples.append(Sample(board.features(to_play), pi, to_play))

        temperature = 1.0 if move_count < temperature_moves else 0.0
        move = mcts.select_move(root, board_size, temperature=temperature)
        apply_move(board, move, to_play)
        to_play = goboard.opponent(to_play)

    black_margin = board.score()
    for sample in samples:
        black_won = black_margin > 0
        sample.z = 1.0 if black_won == (sample.to_play == Stone.BLACK) else -1.0
    return samples, black_margin


def symmetries(planes: np.ndarray, pi: np.ndarray, board_size: int):
    """Yield the 8 dihedral transforms of (planes, pi) for augmentation."""
    board_pi = pi[:-1].reshape(board_size, board_size)
    pass_prob = pi[-1]
    for k in range(4):
        for flip in (False, True):
            p = np.rot90(planes, k, axes=(1, 2))
            b = np.rot90(board_pi, k)
            if flip:
                p = np.flip(p, axis=2)
                b = np.flip(b, axis=1)
            yield (np.ascontiguousarray(p),
                   np.append(b.ravel(), pass_prob).astype(np.float32))
