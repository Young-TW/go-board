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

    def __init__(self, net, device=None, compile: bool = False):
        self.device = device if device is not None else default_device()
        self.net = net.to(self.device).eval()
        self.forward = self.net
        self.autocast = self.device.type == "cuda"
        if compile and self.device.type == "cuda":
            try:
                # dynamic=True: the batch size changes every round as
                # games finish, and per-size recompiles would dominate.
                self.forward = torch.compile(self.net, dynamic=True)
            except Exception:
                pass  # compile is best-effort; eager works everywhere

    def __call__(self, board: Board, to_play: Stone):
        probs, values = self.evaluate_batch([(board, to_play)])
        return probs[0], float(values[0])

    def evaluate_batch(self, positions):
        """Evaluate [(board, to_play), ...] in one forward pass."""
        planes = np.stack(
            [board.features(to_play) for board, to_play in positions])
        return self.evaluate_planes(planes)

    def _ensure_buffers(self, count: int, plane_shape, stride: int) -> None:
        """(Re)allocate persistent pinned staging buffers."""
        if getattr(self, "_in_buf", None) is not None \
                and self._in_buf.shape[0] >= count \
                and self._in_buf.shape[1:] == plane_shape:
            return
        capacity = max(count, 2 * getattr(self, "_capacity", 0))
        self._capacity = capacity
        self._in_buf = torch.empty((capacity, *plane_shape),
                                   dtype=torch.float32, pin_memory=True)
        self._out_priors = torch.empty((capacity, stride),
                                       dtype=torch.float32, pin_memory=True)
        self._out_values = torch.empty(capacity, dtype=torch.float32,
                                       pin_memory=True)

    def evaluate_planes(self, planes):
        """Evaluate a (count, planes, size, size) float32 batch.

        On CUDA the returned arrays are views of persistent pinned
        buffers, valid until the next call — callers must consume them
        immediately (SelfPlayPool.submit copies synchronously).
        """
        # Nets trained on the older, smaller encoding use a prefix of
        # the feature planes.
        planes = planes[:, :self.net.in_planes]
        autocast = torch.autocast(device_type=self.device.type,
                                  dtype=torch.bfloat16,
                                  enabled=self.autocast)
        if self.device.type != "cuda":
            x = torch.from_numpy(np.ascontiguousarray(planes))
            with torch.no_grad(), autocast:
                logits, values, _, _ = self.forward(x)
            probs = torch.softmax(logits.float(), dim=1).numpy()
            return probs, values.float().numpy()

        count = planes.shape[0]
        stride = planes.shape[-1] ** 2 + 1
        self._ensure_buffers(count, planes.shape[1:], stride)
        stage = self._in_buf[:count]
        stage.copy_(torch.from_numpy(np.ascontiguousarray(planes)))
        x = stage.to(self.device, non_blocking=True)
        with torch.no_grad(), autocast:
            logits, values, _, _ = self.forward(x)
        probs = torch.softmax(logits.float(), dim=1)
        self._out_priors[:count].copy_(probs, non_blocking=True)
        self._out_values[:count].copy_(values.float(), non_blocking=True)
        torch.cuda.synchronize()
        return (self._out_priors[:count].numpy(),
                self._out_values[:count].numpy())


@dataclass
class Sample:
    planes: np.ndarray  # (FEATURE_PLANES, size, size) float32
    pi: np.ndarray      # (size*size + 1,) MCTS visit distribution
    to_play: Stone
    z: float = field(default=0.0)  # final outcome for to_play, in [-1, 1]
    # Weight of this sample's policy target (0 for cheap-search moves
    # under playout cap randomization).
    train_pi: float = field(default=1.0)
    # Auxiliary targets from to_play's perspective: final per-point
    # ownership (flat, in {-1, 0, 1}) and final score margin. w_own is
    # 0 for resigned games (no scored final position exists).
    ownership: np.ndarray | None = field(default=None)
    score: float = field(default=0.0)
    w_own: float = field(default=1.0)


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
        if black_margin == 0.0:
            sample.z = 0.0  # jigo
        else:
            black_won = black_margin > 0
            sample.z = (1.0 if black_won == (sample.to_play == Stone.BLACK)
                        else -1.0)
    return samples, black_margin


def apply_random_symmetries(planes: np.ndarray, pi: np.ndarray,
                            ownership: np.ndarray,
                            rng: np.random.Generator):
    """Apply an independent random dihedral transform to every sample
    of a training batch (KataGo-style train-time augmentation, so the
    replay buffer holds each position once instead of eight times).

    planes: (N, C, n, n); pi: (N, n*n + 1); ownership: (N, n*n).
    Returns transformed copies.
    """
    count, _, size, _ = planes.shape
    out_planes = np.empty_like(planes)
    out_pi = pi.copy()
    out_own = np.empty_like(ownership)
    board_pi = pi[:, :-1].reshape(count, size, size)
    own = ownership.reshape(count, size, size)
    transform = rng.integers(0, 8, size=count)
    for code in range(8):
        mask = transform == code
        if not mask.any():
            continue
        k, flip = code % 4, code >= 4
        p = np.rot90(planes[mask], k, axes=(2, 3))
        b = np.rot90(board_pi[mask], k, axes=(1, 2))
        o = np.rot90(own[mask], k, axes=(1, 2))
        if flip:
            p = np.flip(p, axis=3)
            b = np.flip(b, axis=2)
            o = np.flip(o, axis=2)
        out_planes[mask] = p
        out_pi[mask, :-1] = b.reshape(mask.sum(), -1)
        out_own[mask] = o.reshape(mask.sum(), -1)
    return out_planes, out_pi, out_own


def symmetries(planes: np.ndarray, pi: np.ndarray, board_size: int,
               ownership: np.ndarray | None = None):
    """Yield the 8 dihedral transforms of (planes, pi[, ownership])."""
    board_pi = pi[:-1].reshape(board_size, board_size)
    pass_prob = pi[-1]
    owner = (ownership.reshape(board_size, board_size)
             if ownership is not None else None)
    for k in range(4):
        for flip in (False, True):
            p = np.rot90(planes, k, axes=(1, 2))
            b = np.rot90(board_pi, k)
            o = np.rot90(owner, k) if owner is not None else None
            if flip:
                p = np.flip(p, axis=2)
                b = np.flip(b, axis=1)
                o = np.flip(o, axis=1) if o is not None else None
            result = (np.ascontiguousarray(p),
                      np.append(b.ravel(), pass_prob).astype(np.float32))
            if owner is not None:
                result += (o.ravel().astype(np.float32),)
            yield result
