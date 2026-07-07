"""PUCT Monte Carlo tree search (AlphaZero-style) over goboard positions.

The search is decoupled from the network: it takes an `evaluate`
callable mapping (board, to_play) to (priors, value), where priors is
an array of size*size + 1 move probabilities (last index = pass) and
value is the expected outcome in [-1, 1] for the side to play.

Values stored in a node are from the perspective of the player to move
at that node, so selection scores an edge with the negated child value.
"""

import math

import numpy as np

import goboard
from goboard import Board, Stone


class Node:
    __slots__ = ("prior", "visits", "value_sum", "children")

    def __init__(self, prior: float):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}

    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def terminal_value(board: Board, to_play: Stone) -> float:
    """Outcome of a finished game for the side to play."""
    black_margin = board.score()
    if black_margin == 0.0:
        return 0.0  # jigo (integer komi)
    return 1.0 if (black_margin > 0) == (to_play == Stone.BLACK) else -1.0


class MCTS:
    def __init__(self, evaluate, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, noise_fraction: float = 0.25,
                 rng: np.random.Generator | None = None):
        self.evaluate = evaluate
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.rng = rng if rng is not None else np.random.default_rng()

    def run(self, board: Board, to_play: Stone, simulations: int,
            add_noise: bool = False) -> Node:
        root = Node(0.0)
        priors, value = self.evaluate(board, to_play)
        self.expand(root, board, to_play, priors, value)
        if add_noise:
            self.add_dirichlet_noise(root)
        for _ in range(simulations):
            self._simulate(root, board.copy(), to_play)
        return root

    def policy(self, root: Node, board_size: int) -> np.ndarray:
        """Visit-count distribution over size*size + 1 moves."""
        pi = np.zeros(board_size * board_size + 1, dtype=np.float32)
        for move, child in root.children.items():
            pi[move] = child.visits
        total = pi.sum()
        return pi / total if total > 0 else pi

    def select_move(self, root: Node, board_size: int,
                    temperature: float = 0.0) -> int:
        pi = self.policy(root, board_size)
        if temperature == 0.0:
            return int(pi.argmax())
        pi = pi ** (1.0 / temperature)
        pi /= pi.sum()
        return int(self.rng.choice(len(pi), p=pi))

    def _simulate(self, root: Node, board: Board, to_play: Stone) -> None:
        path, board, color = self.descend(root, board, to_play)
        if board.is_terminal():
            value = terminal_value(board, color)
        else:
            priors, value = self.evaluate(board, color)
            value = self.expand(path[-1], board, color, priors, value)
        self.backprop(path, value)

    def descend(self, root: Node, board: Board,
                to_play: Stone) -> tuple[list[Node], Board, Stone]:
        """Walk to a leaf, mutating `board` along the way."""
        node = root
        color = to_play
        path = [root]
        while node.children:
            move, node = self._select_child(node)
            apply_move(board, move, color)
            color = goboard.opponent(color)
            path.append(node)
        return path, board, color

    def backprop(self, path: list[Node], value: float,
                 visit_delta: int = 1) -> None:
        """`value` is from the perspective of the player to move at the
        leaf; the sign flips at every step back up. visit_delta=0 lets
        a virtual loss be corrected without recounting the visit."""
        for node in reversed(path):
            node.visits += visit_delta
            node.value_sum += value
            value = -value

    def _select_child(self, node: Node) -> tuple[int, "Node"]:
        sqrt_visits = math.sqrt(node.visits)
        best_score = -float("inf")
        best = None
        for move, child in node.children.items():
            q = -child.value()
            u = self.c_puct * child.prior * sqrt_visits / (1 + child.visits)
            if q + u > best_score:
                best_score = q + u
                best = (move, child)
        return best

    def expand(self, node: Node, board: Board, to_play: Stone,
               priors: np.ndarray, value: float) -> float:
        pass_move = board.size * board.size
        legal = board.legal_moves(to_play) + [pass_move]

        legal_priors = np.asarray([priors[m] for m in legal], dtype=np.float64)
        total = legal_priors.sum()
        if total > 0:
            legal_priors /= total
        else:
            legal_priors[:] = 1.0 / len(legal)
        for move, prior in zip(legal, legal_priors):
            node.children[move] = Node(float(prior))
        return float(value)

    def add_dirichlet_noise(self, root: Node) -> None:
        moves = list(root.children)
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(moves))
        for move, eta in zip(moves, noise):
            child = root.children[move]
            child.prior = ((1 - self.noise_fraction) * child.prior
                           + self.noise_fraction * eta)


def apply_move(board: Board, move: int, color: Stone) -> None:
    """Apply a flattened move index; size*size means pass."""
    size = board.size
    if move == size * size:
        board.pass_()
        return
    if not board.play(move % size, move // size, color):
        raise ValueError(f"illegal move {move} for {color}")


def uniform_evaluate(board: Board, to_play: Stone) -> tuple[np.ndarray, float]:
    """Network-free evaluator, useful for tests and sanity checks."""
    n = board.size * board.size + 1
    return np.full(n, 1.0 / n, dtype=np.float32), 0.0
