"""Parallel self-play with batched network evaluation.

Throughput comes from three tricks layered on lockstep batching:

- Tree reuse: after a move, the chosen child becomes the next root,
  so its visits count toward the next move's simulation target.
- Virtual loss: each game contributes several concurrent descents per
  round (paths are temporarily penalized so they diverge), multiplying
  the evaluation batch size.
- Continuous pool: finished games are replaced immediately while the
  per-iteration game quota lasts, so the batch never drains to a tail
  of one slow game.
"""

from dataclasses import dataclass, field

import numpy as np

import goboard
from goboard import Board, Stone

from train.mcts import MCTS, Node, apply_move, terminal_value
from train.selfplay import Sample

VIRTUAL_LOSS = -1.0


@dataclass
class _Game:
    board: Board
    to_play: Stone = Stone.BLACK
    root: Node | None = None
    move_count: int = 0
    samples: list[Sample] = field(default_factory=list)


def play_games(evaluate_batch, n_games: int, board_size: int = 9,
               komi: float = 7.5, simulations: int = 128,
               temperature_moves: int = 8, leaves_per_game: int = 4,
               parallel: int | None = None,
               rng: np.random.Generator | None = None):
    """Play n_games self-play games, at most `parallel` concurrently.

    evaluate_batch maps [(board, to_play), ...] to (priors, values)
    arrays. Returns (list_of_sample_lists, list_of_black_margins).
    """
    rng = rng if rng is not None else np.random.default_rng()
    mcts = MCTS(lambda *_: None, rng=rng)  # evaluate goes through batches
    max_moves = board_size * board_size * 2
    all_samples: list[list[Sample]] = []
    margins: list[float] = []
    started = 0

    def new_game() -> _Game:
        nonlocal started
        started += 1
        return _Game(Board(size=board_size, komi=komi))

    def play_move(game: _Game) -> None:
        pi = mcts.policy(game.root, board_size)
        game.samples.append(
            Sample(game.board.features(game.to_play), pi, game.to_play))
        temperature = 1.0 if game.move_count < temperature_moves else 0.0
        move = mcts.select_move(game.root, board_size, temperature)
        child = game.root.children.get(move)
        apply_move(game.board, move, game.to_play)
        game.to_play = goboard.opponent(game.to_play)
        game.move_count += 1
        # Tree reuse: keep the played child's subtree when it has been
        # expanded; noise is refreshed for its new life as the root.
        game.root = child if child is not None and child.children else None
        if game.root is not None:
            mcts.add_dirichlet_noise(game.root)

    def finish(game: _Game) -> None:
        black_margin = game.board.score()
        for sample in game.samples:
            black_won = black_margin > 0
            sample.z = (1.0 if black_won == (sample.to_play == Stone.BLACK)
                        else -1.0)
        all_samples.append(game.samples)
        margins.append(black_margin)

    pool = [new_game() for _ in range(min(parallel or n_games, n_games))]

    while pool:
        requests = []  # (kind, game, path, board, color)
        next_pool = []
        for game in pool:
            # Play out every move whose root already has enough visits.
            done = False
            while (game.root is not None
                   and game.root.visits >= simulations):
                play_move(game)
                if (game.board.is_terminal()
                        or game.move_count >= max_moves):
                    finish(game)
                    done = True
                    break
            if done:
                if started >= n_games:
                    continue
                game = new_game()
            next_pool.append(game)

            if game.root is None:
                requests.append(("root", game, None, game.board,
                                 game.to_play))
                continue
            for _ in range(leaves_per_game):
                path, board, color = mcts.descend(
                    game.root, game.board.copy(), game.to_play)
                if board.is_terminal():
                    mcts.backprop(path, terminal_value(board, color))
                else:
                    # Virtual loss: penalize the path so the next
                    # descent in this round explores elsewhere.
                    mcts.backprop(path, VIRTUAL_LOSS)
                    requests.append(("leaf", game, path, board, color))
        pool = next_pool
        if not requests:
            continue

        priors, values = evaluate_batch(
            [(board, color) for _, _, _, board, color in requests])
        for (kind, game, path, board, color), p, v in zip(requests, priors,
                                                          values):
            if kind == "root":
                root = Node(0.0)
                mcts.expand(root, board, color, p, 0.0)
                mcts.add_dirichlet_noise(root)
                game.root = root
            else:
                leaf = path[-1]
                value = float(v)
                if not leaf.children:
                    value = mcts.expand(leaf, board, color, p, value)
                # Replace the virtual loss with the real value; the
                # visit was already counted.
                mcts.backprop(path, value - VIRTUAL_LOSS, visit_delta=0)

    return all_samples, margins
