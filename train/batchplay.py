"""Parallel self-play with batched network evaluation.

Plays many games concurrently and steps their MCTS simulations in
lockstep: each round collects one leaf per active game and evaluates
them all in a single forward pass, which is what actually keeps the
GPU busy (single-position evaluation leaves it ~idle).
"""

from dataclasses import dataclass, field

import numpy as np

import goboard
from goboard import Board, Stone

from train.mcts import MCTS, Node, apply_move, terminal_value
from train.selfplay import Sample


@dataclass
class _Game:
    board: Board
    to_play: Stone = Stone.BLACK
    root: Node | None = None
    move_count: int = 0
    samples: list[Sample] = field(default_factory=list)
    done: bool = False


def play_games(evaluate_batch, n_games: int, board_size: int = 9,
               komi: float = 7.5, simulations: int = 128,
               temperature_moves: int = 8,
               rng: np.random.Generator | None = None):
    """Play n_games concurrent self-play games.

    evaluate_batch maps [(board, to_play), ...] to (priors, values)
    arrays. Returns (list_of_sample_lists, list_of_black_margins).
    """
    rng = rng if rng is not None else np.random.default_rng()
    mcts = MCTS(lambda *_: None, rng=rng)  # evaluate goes through batches
    games = [_Game(Board(size=board_size, komi=komi))
             for _ in range(n_games)]
    max_moves = board_size * board_size * 2

    while any(not g.done for g in games):
        active = [g for g in games if not g.done]

        # Fresh roots for games starting a new move, expanded in one batch.
        fresh = [g for g in active if g.root is None]
        if fresh:
            priors, values = evaluate_batch(
                [(g.board, g.to_play) for g in fresh])
            for game, p in zip(fresh, priors):
                game.root = Node(0.0)
                mcts.expand(game.root, game.board, game.to_play, p, 0.0)
                mcts.add_dirichlet_noise(game.root)

        # One simulation per active game per round, evaluated together.
        for _ in range(simulations):
            pending = []  # (game, path, leaf board, leaf color)
            for game in active:
                path, board, color = mcts.descend(
                    game.root, game.board.copy(), game.to_play)
                if board.is_terminal():
                    mcts.backprop(path, terminal_value(board, color))
                else:
                    pending.append((game, path, board, color))
            if not pending:
                continue
            priors, values = evaluate_batch(
                [(board, color) for _, _, board, color in pending])
            for (game, path, board, color), p, v in zip(
                    pending, priors, values):
                value = mcts.expand(path[-1], board, color, p, float(v))
                mcts.backprop(path, value)

        # Every active game now makes its move.
        for game in active:
            pi = mcts.policy(game.root, board_size)
            game.samples.append(
                Sample(game.board.features(game.to_play), pi, game.to_play))
            temperature = (1.0 if game.move_count < temperature_moves
                           else 0.0)
            move = mcts.select_move(game.root, board_size, temperature)
            apply_move(game.board, move, game.to_play)
            game.to_play = goboard.opponent(game.to_play)
            game.move_count += 1
            game.root = None
            if game.board.is_terminal() or game.move_count >= max_moves:
                game.done = True

    margins = []
    for game in games:
        black_margin = game.board.score()
        margins.append(black_margin)
        for sample in game.samples:
            black_won = black_margin > 0
            sample.z = (1.0 if black_won == (sample.to_play == Stone.BLACK)
                        else -1.0)
    return [game.samples for game in games], margins
