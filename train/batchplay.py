"""Parallel self-play driven by the C++ SelfPlayPool.

The whole game loop — PUCT search with tree reuse and virtual loss,
move selection, game replacement — runs in C++ (see selfplay_pool.h);
Python only evaluates the feature batches on the GPU:

    while not pool.done():
        planes = pool.collect()
        pool.submit(*evaluate_planes(planes))
"""

import numpy as np

import goboard
from goboard import Stone

from train.selfplay import Sample


def play_games(evaluate_planes, n_games: int, board_size: int = 9,
               komi: float = 7.5, simulations: int = 128,
               temperature_moves: int = 8, leaves_per_game: int = 4,
               parallel: int | None = None,
               rng: np.random.Generator | None = None):
    """Play n_games self-play games, at most `parallel` concurrently.

    evaluate_planes maps a float32 array (count, FEATURE_PLANES, size,
    size) to (priors, values) arrays. Returns (list_of_sample_lists,
    list_of_black_margins).
    """
    rng = rng if rng is not None else np.random.default_rng()
    pool = goboard.SelfPlayPool(
        n_games, board_size=board_size, komi=komi, simulations=simulations,
        temperature_moves=temperature_moves,
        leaves_per_game=leaves_per_game, parallel=parallel or 0,
        seed=int(rng.integers(0, 2**63 - 1)))

    while not pool.done():
        planes = pool.collect()
        if planes.shape[0] == 0:
            continue
        priors, values = evaluate_planes(planes)
        pool.submit(np.ascontiguousarray(priors, dtype=np.float32),
                    np.ascontiguousarray(values, dtype=np.float32))

    all_samples = []
    margins = []
    for features, pi, z, black_margin in pool.take_results():
        samples = []
        for i in range(features.shape[0]):
            # Feature plane 2 is the black-to-play indicator.
            to_play = Stone.BLACK if features[i, 2].max() > 0.5 \
                else Stone.WHITE
            samples.append(Sample(features[i], pi[i], to_play, float(z[i])))
        all_samples.append(samples)
        margins.append(black_margin)
    return all_samples, margins
