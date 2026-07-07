"""Parallel self-play driven by the C++ SelfPlayPool.

The whole game loop — PUCT search with tree reuse and virtual loss,
move selection, game replacement — runs in C++ (see selfplay_pool.h);
Python only evaluates the feature batches on the GPU:

    while not pool.done():
        planes = pool.collect()
        pool.submit(*evaluate_planes(planes))
"""

import os
from pathlib import Path

import numpy as np

import goboard
from goboard import Stone

from train.selfplay import Sample


def _write_spectate(pool, path: Path, n_games: int) -> None:
    board, moves, black_to_play, finished, _ = pool.spectate()
    text = (f"moves={moves} to_play={'B' if black_to_play else 'W'} "
            f"finished={finished}/{n_games}\n{board}")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)  # atomic: readers never see a partial file


def play_games(evaluate_planes, n_games: int, board_size: int = 9,
               komi: float = 7.5, simulations: int = 128,
               temperature_moves: int = 8, leaves_per_game: int = 4,
               parallel: int | None = None, noise_fraction: float = 0.25,
               rng: np.random.Generator | None = None,
               spectate_path: Path | None = None,
               spectate_every: int = 8):
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
        noise_fraction=noise_fraction,
        seed=int(rng.integers(0, 2**63 - 1)))

    rounds = 0
    while not pool.done():
        if spectate_path is not None and rounds % spectate_every == 0:
            _write_spectate(pool, spectate_path, n_games)
        rounds += 1
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
