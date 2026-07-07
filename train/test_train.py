from collections import deque
from pathlib import Path

import numpy as np

from train.train import load_buffer, save_buffer


def test_buffer_roundtrip(tmp_path):
    buffer = deque(maxlen=100)
    rng = np.random.default_rng(0)
    for i in range(10):
        buffer.append((rng.random((7, 5, 5)).astype(np.float32),
                       rng.random(26).astype(np.float32), 1.0,
                       float(i % 2),
                       rng.random(25).astype(np.float32), 3.5))
    path = tmp_path / "buffer.npz"
    save_buffer(buffer, path)

    restored = deque(maxlen=100)
    assert load_buffer(path, restored, board_size=5)
    assert len(restored) == 10
    assert np.array_equal(restored[3][0], buffer[3][0])
    assert np.array_equal(restored[3][1], buffer[3][1])
    assert restored[3][2] == 1.0
    assert restored[3][3] == 1.0
    assert restored[4][3] == 0.0
    assert np.array_equal(restored[3][4], buffer[3][4])
    assert restored[3][5] == 3.5

    # Board size mismatch is refused rather than poisoning the buffer.
    other = deque(maxlen=100)
    assert not load_buffer(path, other, board_size=9)
    assert len(other) == 0
