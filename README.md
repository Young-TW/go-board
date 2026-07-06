# go-board

A Go (圍棋) rules engine in C++ with an AlphaZero-style self-play
training stack (PyTorch + MCTS) on top, plus a simple terminal client.

## Rules implemented

- Board sizes 2–19 (9/13/19 for real games)
- Liberty tracking, capture, and the suicide rule
- Positional superko via incremental Zobrist hashing (covers simple ko)
- Pass; the game ends after two consecutive passes
- Tromp-Taylor area scoring with configurable komi (default 7.5)
- Legal move generation (`Board::legal_moves`)

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces the `go-board` terminal client and a `go_board_core`
static library that other programs (e.g. Python bindings for a
training loop) can link against.

## Python bindings

The recommended path is uv + scikit-build-core, which builds the
`goboard` extension automatically:

```bash
uv sync          # builds and installs goboard into .venv
uv run python -c "import goboard"
```

After changing C++ sources, rebuild with
`uv sync --reinstall-package goboard`.

Manual CMake build also works
(`cmake -B build -DGO_BOARD_PYTHON=ON`, module lands in `build/`,
use with `PYTHONPATH=build`).

```python
import goboard
from goboard import Board, Stone

board = Board(size=9, komi=7.5)
board.play(2, 3, Stone.BLACK)
board.legal_moves(Stone.WHITE)     # flattened indices
board.features(Stone.WHITE)        # float32 ndarray, shape (3, 9, 9)
board.pass_()                      # 'pass' is a Python keyword
board.is_terminal(); board.score(); board.hash()
print(board)
```

## Training (AlphaZero-style)

`train/` contains the self-play stack: `net.py` (policy/value ResNet),
`mcts.py` (PUCT search), `selfplay.py` (game generation + symmetry
augmentation), `train.py` (the loop). torch comes from the ROCm wheel
index via the `train` dependency group:

```bash
uv sync --group train
uv run python -m train.train --board-size 9 --iterations 50
```

Checkpoints land in `checkpoints/` (gitignored); resume with
`--resume checkpoints/iter_0049.pt`.

## Test

```bash
ctest --test-dir build --output-on-failure   # C++ + bindings
uv run pytest                                # Python training stack
```

## Play in the terminal

```bash
./build/go-board
```

Enter moves as `x y` (1-based), `pass` to pass, `quit` to abort.
Two consecutive passes end the game and print the Tromp-Taylor result.

## API sketch

```cpp
Board board(9, 7.5f);              // size, komi
board.play(2, 3, Stone::Black);    // resolves captures; false if illegal
board.is_legal(x, y, color);
board.legal_moves(Stone::White);   // flattened indices (y * size + x)
board.pass();
board.is_terminal();               // two consecutive passes
board.score();                     // black's margin, komi included
board.hash();                      // Zobrist hash of the position
```
