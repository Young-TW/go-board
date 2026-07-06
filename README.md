# go-board

A Go (圍棋) rules engine in C++, intended as the game-logic core for
self-play training (AlphaZero-style) and as a simple terminal client.

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

## Test

```bash
ctest --test-dir build --output-on-failure
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
