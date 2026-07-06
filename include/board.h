#ifndef GO_BOARD_BOARD_H
#define GO_BOARD_BOARD_H

#include <cstdint>
#include <unordered_set>
#include <vector>

enum class Stone : std::int8_t { Empty = 0, Black = 1, White = 2 };

Stone opponent(Stone color);

class Board {
public:
    explicit Board(int size = 19);

    int size() const { return size_; }
    bool in_bounds(int x, int y) const;
    Stone at(int x, int y) const;

    // Places a stone and resolves captures. Returns false and leaves the
    // board unchanged if the move is out of bounds, on an occupied point,
    // suicide, or recreates a previous position (positional superko).
    bool play(int x, int y, Stone color);
    bool is_legal(int x, int y, Stone color) const;

    std::uint64_t hash() const { return hash_; }

    void print() const;

private:
    int index(int x, int y) const { return y * size_ + x; }
    int neighbors(int idx, int out[4]) const;
    // Flood-fills the chain containing idx into `stones`, returns its
    // number of distinct liberties.
    int chain_liberties(int idx, std::vector<int>& stones) const;
    void remove_chain(const std::vector<int>& stones);

    int size_;
    std::vector<Stone> point_;
    std::uint64_t hash_ = 0;
    std::unordered_set<std::uint64_t> history_;
};

#endif
