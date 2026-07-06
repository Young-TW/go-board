#include "board.h"

#include <array>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

constexpr int kMaxPoints = 19 * 19;

// One random key per (point, color); a position's hash is the XOR of the
// keys of its stones, so it can be updated incrementally on place/remove.
const std::array<std::array<std::uint64_t, 2>, kMaxPoints>& zobrist() {
    static const auto table = [] {
        std::array<std::array<std::uint64_t, 2>, kMaxPoints> t;
        std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
        for (auto& keys : t) {
            keys[0] = rng();
            keys[1] = rng();
        }
        return t;
    }();
    return table;
}

std::uint64_t stone_key(int idx, Stone color) {
    return zobrist()[idx][color == Stone::Black ? 0 : 1];
}

}  // namespace

Stone opponent(Stone color) {
    return color == Stone::Black ? Stone::White : Stone::Black;
}

Board::Board(int size) : size_(size), point_(size * size, Stone::Empty) {
    if (size < 2 || size > 19) {
        throw std::invalid_argument("board size must be between 2 and 19");
    }
    history_.insert(hash_);
}

bool Board::in_bounds(int x, int y) const {
    return x >= 0 && x < size_ && y >= 0 && y < size_;
}

Stone Board::at(int x, int y) const { return point_[index(x, y)]; }

int Board::neighbors(int idx, int out[4]) const {
    const int x = idx % size_;
    const int y = idx / size_;
    int count = 0;
    if (x > 0) out[count++] = idx - 1;
    if (x < size_ - 1) out[count++] = idx + 1;
    if (y > 0) out[count++] = idx - size_;
    if (y < size_ - 1) out[count++] = idx + size_;
    return count;
}

int Board::chain_liberties(int idx, std::vector<int>& stones) const {
    const Stone color = point_[idx];
    std::vector<char> visited(point_.size(), 0);
    std::vector<char> counted(point_.size(), 0);
    int liberties = 0;

    stones.clear();
    stones.push_back(idx);
    visited[idx] = 1;

    for (std::size_t i = 0; i < stones.size(); i++) {
        int nbr[4];
        const int count = neighbors(stones[i], nbr);
        for (int n = 0; n < count; n++) {
            const int next = nbr[n];
            if (point_[next] == Stone::Empty) {
                if (!counted[next]) {
                    counted[next] = 1;
                    liberties++;
                }
            } else if (point_[next] == color && !visited[next]) {
                visited[next] = 1;
                stones.push_back(next);
            }
        }
    }
    return liberties;
}

void Board::remove_chain(const std::vector<int>& stones) {
    for (int idx : stones) point_[idx] = Stone::Empty;
}

bool Board::play(int x, int y, Stone color) {
    if (color == Stone::Empty || !in_bounds(x, y)) return false;
    const int idx = index(x, y);
    if (point_[idx] != Stone::Empty) return false;

    point_[idx] = color;

    // Capture adjacent opponent chains left without liberties.
    const Stone enemy = opponent(color);
    std::vector<int> captured;
    std::vector<int> chain;
    int nbr[4];
    const int count = neighbors(idx, nbr);
    for (int n = 0; n < count; n++) {
        if (point_[nbr[n]] != enemy) continue;
        if (chain_liberties(nbr[n], chain) == 0) {
            remove_chain(chain);
            captured.insert(captured.end(), chain.begin(), chain.end());
        }
    }

    // A move that captures always gains a liberty, so suicide can only
    // happen when nothing was captured.
    if (captured.empty() && chain_liberties(idx, chain) == 0) {
        point_[idx] = Stone::Empty;
        return false;
    }

    // Positional superko: reject any move recreating a previous position.
    std::uint64_t new_hash = hash_ ^ stone_key(idx, color);
    for (int c : captured) new_hash ^= stone_key(c, enemy);
    if (history_.contains(new_hash)) {
        for (int c : captured) point_[c] = enemy;
        point_[idx] = Stone::Empty;
        return false;
    }

    hash_ = new_hash;
    history_.insert(new_hash);
    return true;
}

bool Board::is_legal(int x, int y, Stone color) const {
    Board copy(*this);
    return copy.play(x, y, color);
}

void Board::print() const {
    for (int y = 0; y < size_; y++) {
        for (int x = 0; x < size_; x++) {
            switch (at(x, y)) {
                case Stone::Black: std::cout << "X "; break;
                case Stone::White: std::cout << "O "; break;
                default: std::cout << ". "; break;
            }
        }
        std::cout << "\n";
    }
}
