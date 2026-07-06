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

Board::Board(int size, float komi)
    : size_(size), komi_(komi), point_(size * size, Stone::Empty) {
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
    consecutive_passes_ = 0;
    return true;
}

void Board::pass() { consecutive_passes_++; }

float Board::score() const {
    int black = 0;
    int white = 0;
    std::vector<char> visited(point_.size(), 0);
    std::vector<int> region;

    for (int start = 0; start < static_cast<int>(point_.size()); start++) {
        if (point_[start] == Stone::Black) {
            black++;
        } else if (point_[start] == Stone::White) {
            white++;
        } else if (!visited[start]) {
            // Flood-fill the empty region and note which colours it touches.
            region.clear();
            region.push_back(start);
            visited[start] = 1;
            bool touches_black = false;
            bool touches_white = false;
            for (std::size_t i = 0; i < region.size(); i++) {
                int nbr[4];
                const int count = neighbors(region[i], nbr);
                for (int n = 0; n < count; n++) {
                    const int next = nbr[n];
                    if (point_[next] == Stone::Black) {
                        touches_black = true;
                    } else if (point_[next] == Stone::White) {
                        touches_white = true;
                    } else if (!visited[next]) {
                        visited[next] = 1;
                        region.push_back(next);
                    }
                }
            }
            if (touches_black && !touches_white) {
                black += region.size();
            } else if (touches_white && !touches_black) {
                white += region.size();
            }
        }
    }
    return static_cast<float>(black - white) - komi_;
}

std::vector<int> Board::legal_moves(Stone color) const {
    std::vector<int> moves;
    for (int y = 0; y < size_; y++) {
        for (int x = 0; x < size_; x++) {
            if (is_legal(x, y, color)) moves.push_back(index(x, y));
        }
    }
    return moves;
}

std::vector<float> Board::features(Stone to_play) const {
    const int points = size_ * size_;
    const Stone enemy = opponent(to_play);
    std::vector<float> planes(3 * points, 0.0f);
    for (int i = 0; i < points; i++) {
        if (point_[i] == to_play) {
            planes[i] = 1.0f;
        } else if (point_[i] == enemy) {
            planes[points + i] = 1.0f;
        }
    }
    if (to_play == Stone::Black) {
        for (int i = 0; i < points; i++) planes[2 * points + i] = 1.0f;
    }
    return planes;
}

// Mirrors play() without mutating: cheap enough to call for every
// point when generating legal moves (no board/history copy).
bool Board::is_legal(int x, int y, Stone color) const {
    if (color == Stone::Empty || !in_bounds(x, y)) return false;
    const int idx = index(x, y);
    if (point_[idx] != Stone::Empty) return false;

    const Stone enemy = opponent(color);
    std::uint64_t new_hash = hash_ ^ stone_key(idx, color);

    // An adjacent enemy chain is captured iff its single liberty is
    // this point (idx is empty and adjacent, so at one liberty it is
    // exactly idx). `seen` avoids re-flooding a chain reachable via
    // two neighbors, which would corrupt the hash.
    bool captures = false;
    std::vector<char> seen(point_.size(), 0);
    std::vector<int> chain;
    int nbr[4];
    const int count = neighbors(idx, nbr);
    for (int n = 0; n < count; n++) {
        const int next = nbr[n];
        if (point_[next] != enemy || seen[next]) continue;
        const bool dead = chain_liberties(next, chain) == 1;
        for (int c : chain) {
            seen[c] = 1;
            if (dead) new_hash ^= stone_key(c, enemy);
        }
        captures |= dead;
    }

    if (!captures) {
        // Suicide unless the stone keeps a liberty: an empty neighbor,
        // or a friendly chain with a liberty besides this point.
        bool has_liberty = false;
        for (int n = 0; n < count && !has_liberty; n++) {
            const int next = nbr[n];
            if (point_[next] == Stone::Empty) {
                has_liberty = true;
            } else if (point_[next] == color && !seen[next]) {
                if (chain_liberties(next, chain) >= 2) has_liberty = true;
                for (int c : chain) seen[c] = 1;
            }
        }
        if (!has_liberty) return false;
    }

    return !history_.contains(new_hash);
}

std::string Board::to_string() const {
    std::string out;
    out.reserve((size_ * 2 + 1) * size_);
    for (int y = 0; y < size_; y++) {
        for (int x = 0; x < size_; x++) {
            switch (at(x, y)) {
                case Stone::Black: out += "X "; break;
                case Stone::White: out += "O "; break;
                default: out += ". "; break;
            }
        }
        out += "\n";
    }
    return out;
}

void Board::print() const { std::cout << to_string(); }
