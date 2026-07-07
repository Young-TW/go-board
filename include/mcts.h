#ifndef GO_BOARD_MCTS_H
#define GO_BOARD_MCTS_H

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "board.h"

struct Edge;

struct Node {
    int visits = 0;
    double value_sum = 0.0;
    std::vector<Edge> edges;

    bool expanded() const { return !edges.empty(); }
    double value() const { return visits ? value_sum / visits : 0.0; }
};

struct Edge {
    int move;  // flattened y * size + x; size * size means pass
    float prior;
    std::unique_ptr<Node> child;
};

// Applies a flattened move; the caller guarantees legality.
void apply_move(Board& board, int move, Stone color);

// Outcome of a finished game for the side to play, in {-1, 1}.
float terminal_value(const Board& board, Stone to_play);

// PUCT search primitives shared by the self-play pool. Values stored
// in a node are from the perspective of the player to move at that
// node, so selection scores an edge with the negated child value.
class Search {
public:
    Search(float c_puct, float dirichlet_alpha, float noise_fraction,
           std::uint64_t seed);

    // Walks to a leaf, mutating `board` and `to_play` along the way.
    std::vector<Node*> descend(Node& root, Board& board,
                               Stone& to_play) const;

    // Creates edges for every legal move using `priors` (an array of
    // size*size + 1 move probabilities), masked and renormalized.
    void expand(Node& node, const Board& board, Stone to_play,
                const float* priors) const;

    // `value` is from the perspective of the player to move at the
    // leaf; the sign flips at every step back up. visit_delta = 0
    // corrects a virtual loss without recounting the visit.
    static void backprop(const std::vector<Node*>& path, double value,
                         int visit_delta);

    void add_dirichlet_noise(Node& root);

    // Visit-count distribution over size*size + 1 moves.
    std::vector<float> policy(const Node& root, int points) const;

    int select_move(const Node& root, int points, double temperature);

private:
    float c_puct_;
    float dirichlet_alpha_;
    float noise_fraction_;
    mutable std::mt19937_64 rng_;
};

#endif
