#include "mcts.h"

#include <algorithm>
#include <cmath>

void apply_move(Board& board, int move, Stone color) {
    const int size = board.size();
    if (move == size * size) {
        board.pass();
    } else {
        board.play(move % size, move / size, color);
    }
}

float terminal_value(const Board& board, Stone to_play) {
    const float margin = board.score();
    if (margin == 0.0f) return 0.0f;  // jigo (integer komi)
    return (margin > 0) == (to_play == Stone::Black) ? 1.0f : -1.0f;
}

Search::Search(float c_puct, float dirichlet_alpha, float noise_fraction,
               std::uint64_t seed)
    : c_puct_(c_puct),
      dirichlet_alpha_(dirichlet_alpha),
      noise_fraction_(noise_fraction),
      rng_(seed) {}

std::vector<Node*> Search::descend(Node& root, Board& board,
                                   Stone& to_play,
                                   bool force_at_root) const {
    std::vector<Node*> path;
    path.push_back(&root);
    Node* node = &root;
    bool at_root = true;
    while (node->expanded()) {
        Edge* best = nullptr;
        if (at_root && force_at_root) {
            double best_deficit = 0.0;
            for (Edge& edge : node->edges) {
                const double forced =
                    kForcedPlayoutK *
                    std::sqrt(double(edge.prior) * double(node->visits));
                const double deficit = forced - edge.child->visits;
                if (deficit > best_deficit) {
                    best_deficit = deficit;
                    best = &edge;
                }
            }
        }
        if (!best) {
            const double sqrt_visits = std::sqrt(double(node->visits));
            double best_score = -1e30;
            for (Edge& edge : node->edges) {
                const double q = -edge.child->value();
                const double u = c_puct_ * edge.prior * sqrt_visits /
                                 (1 + edge.child->visits);
                if (q + u > best_score) {
                    best_score = q + u;
                    best = &edge;
                }
            }
        }
        at_root = false;
        apply_move(board, best->move, to_play);
        to_play = opponent(to_play);
        node = best->child.get();
        path.push_back(node);
    }
    return path;
}

void Search::expand(Node& node, const Board& board, Stone to_play,
                    const float* priors, bool allow_pass) const {
    const int points = board.size() * board.size();
    std::vector<int> legal = board.legal_moves(to_play);
    if (allow_pass || legal.empty()) {
        legal.push_back(points);
    }

    double total = 0.0;
    for (int move : legal) total += std::max(priors[move], 0.0f);

    node.edges.reserve(legal.size());
    for (int move : legal) {
        const float prior =
            total > 0 ? float(std::max(priors[move], 0.0f) / total)
                      : 1.0f / legal.size();
        node.edges.push_back({move, prior, std::make_unique<Node>()});
    }
}

void Search::backprop(const std::vector<Node*>& path, double value,
                      int visit_delta) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        (*it)->visits += visit_delta;
        (*it)->value_sum += value;
        value = -value;
    }
}

void Search::add_dirichlet_noise(Node& root) {
    std::gamma_distribution<double> gamma(dirichlet_alpha_, 1.0);
    std::vector<double> noise(root.edges.size());
    double total = 0.0;
    for (double& n : noise) {
        n = gamma(rng_);
        total += n;
    }
    if (total <= 0) return;
    for (std::size_t i = 0; i < root.edges.size(); i++) {
        root.edges[i].prior =
            (1 - noise_fraction_) * root.edges[i].prior +
            noise_fraction_ * float(noise[i] / total);
    }
}

std::vector<float> Search::policy(const Node& root, int points,
                                  float prune_forced_k) const {
    std::vector<float> pi(points + 1, 0.0f);
    const Edge* best = nullptr;
    for (const Edge& edge : root.edges) {
        if (!best || edge.child->visits > best->child->visits) {
            best = &edge;
        }
    }
    double total = 0.0;
    for (const Edge& edge : root.edges) {
        double visits = edge.child->visits;
        if (prune_forced_k > 0 && &edge != best) {
            visits -= prune_forced_k * std::sqrt(double(edge.prior) *
                                                 double(root.visits));
            if (visits < 1.0) visits = 0.0;
        }
        pi[edge.move] = float(visits);
        total += visits;
    }
    if (total > 0) {
        for (float& p : pi) p = float(p / total);
    }
    return pi;
}

int Search::select_move(const Node& root, int points, double temperature) {
    std::vector<float> pi = policy(root, points);
    if (temperature <= 0.0) {
        return int(std::max_element(pi.begin(), pi.end()) - pi.begin());
    }
    std::vector<double> weights(pi.size());
    for (std::size_t i = 0; i < pi.size(); i++) {
        weights[i] = pi[i] > 0 ? std::pow(double(pi[i]), 1.0 / temperature)
                               : 0.0;
    }
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return dist(rng_);
}
