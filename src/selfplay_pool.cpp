#include "selfplay_pool.h"

#include <algorithm>
#include <utility>

namespace {
constexpr double kVirtualLoss = -1.0;
}

SelfPlayPool::SelfPlayPool(int n_games, int board_size, float komi,
                           int simulations, int temperature_moves,
                           int leaves_per_game, int parallel, float c_puct,
                           float dirichlet_alpha, float noise_fraction,
                           std::uint64_t seed)
    : n_games_(n_games),
      board_size_(board_size),
      komi_(komi),
      simulations_(simulations),
      temperature_moves_(temperature_moves),
      leaves_per_game_(leaves_per_game),
      max_moves_(board_size * board_size * 2),
      search_(c_puct, dirichlet_alpha, noise_fraction, seed) {
    const int pool_size =
        std::min(parallel > 0 ? parallel : n_games, n_games);
    for (int i = 0; i < pool_size; i++) slots_.push_back(new_game());
}

std::unique_ptr<SelfPlayPool::GameSlot> SelfPlayPool::new_game() {
    started_++;
    return std::make_unique<GameSlot>(board_size_, komi_);
}

void SelfPlayPool::play_move(GameSlot& game) {
    const int points = board_size_ * board_size_;
    SampleRec rec;
    rec.features = game.board.features(game.to_play);
    rec.pi = search_.policy(*game.root, points);
    rec.to_play = game.to_play;
    game.samples.push_back(std::move(rec));

    const double temperature =
        game.move_count < temperature_moves_ ? 1.0 : 0.0;
    const int move = search_.select_move(*game.root, points, temperature);

    // Tree reuse: the played child's subtree survives as the next root.
    std::unique_ptr<Node> child;
    for (Edge& edge : game.root->edges) {
        if (edge.move == move) {
            child = std::move(edge.child);
            break;
        }
    }
    apply_move(game.board, move, game.to_play);
    game.to_play = opponent(game.to_play);
    game.move_count++;
    game.root = child && child->expanded() ? std::move(child) : nullptr;
    if (game.root) search_.add_dirichlet_noise(*game.root);
}

void SelfPlayPool::finish(GameSlot& game) {
    GameResult result;
    result.black_margin = game.board.score();
    const bool black_won = result.black_margin > 0;
    for (SampleRec& sample : game.samples) {
        sample.z =
            black_won == (sample.to_play == Stone::Black) ? 1.0f : -1.0f;
    }
    result.samples = std::move(game.samples);
    results_.push_back(std::move(result));
}

int SelfPlayPool::collect() {
    pending_.clear();
    features_.clear();

    for (std::size_t i = 0; i < slots_.size();) {
        GameSlot* game = slots_[i].get();

        bool finished = false;
        while (game->root && game->root->visits >= simulations_) {
            play_move(*game);
            if (game->board.is_terminal()
                || game->move_count >= max_moves_) {
                finish(*game);
                finished = true;
                break;
            }
        }
        if (finished) {
            if (started_ >= n_games_) {
                slots_.erase(slots_.begin() + i);
                continue;
            }
            slots_[i] = new_game();
            game = slots_[i].get();
        }

        if (!game->root) {
            std::vector<float> f = game->board.features(game->to_play);
            features_.insert(features_.end(), f.begin(), f.end());
            pending_.push_back(
                {game, {}, game->board, game->to_play});
        } else {
            for (int k = 0; k < leaves_per_game_; k++) {
                Board board(game->board);
                Stone color = game->to_play;
                std::vector<Node*> path =
                    search_.descend(*game->root, board, color);
                if (board.is_terminal()) {
                    Search::backprop(path, terminal_value(board, color), 1);
                    continue;
                }
                // Virtual loss: penalize the path so the next descent
                // in this round explores elsewhere.
                Search::backprop(path, kVirtualLoss, 1);
                std::vector<float> f = board.features(color);
                features_.insert(features_.end(), f.begin(), f.end());
                pending_.push_back(
                    {nullptr, std::move(path), std::move(board), color});
            }
        }
        i++;
    }
    return int(pending_.size());
}

void SelfPlayPool::submit(const float* priors, const float* values,
                          int count) {
    const int stride = board_size_ * board_size_ + 1;
    for (int i = 0; i < count && i < int(pending_.size()); i++) {
        Pending& pending = pending_[i];
        const float* row = priors + std::size_t(i) * stride;
        if (pending.slot != nullptr) {  // root expansion
            auto root = std::make_unique<Node>();
            search_.expand(*root, pending.board, pending.to_play, row);
            search_.add_dirichlet_noise(*root);
            pending.slot->root = std::move(root);
        } else {
            Node* leaf = pending.path.back();
            if (!leaf->expanded()) {
                search_.expand(*leaf, pending.board, pending.to_play, row);
            }
            // Replace the virtual loss with the real value; the visit
            // was already counted.
            Search::backprop(pending.path, values[i] - kVirtualLoss, 0);
        }
    }
    pending_.clear();
}

std::vector<GameResult> SelfPlayPool::take_results() {
    return std::move(results_);
}
