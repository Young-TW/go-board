#include "selfplay_pool.h"

#include <algorithm>
#include <utility>

namespace {
constexpr double kVirtualLoss = -1.0;
constexpr std::uint64_t kBlackToPlayKey = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kWhiteToPlayKey = 0x517CC1B727220A95ULL;
}

std::uint64_t SelfPlayPool::cache_key(const Board& board, Stone to_play) {
    return board.hash() ^
           (to_play == Stone::Black ? kBlackToPlayKey : kWhiteToPlayKey);
}

const SelfPlayPool::CacheEntry* SelfPlayPool::cache_probe(
    std::uint64_t key) {
    cache_lookups_++;
    const auto it = eval_cache_.find(key);
    if (it == eval_cache_.end()) return nullptr;
    cache_hits_++;
    return &it->second;
}

void SelfPlayPool::cache_store(std::uint64_t key, const float* priors,
                               float value) {
    if (eval_cache_.size() >= kCacheCap) return;  // openings cached first
    const int stride = board_size_ * board_size_ + 1;
    eval_cache_.emplace(
        key, CacheEntry{std::vector<float>(priors, priors + stride), value});
}

SelfPlayPool::SelfPlayPool(int n_games, int board_size, float komi,
                           int simulations, int cheap_simulations,
                           float full_search_prob, int temperature_moves,
                           int leaves_per_game, int parallel, float c_puct,
                           float dirichlet_alpha, float noise_fraction,
                           float resign_threshold, float no_resign_fraction,
                           std::uint64_t seed)
    : n_games_(n_games),
      board_size_(board_size),
      komi_(komi),
      simulations_(simulations),
      cheap_simulations_(cheap_simulations),
      full_search_prob_(full_search_prob),
      temperature_moves_(temperature_moves),
      leaves_per_game_(leaves_per_game),
      max_moves_(board_size * board_size * 2),
      resign_threshold_(resign_threshold),
      no_resign_fraction_(no_resign_fraction),
      search_(c_puct, dirichlet_alpha, noise_fraction, seed),
      rng_(seed + 1) {
    const int pool_size =
        std::min(parallel > 0 ? parallel : n_games, n_games);
    for (int i = 0; i < pool_size; i++) slots_.push_back(new_game());
}

std::unique_ptr<SelfPlayPool::GameSlot> SelfPlayPool::new_game() {
    started_++;
    auto game = std::make_unique<GameSlot>(board_size_, komi_);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    game->allow_resign = uniform(rng_) >= no_resign_fraction_;
    begin_move(*game);
    return game;
}

bool SelfPlayPool::maybe_resign(GameSlot& game) {
    // Never this early (delusional resigns cluster in the opening;
    // genuinely lost positions emerge in the midgame), and always
    // track the fixed probe threshold so false-positive calibration
    // exists even while resignation is disabled.
    if (game.move_count < 3 * board_size_) return false;
    const double value = game.root->value();
    if (value < -kResignProbe && game.would_resign == Stone::Empty) {
        game.would_resign = game.to_play;
    }
    if (resign_threshold_ >= 1.0f) return false;
    if (value >= -resign_threshold_) return false;
    if (!game.allow_resign) return false;
    finish_resigned(game, opponent(game.to_play));
    return true;
}

void SelfPlayPool::begin_move(GameSlot& game) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    game.full_search = uniform(rng_) < full_search_prob_;
    game.sims_left =
        game.full_search ? simulations_ : cheap_simulations_;
    // Root noise belongs to full-search moves only; cheap moves exist
    // to generate game/value data, not policy targets.
    if (game.root && game.full_search) {
        search_.add_dirichlet_noise(*game.root);
    }
}

void SelfPlayPool::play_move(GameSlot& game) {
    const int points = board_size_ * board_size_;
    SampleRec rec;
    rec.features = game.board.features(game.to_play);
    rec.pi = search_.policy(
        *game.root, points,
        game.full_search ? Search::kForcedPlayoutK : -1.0f);
    rec.to_play = game.to_play;
    rec.train_policy = game.full_search;
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
    begin_move(game);
}

void SelfPlayPool::finish_resigned(GameSlot& game, Stone winner) {
    GameResult result;
    // Sentinel: no scored final position exists for a resigned game.
    result.black_margin = winner == Stone::Black ? 10000.0f : -10000.0f;
    const int points = board_size_ * board_size_;
    for (SampleRec& sample : game.samples) {
        sample.z = sample.to_play == winner ? 1.0f : -1.0f;
        sample.has_ownership = false;
        sample.ownership.assign(points, 0.0f);
        sample.score_target = 0.0f;
    }
    result.samples = std::move(game.samples);
    results_.push_back(std::move(result));
}

void SelfPlayPool::finish(GameSlot& game) {
    GameResult result;
    result.black_margin = game.board.score();
    const bool black_won = result.black_margin > 0;
    const std::vector<std::int8_t> owner = game.board.ownership();
    if (game.would_resign != Stone::Empty) {
        calibration_games_++;
        if (result.black_margin != 0.0f
            && black_won == (game.would_resign == Stone::Black)) {
            calibration_wrong_++;  // the would-resigner actually won
        }
    }
    for (SampleRec& sample : game.samples) {
        const float sign = sample.to_play == Stone::Black ? 1.0f : -1.0f;
        if (result.black_margin == 0.0f) {
            sample.z = 0.0f;  // jigo
        } else {
            sample.z = black_won == (sample.to_play == Stone::Black)
                           ? 1.0f
                           : -1.0f;
        }
        sample.score_target = sign * result.black_margin;
        sample.ownership.resize(owner.size());
        for (std::size_t i = 0; i < owner.size(); i++) {
            sample.ownership[i] = sign * owner[i];
        }
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
        while (game->root && game->sims_left <= 0) {
            if (maybe_resign(*game)) {
                finished = true;
                break;
            }
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
            const std::uint64_t key = cache_key(game->board, game->to_play);
            if (const CacheEntry* entry = cache_probe(key)) {
                auto root = std::make_unique<Node>();
                search_.expand(*root, game->board, game->to_play,
                               entry->priors.data());
                if (game->full_search) {
                    search_.add_dirichlet_noise(*root);
                }
                game->root = std::move(root);
                // fall through: descents can start this same round
            } else {
                std::vector<float> f =
                    game->board.features(game->to_play);
                features_.insert(features_.end(), f.begin(), f.end());
                pending_.push_back(
                    {game, {}, game->board, game->to_play, key});
            }
        }
        // Early termination: once the visit lead of the best child
        // exceeds the remaining budget, further search cannot change
        // the chosen move. Only for cheap (non-policy-target) moves
        // past the temperature phase, where selection is argmax.
        if (game->root && game->sims_left > 0 && !game->full_search
            && game->move_count >= temperature_moves_) {
            int best = 0;
            int second = 0;
            for (const Edge& edge : game->root->edges) {
                const int visits = edge.child->visits;
                if (visits > best) {
                    second = best;
                    best = visits;
                } else if (visits > second) {
                    second = visits;
                }
            }
            if (best - second > game->sims_left) {
                game->sims_left = 0;
                early_stops_++;
            }
        }

        if (game->root) {
            const int budget = std::min(leaves_per_game_, game->sims_left);
            for (int k = 0; k < budget; k++) {
                game->sims_left--;
                Board board(game->board);
                Stone color = game->to_play;
                std::vector<Node*> path = search_.descend(
                    *game->root, board, color, game->full_search);
                if (board.is_terminal()) {
                    Search::backprop(path, terminal_value(board, color), 1);
                    continue;
                }
                const std::uint64_t key = cache_key(board, color);
                if (const CacheEntry* entry = cache_probe(key)) {
                    Node* leaf = path.back();
                    if (!leaf->expanded()) {
                        search_.expand(*leaf, board, color,
                                       entry->priors.data());
                    }
                    Search::backprop(path, entry->value, 1);
                    continue;
                }
                // Virtual loss: penalize the path so the next descent
                // in this round explores elsewhere.
                Search::backprop(path, kVirtualLoss, 1);
                std::vector<float> f = board.features(color);
                features_.insert(features_.end(), f.begin(), f.end());
                pending_.push_back(
                    {nullptr, std::move(path), std::move(board), color,
                     key});
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
        cache_store(pending.cache_key, row, values[i]);
        if (pending.slot != nullptr) {  // root expansion
            auto root = std::make_unique<Node>();
            search_.expand(*root, pending.board, pending.to_play, row);
            if (pending.slot->full_search) {
                search_.add_dirichlet_noise(*root);
            }
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

std::string SelfPlayPool::spectate_board() const {
    return slots_.empty() ? std::string() : slots_[0]->board.to_string();
}

int SelfPlayPool::spectate_moves() const {
    return slots_.empty() ? 0 : slots_[0]->move_count;
}

bool SelfPlayPool::spectate_black_to_play() const {
    return !slots_.empty() && slots_[0]->to_play == Stone::Black;
}
