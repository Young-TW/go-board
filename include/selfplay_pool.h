#ifndef GO_BOARD_SELFPLAY_POOL_H
#define GO_BOARD_SELFPLAY_POOL_H

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "board.h"
#include "mcts.h"

struct SampleRec {
    std::vector<float> features;  // kFeaturePlanes * points
    std::vector<float> pi;        // points + 1
    Stone to_play;
    float z = 0.0f;
    // Playout cap randomization: only moves searched with the full
    // budget produce policy targets worth training on.
    bool train_policy = true;
    // Auxiliary targets, both from to_play's perspective: final
    // per-point ownership in {-1, 0, 1} and the final score margin.
    // Resigned games have no final position, so theirs carry no
    // weight (has_ownership = false).
    std::vector<float> ownership;  // points
    float score_target = 0.0f;
    bool has_ownership = true;
};

struct GameResult {
    std::vector<SampleRec> samples;
    float black_margin = 0.0f;
};

// Drives n_games self-play games with batched evaluation, entirely in
// C++. Usage from the training loop:
//
//     while (!pool.done()) {
//         int count = pool.collect();   // builds feature batch
//         if (count) pool.submit(priors, values, count);
//     }
//     auto results = pool.take_results();
//
// collect() advances every game (playing moves whose root reached the
// simulation target, finishing/replacing games), then gathers up to
// leaves_per_game descents per game under virtual loss plus root
// expansion requests, exposing their features via features().
class SelfPlayPool {
public:
    // simulations is the full search budget; a move gets it with
    // probability full_search_prob, otherwise cheap_simulations
    // (playout cap randomization). Root noise applies only to
    // full-search moves. A side whose root value drops below
    // -resign_threshold resigns (>= 1 disables); a no_resign_fraction
    // of games always plays to the end to calibrate false positives.
    // komi_jitter > 0 randomizes each game's komi to
    // komi +/- jitter in half-point steps, so the net (fed komi as a
    // feature plane) learns the komi/initiative trade-off instead of
    // collapsing into a fixed-komi equilibrium.
    // min_pass_moves > 0 removes the pass edge from searches before
    // that move number, a self-play guard against the early pass-out
    // equilibrium (both sides learning to end near-empty games).
    SelfPlayPool(int n_games, int board_size, float komi, int simulations,
                 int cheap_simulations, float full_search_prob,
                 int temperature_moves, int leaves_per_game, int parallel,
                 float c_puct, float dirichlet_alpha, float noise_fraction,
                 float resign_threshold, float no_resign_fraction,
                 float komi_jitter, int min_pass_moves,
                 std::uint64_t seed);

    bool done() const { return slots_.empty(); }
    int board_size() const { return board_size_; }

    int collect();
    const std::vector<float>& features() const { return features_; }

    // Live view of the game slots, for spectator UIs.
    int slot_count() const { return int(slots_.size()); }
    std::string spectate_board(int slot) const;
    int spectate_moves(int slot) const;
    bool spectate_black_to_play(int slot) const;
    int games_started() const { return started_; }
    int games_finished() const { return int(results_.size()); }

    // priors: count rows of points+1; values: count entries, from the
    // perspective of the side to play at the evaluated position.
    void submit(const float* priors, const float* values, int count);

    std::vector<GameResult> take_results();

    // Resign calibration from the no-resign games: how many would
    // have resigned, and how often the would-resigner actually won.
    int resign_calibration_games() const { return calibration_games_; }
    int resign_false_positives() const { return calibration_wrong_; }

    // Fixed probe threshold used for calibration tracking, independent
    // of the active resign_threshold.
    static constexpr float kResignProbe = 0.95f;

    // Evaluation cache statistics for this pool's lifetime.
    long eval_cache_lookups() const { return cache_lookups_; }
    long eval_cache_hits() const { return cache_hits_; }
    long early_terminations() const { return early_stops_; }

private:
    struct GameSlot {
        Board board;
        Stone to_play = Stone::Black;
        std::unique_ptr<Node> root;
        int move_count = 0;
        // Fresh descents still owed before the next move is played.
        // A reused subtree keeps its visits (deeper search) but never
        // substitutes for new simulations, or exploration collapses.
        int sims_left = 0;
        bool full_search = true;
        bool allow_resign = true;
        // First side that crossed the resign threshold in a no-resign
        // (calibration) game; Empty means none yet.
        Stone would_resign = Stone::Empty;
        std::vector<SampleRec> samples;

        explicit GameSlot(int size, float komi) : board(size, komi) {}
    };

    struct Pending {
        GameSlot* slot;            // root expansion target, or nullptr
        std::vector<Node*> path;   // leaf path (empty for root requests)
        Board board;               // position that was evaluated
        Stone to_play;
        std::uint64_t cache_key;
        bool allow_pass;
    };

    bool pass_allowed(const GameSlot& game) const {
        return game.move_count >= min_pass_moves_;
    }

    struct CacheEntry {
        std::vector<float> priors;  // raw net output, pre-noise
        float value;
    };

    std::unique_ptr<GameSlot> new_game();
    float draw_komi();
    void begin_move(GameSlot& game);
    void play_move(GameSlot& game);
    // Returns true when the game ended by resignation.
    bool maybe_resign(GameSlot& game);
    void finish(GameSlot& game);
    void finish_resigned(GameSlot& game, Stone winner);

    int n_games_;
    int board_size_;
    float komi_;
    int simulations_;
    int cheap_simulations_;
    float full_search_prob_;
    int temperature_moves_;
    int leaves_per_game_;
    int max_moves_;
    static std::uint64_t cache_key(const Board& board, Stone to_play);
    const CacheEntry* cache_probe(std::uint64_t key);
    void cache_store(std::uint64_t key, const float* priors, float value);

    float resign_threshold_;
    float no_resign_fraction_;
    float komi_jitter_;
    int min_pass_moves_;
    int started_ = 0;
    int calibration_games_ = 0;
    int calibration_wrong_ = 0;
    long cache_lookups_ = 0;
    long cache_hits_ = 0;
    long early_stops_ = 0;
    // Same position + side to play => same features => same net
    // output, so the cache is exact for one net version (one pool).
    std::unordered_map<std::uint64_t, CacheEntry> eval_cache_;
    static constexpr std::size_t kCacheCap = 250000;
    Search search_;
    std::mt19937_64 rng_;
    std::vector<std::unique_ptr<GameSlot>> slots_;
    std::vector<Pending> pending_;
    std::vector<float> features_;
    std::vector<GameResult> results_;
};

#endif
