#ifndef GO_BOARD_SELFPLAY_POOL_H
#define GO_BOARD_SELFPLAY_POOL_H

#include <cstdint>
#include <memory>
#include <vector>

#include "board.h"
#include "mcts.h"

struct SampleRec {
    std::vector<float> features;  // kFeaturePlanes * points
    std::vector<float> pi;        // points + 1
    Stone to_play;
    float z = 0.0f;
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
    SelfPlayPool(int n_games, int board_size, float komi, int simulations,
                 int temperature_moves, int leaves_per_game, int parallel,
                 float c_puct, float dirichlet_alpha, float noise_fraction,
                 std::uint64_t seed);

    bool done() const { return slots_.empty(); }
    int board_size() const { return board_size_; }

    int collect();
    const std::vector<float>& features() const { return features_; }

    // Live view of the first game slot, for spectator UIs.
    std::string spectate_board() const;
    int spectate_moves() const;
    bool spectate_black_to_play() const;
    int games_started() const { return started_; }
    int games_finished() const { return int(results_.size()); }

    // priors: count rows of points+1; values: count entries, from the
    // perspective of the side to play at the evaluated position.
    void submit(const float* priors, const float* values, int count);

    std::vector<GameResult> take_results();

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
        std::vector<SampleRec> samples;

        explicit GameSlot(int size, float komi) : board(size, komi) {}
    };

    struct Pending {
        GameSlot* slot;            // root expansion target, or nullptr
        std::vector<Node*> path;   // leaf path (empty for root requests)
        Board board;               // position that was evaluated
        Stone to_play;
    };

    std::unique_ptr<GameSlot> new_game();
    void play_move(GameSlot& game);
    void finish(GameSlot& game);

    int n_games_;
    int board_size_;
    float komi_;
    int simulations_;
    int temperature_moves_;
    int leaves_per_game_;
    int max_moves_;
    int started_ = 0;
    Search search_;
    std::vector<std::unique_ptr<GameSlot>> slots_;
    std::vector<Pending> pending_;
    std::vector<float> features_;
    std::vector<GameResult> results_;
};

#endif
