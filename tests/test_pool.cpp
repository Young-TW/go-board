#include <cmath>
#include <iostream>
#include <vector>

#include "selfplay_pool.h"

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::cout << "FAIL " << __func__ << " (" << __FILE__ << ":"   \
                      << __LINE__ << "): " #cond "\n";                    \
            failures++;                                                   \
        }                                                                 \
    } while (0)

static void test_pool_plays_games_to_completion() {
    const int board_size = 5;
    const int points = board_size * board_size;
    const int stride = points + 1;
    SelfPlayPool pool(/*n_games=*/3, board_size, /*komi=*/7.5f,
                      /*simulations=*/12, /*cheap_simulations=*/6,
                      /*full_search_prob=*/0.5f, /*temperature_moves=*/4,
                      /*leaves_per_game=*/4, /*parallel=*/2,
                      /*c_puct=*/1.5f, /*dirichlet_alpha=*/0.3f,
                      /*noise_fraction=*/0.25f, /*seed=*/42);

    int rounds = 0;
    while (!pool.done() && rounds < 200000) {
        rounds++;
        const int count = pool.collect();
        if (count == 0) continue;
        CHECK(pool.features().size() ==
              std::size_t(count) * Board::kFeaturePlanes * points);
        std::vector<float> priors(std::size_t(count) * stride,
                                  1.0f / stride);
        std::vector<float> values(count, 0.0f);
        pool.submit(priors.data(), values.data(), count);
    }
    CHECK(pool.done());

    const auto results = pool.take_results();
    CHECK(results.size() == 3u);
    int full = 0;
    int cheap = 0;
    for (const GameResult& result : results) {
        for (const SampleRec& sample : result.samples) {
            (sample.train_policy ? full : cheap)++;
        }
    }
    CHECK(full > 0);   // full_search_prob = 0.5 should yield both
    CHECK(cheap > 0);
    for (const GameResult& result : results) {
        CHECK(!result.samples.empty());
        CHECK(result.black_margin != 0.0f);  // komi 7.5: no draws
        const bool black_won = result.black_margin > 0;
        for (const SampleRec& sample : result.samples) {
            CHECK(sample.features.size() ==
                  std::size_t(Board::kFeaturePlanes) * points);
            CHECK(sample.pi.size() == std::size_t(stride));
            float total = 0.0f;
            for (float p : sample.pi) {
                CHECK(p >= 0.0f);
                total += p;
            }
            CHECK(std::abs(total - 1.0f) < 1e-4f);
            const float expected =
                black_won == (sample.to_play == Stone::Black) ? 1.0f
                                                              : -1.0f;
            CHECK(sample.z == expected);
        }
    }
}

static void test_policy_target_pruning() {
    // Root with three children: visits 10 / 5 / 1, root.visits = 16.
    Node root;
    root.visits = 16;
    const float priors[] = {0.2f, 0.3f, 0.5f};
    const int visits[] = {10, 5, 1};
    for (int i = 0; i < 3; i++) {
        Edge edge{i, priors[i], std::make_unique<Node>()};
        edge.child->visits = visits[i];
        root.edges.push_back(std::move(edge));
    }
    Search search(1.5f, 0.3f, 0.25f, 1);

    const auto raw = search.policy(root, 4);
    CHECK(std::abs(raw[0] - 10.0f / 16.0f) < 1e-5f);

    // k=2: child 1 keeps max(0, 5 - 2*sqrt(.3*16)) = 0.62 -> <1 -> 0;
    // child 2 is pruned to 0 as well; the best child keeps everything.
    const auto pruned = search.policy(root, 4, 2.0f);
    CHECK(pruned[0] == 1.0f);
    CHECK(pruned[1] == 0.0f);
    CHECK(pruned[2] == 0.0f);
}

int main() {
    test_policy_target_pruning();
    test_pool_plays_games_to_completion();
    if (failures == 0) {
        std::cout << "all tests passed\n";
        return 0;
    }
    std::cout << failures << " check(s) failed\n";
    return 1;
}
