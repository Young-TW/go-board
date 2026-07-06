#include <iostream>

#include "board.h"

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::cout << "FAIL " << __func__ << " (" << __FILE__ << ":"   \
                      << __LINE__ << "): " #cond "\n";                    \
            failures++;                                                   \
        }                                                                 \
    } while (0)

constexpr Stone B = Stone::Black;
constexpr Stone W = Stone::White;

static void test_place_and_at() {
    Board board(9);
    CHECK(board.play(2, 3, B));
    CHECK(board.at(2, 3) == B);
    CHECK(board.at(3, 2) == Stone::Empty);
}

static void test_occupied_rejected() {
    Board board(9);
    CHECK(board.play(4, 4, B));
    CHECK(!board.play(4, 4, W));
    CHECK(board.at(4, 4) == B);
}

static void test_out_of_bounds_rejected() {
    Board board(9);
    CHECK(!board.play(-1, 0, B));
    CHECK(!board.play(0, 9, B));
    CHECK(!board.play(0, 0, Stone::Empty));
}

static void test_corner_capture() {
    Board board(9);
    CHECK(board.play(0, 0, B));
    CHECK(board.play(1, 0, W));
    CHECK(board.play(0, 1, W));
    CHECK(board.at(0, 0) == Stone::Empty);
}

static void test_chain_capture() {
    Board board(9);
    // Two-stone black chain at (1,1)-(2,1), fully surrounded by white.
    board.play(1, 1, B);
    board.play(2, 1, B);
    board.play(1, 0, W);
    board.play(2, 0, W);
    board.play(0, 1, W);
    board.play(3, 1, W);
    board.play(1, 2, W);
    CHECK(board.at(1, 1) == B);  // one liberty left at (2,2)
    CHECK(board.play(2, 2, W));
    CHECK(board.at(1, 1) == Stone::Empty);
    CHECK(board.at(2, 1) == Stone::Empty);
}

static void test_suicide_rejected() {
    Board board(9);
    board.play(1, 0, W);
    board.play(0, 1, W);
    CHECK(!board.play(0, 0, B));
    CHECK(board.at(0, 0) == Stone::Empty);
}

static void test_capture_is_not_suicide() {
    Board board(9);
    // White (1,0) is reduced to its last liberty at (0,0); black playing
    // there captures it instead of being suicide.
    board.play(1, 0, W);
    board.play(0, 1, W);
    board.play(2, 0, B);
    board.play(1, 1, B);
    CHECK(board.play(0, 0, B));
    CHECK(board.at(1, 0) == Stone::Empty);
    CHECK(board.at(0, 0) == B);
}

static void test_is_legal_does_not_mutate() {
    Board board(9);
    board.play(1, 0, W);
    board.play(0, 1, W);
    CHECK(!board.is_legal(0, 0, B));
    CHECK(board.is_legal(5, 5, B));
    CHECK(board.at(5, 5) == Stone::Empty);
    CHECK(board.at(1, 0) == W);
}

static void test_ko_recapture_forbidden() {
    Board board(9);
    // Classic ko: black surrounds (1,1), white surrounds (2,1).
    board.play(1, 0, B);
    board.play(0, 1, B);
    board.play(1, 2, B);
    board.play(2, 1, B);
    board.play(2, 0, W);
    board.play(3, 1, W);
    board.play(2, 2, W);

    CHECK(board.play(1, 1, W));  // takes the ko, captures B(2,1)
    CHECK(board.at(2, 1) == Stone::Empty);

    // Immediate recapture would recreate the previous position.
    CHECK(!board.play(2, 1, B));
    CHECK(board.at(1, 1) == W);
    CHECK(board.at(2, 1) == Stone::Empty);

    // The board is still playable elsewhere.
    CHECK(board.play(5, 5, B));
}

static void test_two_passes_end_the_game() {
    Board board(9);
    CHECK(!board.is_terminal());
    board.pass();
    CHECK(!board.is_terminal());
    board.play(4, 4, W);  // a move resets consecutive passes
    board.pass();
    CHECK(!board.is_terminal());
    board.pass();
    CHECK(board.is_terminal());
}

static void test_score_empty_board_is_neutral() {
    Board board(9, 7.5f);
    // An empty region touching no stones belongs to nobody.
    CHECK(board.score() == -7.5f);
}

static void test_score_single_stone_owns_everything() {
    Board board(9, 7.5f);
    board.play(4, 4, B);
    CHECK(board.score() == 81.0f - 7.5f);
}

static void test_score_divided_board() {
    Board board(5, 7.5f);
    // Black wall on column x=2 splits the board; the left region is
    // black territory, the right one touches both colours (neutral).
    for (int y = 0; y < 5; y++) board.play(2, y, B);
    board.play(4, 2, W);
    // black: 5 stones + 10 points territory; white: 1 stone.
    CHECK(board.score() == 15.0f - 1.0f - 7.5f);
}

static void test_legal_moves() {
    Board board(5);
    CHECK(board.legal_moves(B).size() == 25);
    board.play(1, 0, W);
    board.play(0, 1, W);
    // (0,0) is now suicide for black but still legal for white.
    CHECK(board.legal_moves(B).size() == 22);
    CHECK(board.legal_moves(W).size() == 23);
}

static void test_hash_updates_and_restores() {
    Board board(9);
    const std::uint64_t empty_hash = board.hash();
    board.play(3, 3, B);
    CHECK(board.hash() != empty_hash);

    // An illegal move must leave the hash untouched.
    const std::uint64_t before = board.hash();
    CHECK(!board.play(3, 3, W));
    CHECK(board.hash() == before);
}

static void test_features_planes() {
    Board board(5);
    board.play(1, 1, B);
    board.play(3, 3, W);
    const int points = 5 * 5;

    const auto black_view = board.features(B);
    CHECK(black_view.size() == 3u * points);
    CHECK(black_view[1 * 5 + 1] == 1.0f);           // own stone
    CHECK(black_view[points + 3 * 5 + 3] == 1.0f);  // opponent stone
    CHECK(black_view[3 * 5 + 3] == 0.0f);
    CHECK(black_view[2 * points] == 1.0f);  // black-to-play plane

    const auto white_view = board.features(W);
    CHECK(white_view[3 * 5 + 3] == 1.0f);           // own stone
    CHECK(white_view[points + 1 * 5 + 1] == 1.0f);  // opponent stone
    CHECK(white_view[2 * points] == 0.0f);  // white to play
}

int main() {
    test_place_and_at();
    test_occupied_rejected();
    test_out_of_bounds_rejected();
    test_corner_capture();
    test_chain_capture();
    test_suicide_rejected();
    test_capture_is_not_suicide();
    test_is_legal_does_not_mutate();
    test_ko_recapture_forbidden();
    test_hash_updates_and_restores();
    test_two_passes_end_the_game();
    test_score_empty_board_is_neutral();
    test_score_single_stone_owns_everything();
    test_score_divided_board();
    test_legal_moves();
    test_features_planes();

    if (failures == 0) {
        std::cout << "all tests passed\n";
        return 0;
    }
    std::cout << failures << " check(s) failed\n";
    return 1;
}
