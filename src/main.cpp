#include <iostream>
#include <string>

#include "board.h"

int main() {
    Board board(19);
    Stone to_play = Stone::Black;

    board.print();
    std::cout << "enter moves as \"x y\" (1-based), \"pass\", or \"quit\"\n";

    std::string token;
    while (!board.is_terminal() && std::cin >> token) {
        if (token == "quit") break;

        if (token == "pass") {
            board.pass();
            to_play = opponent(to_play);
            continue;
        }

        int x, y;
        try {
            x = std::stoi(token);
        } catch (const std::exception&) {
            std::cout << "unknown command\n";
            continue;
        }
        if (!(std::cin >> y)) break;

        if (!board.play(x - 1, y - 1, to_play)) {
            std::cout << "illegal move\n";
            continue;
        }
        to_play = opponent(to_play);
        board.print();
    }

    if (board.is_terminal()) {
        const float score = board.score();
        std::cout << "result: "
                  << (score > 0 ? "B+" : "W+") << (score > 0 ? score : -score)
                  << " (Tromp-Taylor)\n";
    }
    return 0;
}
