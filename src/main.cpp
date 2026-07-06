#include <iostream>
#include <string>

#include "board.h"

int main() {
    Board board(19);
    Stone to_play = Stone::Black;

    board.print();
    std::cout << "enter moves as \"x y\" (1-based), or \"quit\"\n";

    std::string token;
    while (std::cin >> token) {
        if (token == "quit") break;

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
    return 0;
}
