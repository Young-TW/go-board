#include "board.h"

#include <iostream>
#include <vector>

Board::Board() {
    point.resize(19, std::vector<int>(19));
}

int Board::set(int x, int y, int status) {
    point[x][y] = status;
    return 0;
}

int Board::print() {
    for (int x=0;x<19;x++) {
        for (int y=0;y<19;y++) {
            std::cout << point[x][y];
        }

        std::cout << "\n";
    }
    return 0;
}

int Board::calculate() {
    int ans = 0;
    return ans;
}

Board::~Board() {

}