#include <iostream>
#include <vector>

#include "board.h"

template <class T>
void print_vector(std::vector<T> vec) {
    for (int i = 0; i < vec.size(); i++) {
        std::cout << " " << vec[i];
    }
}

template <class T>
std::vector<T> bubble(std::vector<T> v) {
    int reg;
    for (int i = 0; i < v.size(); i++) {
        for (int j = v.size(); j >= 0; j--) {
            if (v[j] < v[j + 1]) {
                reg = v[j];
                v[j] = v[j + 1];
                v[j + 1] = reg;
            }
        }
    }

    return v;
}

enum go_point { black, white };

int main() {
    // const std::vector<int> v = {2, 1, 4, 3, 5, 9, 7, 6, 8, 0, 11, 19, 256,
    // 128, 1024};

    // print_vector(bubble(v));
    Board board;

    int x, y;
    while (board.gamestatus != 2) {
        std::cin >> x >> y;

        if (x == 99 || y == 99) {
            board.gamestatus == 1;
        }

        if (x == 100 || y == 100) {
            board.gamestatus = 2;
        }

        x--;
        y--;

        board.set(x, y, board.rotation % 2 + 1);
        board.print();
        board.rotation++;
    }

    board.~Board();
    return 0;
}