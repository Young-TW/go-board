#ifndef Board_H
#define Board_H

#include <vector>

class Board {
    public:
        Board();
        int set(int x, int y, int status);
        int print();
        int calculate();
        ~Board();

        int rotation = 0;
        int gamestatus = 0;

    private:
        std::vector<std::vector<int>> point;
};

#endif