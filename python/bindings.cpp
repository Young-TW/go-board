#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>

#include "board.h"

namespace py = pybind11;

PYBIND11_MODULE(goboard, m) {
    m.doc() = "Go rules engine (go-board)";

    py::enum_<Stone>(m, "Stone")
        .value("EMPTY", Stone::Empty)
        .value("BLACK", Stone::Black)
        .value("WHITE", Stone::White);

    m.def("opponent", &opponent, py::arg("color"));

    py::class_<Board>(m, "Board")
        .def(py::init<int, float>(), py::arg("size") = 19,
             py::arg("komi") = 7.5f)
        .def_property_readonly("size", &Board::size)
        .def("at", &Board::at, py::arg("x"), py::arg("y"))
        .def("play", &Board::play, py::arg("x"), py::arg("y"),
             py::arg("color"),
             "Place a stone and resolve captures. Returns False and leaves "
             "the board unchanged if the move is illegal.")
        .def("is_legal", &Board::is_legal, py::arg("x"), py::arg("y"),
             py::arg("color"))
        .def("legal_moves", &Board::legal_moves, py::arg("color"),
             "Flattened indices (y * size + x) of all legal moves.")
        // 'pass' is a Python keyword, hence the trailing underscore.
        .def("pass_", &Board::pass)
        .def("is_terminal", &Board::is_terminal)
        .def("score", &Board::score)
        .def("hash", &Board::hash)
        .def(
            "features",
            [](const Board& board, Stone to_play) {
                const int n = board.size();
                const std::vector<float> data = board.features(to_play);
                py::array_t<float> planes({3, n, n});
                std::copy(data.begin(), data.end(), planes.mutable_data());
                return planes;
            },
            py::arg("to_play"),
            "float32 array of shape (3, size, size): own stones, opponent "
            "stones, colour-to-play indicator.")
        .def("__str__", &Board::to_string);
}
