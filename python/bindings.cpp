#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>

#include "board.h"
#include "selfplay_pool.h"

namespace py = pybind11;

PYBIND11_MODULE(goboard, m) {
    m.doc() = "Go rules engine (go-board)";

    py::enum_<Stone>(m, "Stone")
        .value("EMPTY", Stone::Empty)
        .value("BLACK", Stone::Black)
        .value("WHITE", Stone::White);

    m.def("opponent", &opponent, py::arg("color"));
    m.attr("FEATURE_PLANES") = Board::kFeaturePlanes;

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
        .def("copy", [](const Board& board) { return Board(board); },
             "Deep copy, including position history.")
        .def("score", &Board::score)
        .def("hash", &Board::hash)
        .def(
            "features",
            [](const Board& board, Stone to_play) {
                const int n = board.size();
                const std::vector<float> data = board.features(to_play);
                py::array_t<float> planes({Board::kFeaturePlanes, n, n});
                std::copy(data.begin(), data.end(), planes.mutable_data());
                return planes;
            },
            py::arg("to_play"),
            "float32 array of shape (FEATURE_PLANES, size, size); see "
            "board.h for the plane layout.")
        .def("__str__", &Board::to_string);

    py::class_<SelfPlayPool>(m, "SelfPlayPool",
                             "Batched C++ self-play driver; see "
                             "selfplay_pool.h for the collect/submit "
                             "protocol.")
        .def(py::init<int, int, float, int, int, float, int, int, int,
                      float, float, float, std::uint64_t>(),
             py::arg("n_games"), py::arg("board_size") = 9,
             py::arg("komi") = 7.5f, py::arg("simulations") = 128,
             py::arg("cheap_simulations") = 128,
             py::arg("full_search_prob") = 1.0f,
             py::arg("temperature_moves") = 8,
             py::arg("leaves_per_game") = 4,
             py::arg("parallel") = 0,  // 0 means n_games
             py::arg("c_puct") = 1.5f, py::arg("dirichlet_alpha") = 0.3f,
             py::arg("noise_fraction") = 0.25f, py::arg("seed") = 0)
        .def("done", &SelfPlayPool::done)
        .def(
            "collect",
            [](SelfPlayPool& pool) {
                int count;
                {
                    py::gil_scoped_release release;
                    count = pool.collect();
                }
                const int n = pool.board_size();
                py::array_t<float> planes(
                    {count, Board::kFeaturePlanes, n, n});
                std::copy(pool.features().begin(), pool.features().end(),
                          planes.mutable_data());
                return planes;
            },
            "Advance all games and return the feature batch to "
            "evaluate, shape (count, FEATURE_PLANES, size, size).")
        .def(
            "submit",
            [](SelfPlayPool& pool,
               py::array_t<float, py::array::c_style | py::array::forcecast>
                   priors,
               py::array_t<float, py::array::c_style | py::array::forcecast>
                   values) {
                const int count = int(priors.shape(0));
                if (count == 0) return;
                const float* prior_data = priors.data();
                const float* value_data = values.data();
                py::gil_scoped_release release;
                pool.submit(prior_data, value_data, count);
            },
            py::arg("priors"), py::arg("values"))
        .def("spectate",
             [](const SelfPlayPool& pool) {
                 return py::make_tuple(
                     pool.spectate_board(), pool.spectate_moves(),
                     pool.spectate_black_to_play(), pool.games_finished(),
                     pool.games_started());
             },
             "Live view of one running game: (board_text, move_count, "
             "black_to_play, games_finished, games_started).")
        .def("take_results", [](SelfPlayPool& pool) {
            const int n = pool.board_size();
            const int points = n * n;
            py::list out;
            for (GameResult& result : pool.take_results()) {
                const int moves = int(result.samples.size());
                py::array_t<float> features(
                    {moves, Board::kFeaturePlanes, n, n});
                py::array_t<float> pi({moves, points + 1});
                py::array_t<float> z(moves);
                py::array_t<float> train_pi(moves);
                for (int i = 0; i < moves; i++) {
                    const SampleRec& sample = result.samples[i];
                    std::copy(sample.features.begin(),
                              sample.features.end(),
                              features.mutable_data() +
                                  std::size_t(i) * sample.features.size());
                    std::copy(sample.pi.begin(), sample.pi.end(),
                              pi.mutable_data() +
                                  std::size_t(i) * sample.pi.size());
                    z.mutable_at(i) = sample.z;
                    train_pi.mutable_at(i) =
                        sample.train_policy ? 1.0f : 0.0f;
                }
                out.append(py::make_tuple(features, pi, z, train_pi,
                                          result.black_margin));
            }
            return out;
        });
}
