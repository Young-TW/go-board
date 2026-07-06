cmake_minimum_required(VERSION 3.16)
project(go-board VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(go_board_core STATIC src/board.cpp)
target_include_directories(go_board_core PUBLIC include)

add_executable(go-board src/main.cpp)
target_link_libraries(go-board PRIVATE go_board_core)

enable_testing()
add_executable(board_tests tests/test_board.cpp)
target_link_libraries(board_tests PRIVATE go_board_core)
add_test(NAME board_tests COMMAND board_tests)
