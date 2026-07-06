cmake_minimum_required(VERSION 3.16)
project(go-board VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GO_BOARD_PYTHON "Build the goboard Python module" OFF)

add_library(go_board_core STATIC src/board.cpp)
target_include_directories(go_board_core PUBLIC include)
set_target_properties(go_board_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(go-board src/main.cpp)
target_link_libraries(go-board PRIVATE go_board_core)

enable_testing()
add_executable(board_tests tests/test_board.cpp)
target_link_libraries(board_tests PRIVATE go_board_core)
add_test(NAME board_tests COMMAND board_tests)

if(GO_BOARD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(goboard python/bindings.cpp)
    target_link_libraries(goboard PRIVATE go_board_core)

    add_test(NAME python_bindings
             COMMAND ${Python_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/python/test_goboard.py)
    set_tests_properties(python_bindings PROPERTIES
        ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:goboard>")
endif()
