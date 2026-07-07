import numpy as np

from goboard import Board, Stone

from train.mcts import (MCTS, Node, apply_move, terminal_value,
                        uniform_evaluate)


def make_mcts(seed: int = 0) -> MCTS:
    return MCTS(uniform_evaluate, rng=np.random.default_rng(seed))


def test_run_visits_and_policy():
    board = Board(size=5)
    mcts = make_mcts()
    root = mcts.run(board, Stone.BLACK, simulations=50)
    assert root.visits == 50

    pi = mcts.policy(root, board_size=5)
    assert pi.shape == (26,)
    assert abs(pi.sum() - 1.0) < 1e-6

    # Probability mass only on legal moves (whole board + pass here).
    legal = set(board.legal_moves(Stone.BLACK)) | {25}
    assert {i for i in range(26) if pi[i] > 0} <= legal


def test_illegal_moves_get_no_visits():
    board = Board(size=5)
    board.play(1, 0, Stone.WHITE)
    board.play(0, 1, Stone.WHITE)
    mcts = make_mcts()
    root = mcts.run(board, Stone.BLACK, simulations=60)
    pi = mcts.policy(root, board_size=5)
    assert pi[0] == 0.0  # (0,0) is suicide for black
    assert pi[1] == 0.0  # occupied
    assert pi[5] == 0.0  # occupied


def test_select_move_temperature_zero_is_argmax():
    board = Board(size=5)
    mcts = make_mcts()
    root = mcts.run(board, Stone.BLACK, simulations=40)
    pi = mcts.policy(root, board_size=5)
    assert mcts.select_move(root, board_size=5) == int(pi.argmax())


def test_search_does_not_mutate_board():
    board = Board(size=5)
    board.play(2, 2, Stone.BLACK)
    before = board.hash()
    make_mcts().run(board, Stone.WHITE, simulations=30)
    assert board.hash() == before
    assert not board.is_terminal()


def test_terminal_value_perspective():
    board = Board(size=5, komi=7.5)
    board.pass_()
    board.pass_()
    assert board.is_terminal()
    # Empty board: white wins by komi.
    assert terminal_value(board, Stone.BLACK) == -1.0
    assert terminal_value(board, Stone.WHITE) == 1.0


def test_virtual_loss_correction_equals_plain_backprop():
    mcts = make_mcts()
    virtual = [Node(0.0), Node(0.0), Node(0.0)]
    mcts.backprop(virtual, -1.0)                      # virtual loss
    mcts.backprop(virtual, 0.5 - (-1.0), visit_delta=0)  # real value

    plain = [Node(0.0), Node(0.0), Node(0.0)]
    mcts.backprop(plain, 0.5)

    for corrected, expected in zip(virtual, plain):
        assert corrected.visits == expected.visits
        assert abs(corrected.value_sum - expected.value_sum) < 1e-9


def test_apply_move_pass_and_play():
    board = Board(size=5)
    apply_move(board, 7, Stone.BLACK)  # x=2, y=1
    assert board.at(2, 1) == Stone.BLACK
    apply_move(board, 25, Stone.WHITE)  # pass
    board.pass_()
    assert board.is_terminal()
