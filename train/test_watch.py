from train.watch import braille_chart, downsample, parse_losses


def test_parse_losses(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "resumed from x at iteration 5\n"
        "iter 5 | buffer 100 | B wins 3/6 | avg moves 90.0 | "
        "p-loss 3.1000 | v-loss 0.2500 | selfplay 10.0s train 5.0s\n"
        "iter 6 | buffer 200 | B wins 4/6 | avg moves 91.0 | "
        "p-loss 3.0000 | v-loss 0.2000 | selfplay 10.0s train 5.0s\n")
    p_losses, v_losses = parse_losses(log)
    assert p_losses == [3.1, 3.0]
    assert v_losses == [0.25, 0.2]


def test_downsample_preserves_short_series():
    assert downsample([1.0, 2.0], 10) == [1.0, 2.0]
    buckets = downsample(list(range(100)), 10)
    assert len(buckets) == 10
    assert buckets[0] < buckets[-1]


def test_braille_chart_shape_and_bounds():
    values = [3.0, 2.5, 2.0, 2.2, 1.8, 1.5]
    rows, lo, hi = braille_chart(values, cells_w=8, cells_h=3)
    assert len(rows) == 3
    assert all(len(row) == 8 for row in rows)
    assert (lo, hi) == (1.5, 3.0)
    # Something was actually drawn (not all blank braille).
    assert any(ch != "⠀" for row in rows for ch in row)


def test_braille_chart_flat_series():
    rows, lo, hi = braille_chart([1.0, 1.0, 1.0], cells_w=4, cells_h=2)
    assert hi > lo  # epsilon keeps the scale finite
    assert any(ch != "⠀" for row in rows for ch in row)
