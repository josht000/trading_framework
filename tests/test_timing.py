import time

from src.utils import timing


def test_stage_nesting_accumulates_calls_and_wall():
    parent = "timing test parent"
    child = "timing test child"
    with timing.stage(parent):
        time.sleep(0.001)
        with timing.stage(child):
            time.sleep(0.001)

    parent_rec = timing._TREE[(parent,)]
    child_rec = timing._TREE[(parent, child)]
    assert parent_rec[2] == 1
    assert child_rec[2] == 1
    assert parent_rec[0] >= child_rec[0] > 0


def test_worker_timing_round_trip():
    name = "timing test worker"
    with timing.wstage(name):
        time.sleep(0.001)
    drained = timing.drain_worker()
    assert drained[name][1] == 1
    assert drained[name][0] > 0
    assert timing.drain_worker() == {}

    timing.merge_worker(drained)
    assert timing._WTOTAL[name][1] == 1
    assert timing._WTOTAL[name][0] == drained[name][0]


def test_summary_contains_stage_names():
    text = timing.summary(print_it=False)
    assert "TIMING SUMMARY" in text
    assert "timing test parent" in text
    assert "timing test worker" in text
