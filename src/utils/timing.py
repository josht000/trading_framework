"""Lightweight stage timing with an end-of-run summary table.

    from src.utils import timing
    timing.install("main")                 # once per entry point: prints summary + writes JSON at exit
    with timing.stage("load data"): ...    # nested stages build a tree; "<untracked>" shows unattributed time
    @timing.timed("name")                  # decorator form
    with timing.wstage("fill"): ...        # inside process-pool workers (flat, summed CPU time)

Workers return `drain_worker()` with their result; the parent calls `merge_worker(d)`.
Set PROFILE=1 to also write a cProfile dump (parent process only) to results/timing/.
"""

import atexit
import cProfile
import functools
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime

_T0 = time.perf_counter()
_NAME = "run"
_STACK = []
_TREE = {}      # path tuple -> [wall, cpu, calls]
_W = {}         # worker-local flat: name -> [wall, calls]
_WTOTAL = {}    # merged worker totals: name -> [wall, calls]
_installed = False
_prof = None


@contextmanager
def stage(name):
    _STACK.append(name)
    key = tuple(_STACK)
    w0, c0 = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        rec = _TREE.setdefault(key, [0.0, 0.0, 0])
        rec[0] += time.perf_counter() - w0
        rec[1] += time.process_time() - c0
        rec[2] += 1
        _STACK.pop()


def timed(name=None):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            with stage(name or fn.__name__):
                return fn(*a, **k)
        return wrapper
    return deco


@contextmanager
def wstage(name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        rec = _W.setdefault(name, [0.0, 0])
        rec[0] += time.perf_counter() - t0
        rec[1] += 1


def drain_worker():
    out = {k: list(v) for k, v in _W.items()}
    _W.clear()
    return out


def merge_worker(d):
    for k, (wall, n) in (d or {}).items():
        rec = _WTOTAL.setdefault(k, [0.0, 0])
        rec[0] += wall
        rec[1] += n


def _fmt(s):
    return f"{s / 60:.1f}m" if s >= 600 else f"{s:.2f}s"


def summary(print_it=True):
    total = time.perf_counter() - _T0
    lines = [f"\n{'=' * 78}", f"TIMING SUMMARY: {_NAME}  (total wall {_fmt(total)}, {total:.1f}s)", "=" * 78,
             f"{'stage':<52}{'wall':>9}{'%run':>7}{'calls':>7}{'cpu':>8}"]

    def children(prefix):
        n = len(prefix)
        return sorted((k for k in _TREE if len(k) == n + 1 and k[:n] == prefix), key=lambda k: -_TREE[k][0])

    def walk(prefix):
        kids = children(prefix)
        for k in kids:
            wall, cpu, calls = _TREE[k]
            lines.append(f"{'  ' * (len(k) - 1) + k[-1]:<52}{_fmt(wall):>9}{100 * wall / total:>6.1f}%{calls:>7}{_fmt(cpu):>8}")
            walk(k)
            sub = sum(_TREE[c][0] for c in children(k))
            if children(k) and wall - sub > max(0.05 * wall, 0.2):
                lines.append(f"{'  ' * len(k) + '<untracked>':<52}{_fmt(wall - sub):>9}{100 * (wall - sub) / total:>6.1f}%")

    walk(())
    top = sum(_TREE[k][0] for k in children(()))
    if total - top > max(0.02 * total, 0.5):
        lines.append(f"{'<untracked>':<52}{_fmt(total - top):>9}{100 * (total - top) / total:>6.1f}%")
    if _WTOTAL:
        lines += ["-" * 78, "worker stages (summed over all workers: CPU-seconds, not wall)",
                  f"{'stage':<52}{'sum':>9}{'calls':>7}{'mean ms':>10}"]
        for k, (wall, n) in sorted(_WTOTAL.items(), key=lambda kv: -kv[1][0]):
            lines.append(f"{k:<52}{_fmt(wall):>9}{n:>7}{1000 * wall / max(n, 1):>10.2f}")
    lines.append("=" * 78)
    text = "\n".join(lines)
    if print_it:
        print(text, flush=True)
    return text


def _finish():
    global _prof
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join("results", "timing")
    os.makedirs(out, exist_ok=True)
    if _prof is not None:
        _prof.disable()
        _prof.dump_stats(os.path.join(out, f"{_NAME}_{stamp}.prof"))
        print(f"cProfile dump: {os.path.join(out, f'{_NAME}_{stamp}.prof')}  (view: snakeviz or python -m pstats)")
    summary()
    data = {"name": _NAME, "total_wall_s": time.perf_counter() - _T0,
            "stages": {" > ".join(k): {"wall_s": v[0], "cpu_s": v[1], "calls": v[2]} for k, v in _TREE.items()},
            "worker_stages": {k: {"sum_s": v[0], "calls": v[1]} for k, v in _WTOTAL.items()}}
    path = os.path.join(out, f"{_NAME}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print(f"timing json: {path}", flush=True)


def install(name):
    """Register the end-of-run summary (idempotent). Call once at the top of an entry point."""
    global _NAME, _installed, _prof, _T0
    if _installed:
        return
    _installed, _NAME, _T0 = True, name, time.perf_counter()
    if os.getenv("PROFILE") == "1":
        _prof = cProfile.Profile()
        _prof.enable()
    atexit.register(_finish)
