"""Exact 5%-grid optimizer for the catalog's constrained Efficiente 5 portfolio."""

from itertools import product
from math import prod
from pathlib import Path

import numpy as np
from numba import njit
from ortools.sat.python import cp_model

ASSET_CAPS = np.array([4, 2, 4, 4, 4, 4, 4, 4, 4, 2, 2, 10, 10], dtype=np.uint8)
CLASS_GROUPS = (
    ((0, 1, 2), 10),
    ((3, 4, 5), 10),
    ((6, 7), 5),
    ((8, 9, 10), 5),
    ((11, 12), 20),
)
TOTAL_UNITS = 20
GRID_CACHE = Path("data/cache/s044_efficiente5_weight_grid_v1.npy")


def _patterns(caps, total_cap):
    by_sum = {}
    for values in product(*(range(int(cap) + 1) for cap in caps)):
        total = sum(values)
        if total <= total_cap:
            by_sum.setdefault(total, []).append(values)
    return {total: np.asarray(values, dtype=np.uint8) for total, values in by_sum.items()}


def build_weight_grid(asset_caps=ASSET_CAPS, class_groups=CLASS_GROUPS, total_units=TOTAL_UNITS):
    """Generate every feasible integer allocation with exactly `total_units` 5% units."""
    groups = []
    for columns, total_cap in class_groups:
        groups.append((tuple(columns), _patterns(asset_caps[list(columns)], total_cap)))
    sum_ranges = [tuple(sorted(patterns)) for _, patterns in groups]
    count = 0
    for first_four in product(*sum_ranges[:-1]):
        final_sum = int(total_units) - sum(first_four)
        if final_sum in groups[-1][1]:
            count += prod(len(groups[i][1][total]) for i, total in enumerate(first_four)) * len(groups[-1][1][final_sum])
    grid = np.empty((count, len(asset_caps)), dtype=np.uint8)
    offset = 0
    chunk_size = 500_000
    for first_four in product(*sum_ranges[:-1]):
        final_sum = int(total_units) - sum(first_four)
        if final_sum not in groups[-1][1]:
            continue
        totals = tuple(first_four) + (final_sum,)
        choices = [groups[i][1][total] for i, total in enumerate(totals)]
        sizes = [len(choice) for choice in choices]
        block_size = prod(sizes)
        for start in range(0, block_size, chunk_size):
            stop = min(block_size, start + chunk_size)
            idx = np.arange(start, stop, dtype=np.int64)
            for i, (columns, _) in enumerate(groups):
                stride = prod(sizes[i + 1:])
                choice_idx = (idx // stride) % sizes[i] if stride else idx % sizes[i]
                grid[offset + start:offset + stop, list(columns)] = choices[i][choice_idx]
        offset += block_size
    if offset != count:
        raise AssertionError(f"weight-grid size mismatch: expected {count}, wrote {offset}")
    return grid


def load_weight_grid(path=GRID_CACHE, rebuild=False):
    path = Path(path)
    if path.is_file() and not rebuild:
        grid = np.load(path, mmap_mode="r")
        if grid.ndim == 2 and grid.shape[1] == len(ASSET_CAPS) and np.issubdtype(grid.dtype, np.integer):
            return grid
    grid = build_weight_grid()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, grid)
    return np.load(path, mmap_mode="r")


@njit(cache=True)
def _first_feasible(grid, order, covariance, variance_limit):
    n_assets = grid.shape[1]
    for k in range(len(order)):
        row = order[k]
        variance = 0.0
        for i in range(n_assets):
            wi = grid[row, i] * 0.05
            for j in range(n_assets):
                variance += wi * (grid[row, j] * 0.05) * covariance[i, j]
        variance *= 1.0
        if variance <= variance_limit + 1e-12:
            return row
    return -1


@njit(cache=True)
def _minimum_variance(grid, covariance):
    best_row = -1
    best_var = np.inf
    n_assets = grid.shape[1]
    for row in range(len(grid)):
        variance = 0.0
        for i in range(n_assets):
            wi = grid[row, i] * 0.05
            for j in range(n_assets):
                variance += wi * (grid[row, j] * 0.05) * covariance[i, j]
        if variance < best_var:
            best_var = variance
            best_row = row
    return best_row, best_var


def solve_from_grid(grid, mean_returns, covariance, target_vol=0.05, max_target_vol=0.50):
    """Return the exact highest-mean feasible grid point, escalating target by 1%."""
    mu = np.asarray(mean_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    scores = np.zeros(len(grid), dtype=float)
    for j in range(grid.shape[1]):
        scores += np.asarray(grid[:, j], dtype=float) * mu[j]
    target = float(target_vol)
    while target <= max_target_vol + 1e-12:
        candidates = min(50_000, len(grid))
        while True:
            if candidates == len(grid):
                order = np.argsort(scores, kind="stable")[::-1]
            else:
                best = np.argpartition(scores, -candidates)[-candidates:]
                order = best[np.argsort(scores[best], kind="stable")[::-1]]
            best_row = _first_feasible(grid, order.astype(np.int64), cov, (target / np.sqrt(252)) ** 2)
            if best_row >= 0:
                weights = np.asarray(grid[best_row], dtype=float) * 0.05
                realized = float(np.sqrt(max(weights @ cov @ weights, 0.0) * 252))
                return weights, target, float(weights @ mu), realized, False
            if candidates == len(grid):
                break
            candidates = min(len(grid), candidates * 4)
        target = round(target + 0.01, 10)

    best_row, best_var = _minimum_variance(grid, cov)
    weights = np.asarray(grid[best_row], dtype=float) * 0.05
    realized = float(np.sqrt(max(best_var, 0.0) * 252))
    return weights, target, float(weights @ mu), realized, True


def _integer_model(covariance, mean_returns=None, target_variance=None):
    model = cp_model.CpModel()
    q = [model.NewIntVar(0, int(cap), f"q_{i}") for i, cap in enumerate(ASSET_CAPS)]
    model.Add(sum(q) == TOTAL_UNITS)
    for columns, class_cap in CLASS_GROUPS[:-1]:
        model.Add(sum(q[i] for i in columns) <= int(class_cap))

    covariance = (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T) / 2.0
    scale = 10**15
    risk_terms = []
    for i in range(len(q)):
        square = model.NewIntVar(0, int(ASSET_CAPS[i] ** 2), f"q2_{i}")
        model.AddMultiplicationEquality(square, [q[i], q[i]])
        diagonal = int(np.ceil(covariance[i, i] * scale))
        if diagonal:
            risk_terms.append(diagonal * square)
        for j in range(i + 1, len(q)):
            product_var = model.NewIntVar(0, int(ASSET_CAPS[i] * ASSET_CAPS[j]), f"qprod_{i}_{j}")
            model.AddMultiplicationEquality(product_var, [q[i], q[j]])
            coefficient = int(np.ceil(2.0 * covariance[i, j] * scale))
            if coefficient:
                risk_terms.append(coefficient * product_var)
    risk_expression = sum(risk_terms)
    if target_variance is not None:
        rhs = int(np.floor(float(target_variance) / (0.05 ** 2) * scale))
        model.Add(risk_expression <= rhs)
    if mean_returns is not None:
        objective_scale = 10**15
        coefficients = np.rint(np.asarray(mean_returns, dtype=float) * objective_scale).astype(np.int64)
        model.Maximize(sum(int(coefficients[i]) * q[i] for i in range(len(q))))
    else:
        model.Minimize(risk_expression)
    return model, q


def _cp_sat_solve(mean_returns, covariance, target_variance):
    model, q = _integer_model(covariance, mean_returns, target_variance)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        return None, solver
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"CP-SAT did not prove an optimal portfolio: {solver.StatusName(status)}")
    return np.asarray([solver.Value(value) for value in q], dtype=int), solver


def solve_integer_portfolio(mean_returns, covariance, *, target_vol=0.05, max_target_vol=0.50, grid_path=GRID_CACHE,
                            rebuild_grid=False):
    mu = np.asarray(mean_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    n_assets = len(ASSET_CAPS)
    if mu.shape != (n_assets,) or cov.shape != (n_assets, n_assets):
        raise ValueError("Efficiente 5 optimizer requires 13 returns and a 13x13 covariance matrix")
    solver_calls = 0
    target = float(target_vol)
    while target <= float(max_target_vol) + 1e-12:
        variance_limit = (target / np.sqrt(252)) ** 2
        units, solver = _cp_sat_solve(mu, cov, variance_limit)
        solver_calls += 1
        if units is not None:
            weights = units.astype(float) * 0.05
            variance = float(weights @ cov @ weights)
            if variance > variance_limit + 1e-12:
                raise RuntimeError("CP-SAT risk constraint failed floating-point verification")
            realized = float(np.sqrt(max(variance, 0.0) * 252))
            return weights, target, float(weights @ mu), realized, False, solver_calls
        target = round(target + 0.01, 10)

    model, q = _integer_model(cov)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    solver_calls += 1
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f"CP-SAT minimum-variance fallback failed: {solver.StatusName(status)}")
    weights = np.asarray([solver.Value(value) for value in q], dtype=float) * 0.05
    best_var = float(weights @ cov @ weights)
    realized = float(np.sqrt(max(best_var, 0.0) * 252))
    return weights, target, float(weights @ mu), realized, True, solver_calls
