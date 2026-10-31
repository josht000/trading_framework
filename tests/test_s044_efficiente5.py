import numpy as np
import pytest

from src.strategies.s044_efficiente5.optimizer import (
    ASSET_CAPS, CLASS_GROUPS, build_weight_grid, solve_from_grid, solve_integer_portfolio,
)


def test_weight_grid_enforces_integer_units_and_caps():
    grid = build_weight_grid(
        asset_caps=np.array([2, 2], dtype=np.uint8),
        class_groups=(((0, 1), 2),),
        total_units=2,
    )
    assert set(map(tuple, grid.tolist())) == {(0, 2), (1, 1), (2, 0)}
    assert np.all(grid.sum(1) == 2)
    assert np.all(grid <= np.array([2, 2], dtype=np.uint8))


def test_exact_solver_selects_best_return_that_meets_risk_cap():
    grid = np.array([[20, 0], [10, 10], [0, 20]], dtype=np.uint8)
    mean = np.array([0.01, 0.005])
    covariance = np.diag([0.0004, 0.00000004])
    weights, target, expected_return, realized, fallback = solve_from_grid(
        grid, mean, covariance, target_vol=0.05, max_target_vol=0.50
    )
    assert weights.tolist() == pytest.approx([0.0, 1.0])
    assert target == pytest.approx(0.05)
    assert expected_return == pytest.approx(0.005)
    assert realized <= 0.05
    assert not fallback


def test_milp_outer_approximation_respects_efficiente_constraints():
    mean = np.linspace(0.0001, 0.0013, len(ASSET_CAPS))
    covariance = np.eye(len(ASSET_CAPS)) * 1e-8
    weights, target, _, realized, fallback, iterations = solve_integer_portfolio(
        mean, covariance, target_vol=0.05, max_target_vol=0.05
    )
    units = np.rint(weights / 0.05).astype(int)
    assert units.sum() == 20
    assert np.all(units <= ASSET_CAPS)
    for columns, cap in CLASS_GROUPS[:-1]:
        assert units[list(columns)].sum() <= cap
    assert target == pytest.approx(0.05)
    assert realized <= target
    assert not fallback
    assert iterations >= 1
