"""Unit tests for the differentiable Control Barrier Function (CBF) module."""

import pytest
import torch

from src.constraints.cbf import (
    BoxConstraintCBF,
    CombinedBarrier,
    InputConstraintCBF,
    ObstacleAvoidanceCBF,
    StateConstraintCBF,
    discrete_cbf_condition,
)


# ---------------------------------------------------------------------------
# ObstacleAvoidanceCBF
# ---------------------------------------------------------------------------


def test_obstacle_cbf_value_matches_analytic_squared_distance() -> None:
    obstacles = torch.tensor([[0.0]])
    cbf = ObstacleAvoidanceCBF(obstacles, safe_radius=1.0, position_indices=(0,))
    state = torch.tensor([[5.0, 0.0, 0.0, 0.0]])

    h = cbf.value(state)

    assert h.shape == (1, 1)
    assert torch.allclose(h, torch.tensor([[24.0]]))  # (5-0)^2 - 1^2


def test_obstacle_cbf_detects_safety_violation() -> None:
    obstacles = torch.tensor([[0.0]])
    cbf = ObstacleAvoidanceCBF(obstacles, safe_radius=1.0, position_indices=(0,))
    # batch: far away (safe), inside the obstacle radius (unsafe), exactly on boundary (safe)
    state = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    safe = cbf.is_safe(state)

    assert torch.equal(safe, torch.tensor([True, False, True]))


def test_obstacle_cbf_multiple_obstacles_shape_and_min() -> None:
    obstacles = torch.tensor([[0.0], [10.0]])
    cbf = ObstacleAvoidanceCBF(obstacles, safe_radius=1.0, position_indices=(0,))
    state = torch.tensor([[5.0, 0.0, 0.0, 0.0]])

    h = cbf.value(state)
    worst = cbf.min_value(state)

    assert h.shape == (1, 2)
    assert torch.allclose(worst, torch.min(h, dim=-1).values)


def test_obstacle_cbf_gradient_matches_analytic() -> None:
    obstacles = torch.tensor([[0.0]])
    cbf = ObstacleAvoidanceCBF(obstacles, safe_radius=1.0, position_indices=(0,))
    state = torch.tensor([[3.0, 0.0, 0.0, 0.0]], requires_grad=True)

    h = cbf.value(state)
    h.sum().backward()

    # d/dx [(x-0)^2 - 1] = 2x = 6 at x=3; zero for other state dims.
    assert torch.allclose(state.grad, torch.tensor([[6.0, 0.0, 0.0, 0.0]]))


def test_obstacle_cbf_rejects_mismatched_obstacle_shape() -> None:
    obstacles = torch.tensor([0.0, 1.0])  # wrong: 1D instead of [num_obstacles, 1]

    with pytest.raises(ValueError):
        ObstacleAvoidanceCBF(obstacles, safe_radius=1.0, position_indices=(0,))


def test_obstacle_cbf_rejects_nonpositive_radius() -> None:
    with pytest.raises(ValueError):
        ObstacleAvoidanceCBF(torch.tensor([[0.0]]), safe_radius=0.0)


# ---------------------------------------------------------------------------
# BoxConstraintCBF / StateConstraintCBF / InputConstraintCBF
# ---------------------------------------------------------------------------


def test_state_constraint_cbf_bounds_position_and_angle_only() -> None:
    lower = torch.tensor([-2.0, -float("inf"), -0.5, -float("inf")])
    upper = torch.tensor([2.0, float("inf"), 0.5, float("inf")])
    cbf = StateConstraintCBF(lower, upper)
    state = torch.tensor([[0.0, 100.0, 0.0, -100.0]])

    h = cbf.value(state)

    # 2 bounded dims (indices 0, 2) x 2 sides (lower, upper) = 4 columns.
    assert h.shape == (1, 4)


def test_state_constraint_cbf_detects_violation() -> None:
    lower = torch.tensor([-2.0, -float("inf"), -0.5, -float("inf")])
    upper = torch.tensor([2.0, float("inf"), 0.5, float("inf")])
    cbf = StateConstraintCBF(lower, upper)
    state = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],  # safe
            [3.0, 0.0, 0.0, 0.0],  # x exceeds upper bound
            [0.0, 0.0, -0.9, 0.0],  # theta exceeds lower bound
        ]
    )

    safe = cbf.is_safe(state)

    assert torch.equal(safe, torch.tensor([True, False, False]))


def test_input_constraint_cbf_force_bounds() -> None:
    lower = torch.tensor([-10.0])
    upper = torch.tensor([10.0])
    cbf = InputConstraintCBF(lower, upper)
    control = torch.tensor([[5.0], [15.0], [-10.0]])

    safe = cbf.is_safe(control)

    assert torch.equal(safe, torch.tensor([True, False, True]))


def test_box_constraint_cbf_lower_bound_gradient_is_analytic() -> None:
    lower = torch.tensor([-2.0])
    upper = torch.tensor([float("inf")])
    cbf = BoxConstraintCBF(lower, upper)
    x = torch.tensor([[1.0]], requires_grad=True)

    h = cbf.value(x)  # h = x - (-2) = x + 2
    h.sum().backward()

    assert torch.allclose(h, torch.tensor([[3.0]]))
    assert torch.allclose(x.grad, torch.tensor([[1.0]]))


def test_box_constraint_cbf_upper_bound_gradient_is_analytic() -> None:
    lower = torch.tensor([-float("inf")])
    upper = torch.tensor([2.0])
    cbf = BoxConstraintCBF(lower, upper)
    x = torch.tensor([[1.0]], requires_grad=True)

    h = cbf.value(x)  # h = 2 - x
    h.sum().backward()

    assert torch.allclose(h, torch.tensor([[1.0]]))
    assert torch.allclose(x.grad, torch.tensor([[-1.0]]))


def test_box_constraint_cbf_rejects_upper_less_than_lower() -> None:
    with pytest.raises(ValueError):
        BoxConstraintCBF(torch.tensor([5.0]), torch.tensor([1.0]))


def test_box_constraint_cbf_rejects_all_unbounded() -> None:
    with pytest.raises(ValueError):
        BoxConstraintCBF(torch.tensor([-float("inf")]), torch.tensor([float("inf")]))


def test_box_constraint_cbf_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        BoxConstraintCBF(torch.tensor([-1.0, -1.0]), torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# CombinedBarrier
# ---------------------------------------------------------------------------


def test_combined_barrier_concatenates_all_constraints() -> None:
    obstacle_cbf = ObstacleAvoidanceCBF(torch.tensor([[0.0]]), safe_radius=1.0, position_indices=(0,))
    state_cbf = StateConstraintCBF(
        torch.tensor([-2.0, -float("inf"), -0.5, -float("inf")]),
        torch.tensor([2.0, float("inf"), 0.5, float("inf")]),
    )
    combined = CombinedBarrier([obstacle_cbf, state_cbf])
    state = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    h = combined.value(state)

    assert h.shape == (1, 1 + 4)  # 1 obstacle column + 4 state-bound columns


def test_combined_barrier_gradient_flows_through_all_components() -> None:
    obstacle_cbf = ObstacleAvoidanceCBF(torch.tensor([[0.0]]), safe_radius=1.0, position_indices=(0,))
    # One-sided bound: a two-sided bound's gradient would cancel to zero when
    # summed (d/dx[(x-lower) + (upper-x)] = 0), which is correct but would
    # make this assertion trivially true/false regardless of grad wiring.
    input_cbf = InputConstraintCBF(torch.tensor([-10.0]), torch.tensor([float("inf")]))
    state = torch.tensor([[3.0, 0.0, 0.0, 0.0]], requires_grad=True)
    control = torch.tensor([[5.0]], requires_grad=True)

    combined_loss = obstacle_cbf.value(state).sum() + input_cbf.value(control).sum()
    combined_loss.backward()

    assert state.grad is not None
    assert control.grad is not None
    assert torch.any(state.grad != 0)
    assert torch.allclose(control.grad, torch.tensor([[1.0]]))


def test_combined_barrier_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        CombinedBarrier([])


# ---------------------------------------------------------------------------
# discrete_cbf_condition
# ---------------------------------------------------------------------------


def test_discrete_cbf_condition_formula() -> None:
    h_current = torch.tensor([2.0])
    h_next = torch.tensor([1.5])

    result = discrete_cbf_condition(h_current, h_next, alpha=0.5)

    assert torch.allclose(result, torch.tensor([0.5]))  # 1.5 - 0.5*2.0


def test_discrete_cbf_condition_alpha_one_requires_nonnegative_h() -> None:
    h_current = torch.tensor([2.0])
    h_next = torch.tensor([1.5])

    result = discrete_cbf_condition(h_current, h_next, alpha=1.0)

    assert torch.allclose(result, h_next)


def test_discrete_cbf_condition_rejects_invalid_alpha() -> None:
    h_current = torch.tensor([2.0])
    h_next = torch.tensor([1.5])

    with pytest.raises(ValueError):
        discrete_cbf_condition(h_current, h_next, alpha=0.0)

    with pytest.raises(ValueError):
        discrete_cbf_condition(h_current, h_next, alpha=1.5)


def test_discrete_cbf_condition_gradient_flows() -> None:
    h_current = torch.tensor([2.0], requires_grad=True)
    h_next = torch.tensor([1.5], requires_grad=True)

    result = discrete_cbf_condition(h_current, h_next, alpha=0.5)
    result.sum().backward()

    assert torch.allclose(h_current.grad, torch.tensor([-0.5]))
    assert torch.allclose(h_next.grad, torch.tensor([1.0]))
