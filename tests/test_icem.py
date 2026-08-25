"""Unit/integration tests for the batched iCEM trajectory optimizer."""

import pytest
import torch

from src.constraints.cbf import ObstacleAvoidanceCBF
from src.dynamics.cartpole import CartpoleDynamics
from src.optimizers.icem import ICEM
from src.optimizers.ilqr import QuadraticCost


def _make_stabilization_cost() -> QuadraticCost:
    Q = torch.diag(torch.tensor([10.0, 1.0, 50.0, 1.0]))
    R = torch.diag(torch.tensor([0.01]))
    Q_terminal = torch.diag(torch.tensor([50.0, 5.0, 200.0, 5.0]))
    return QuadraticCost(Q=Q, R=R, Q_terminal=Q_terminal, target_state=torch.zeros(4))


def test_icem_stabilizes_cartpole_from_offset_state() -> None:
    torch.manual_seed(0)
    dynamics = CartpoleDynamics()
    cost = _make_stabilization_cost()
    initial_state = torch.tensor([[0.0, 0.0, 0.2, 0.0]])  # ~11.5 degree pole offset

    solver = ICEM(
        dynamics,
        cost.running_cost,
        cost.terminal_cost,
        horizon=20,
        dt=0.05,
        num_samples=800,
        num_elites=80,
        num_iterations=30,
        init_std=2.0,
    )
    result = solver.optimize(initial_state)

    # Compare against the uncontrolled (zero-action) rollout cost: a stable
    # baseline, unlike cost_history[0] which is the noisy best-of-population
    # cost from the first (widest-std) sampling iteration.
    zero_controls = torch.zeros(1, solver.horizon, dynamics.control_dim)
    baseline_states = solver._rollout(initial_state, zero_controls)
    baseline_cost = solver._trajectory_cost(baseline_states, zero_controls)

    final_cost = result.cost_history[-1]
    assert torch.all(final_cost < baseline_cost * 0.05)

    # iCEM is a zeroth-order sampler and, unlike a gradient-based method like
    # iLQR, does not reliably find the precise global optimum over a 20-step
    # action sequence with this sample budget — but it must still recover a
    # stable, near-upright, bounded solution.
    final_state = result.states[:, -1, :]
    assert torch.all(final_state[:, 0].abs() < 0.4)  # cart stays bounded, near center
    assert torch.all(final_state[:, 2].abs() < 0.1)  # pole near upright


def test_icem_best_cost_is_monotonically_non_increasing() -> None:
    torch.manual_seed(1)
    dynamics = CartpoleDynamics()
    cost = _make_stabilization_cost()
    initial_state = torch.tensor([[0.0, 0.0, 0.15, 0.0]])

    solver = ICEM(
        dynamics,
        cost.running_cost,
        cost.terminal_cost,
        horizon=20,
        dt=0.05,
        num_samples=150,
        num_elites=15,
        num_iterations=10,
    )
    result = solver.optimize(initial_state)

    for prev, nxt in zip(result.cost_history[:-1], result.cost_history[1:]):
        assert torch.all(nxt <= prev + 1e-9)


def test_icem_batched_shapes_and_independent_convergence() -> None:
    torch.manual_seed(2)
    dynamics = CartpoleDynamics()
    cost = _make_stabilization_cost()
    initial_state = torch.tensor(
        [
            [0.0, 0.0, 0.15, 0.0],
            [0.0, 0.0, -0.15, 0.0],
        ]
    )

    solver = ICEM(
        dynamics,
        cost.running_cost,
        cost.terminal_cost,
        horizon=20,
        dt=0.05,
        num_samples=150,
        num_elites=15,
        num_iterations=10,
    )
    result = solver.optimize(initial_state)

    assert result.states.shape == (2, 21, 4)
    assert result.controls.shape == (2, 20, 1)
    assert result.mean.shape == (2, 20, 1)
    assert result.std.shape == (2, 20, 1)

    # Both batch elements (symmetric initial offsets) should independently
    # find low-cost, near-upright solutions.
    final_theta = result.states[:, -1, 2]
    assert torch.all(final_theta.abs() < 0.15)


def test_icem_respects_control_bounds() -> None:
    torch.manual_seed(3)
    dynamics = CartpoleDynamics()
    cost = _make_stabilization_cost()
    initial_state = torch.tensor([[0.0, 0.0, 0.2, 0.0]])

    solver = ICEM(
        dynamics,
        cost.running_cost,
        cost.terminal_cost,
        horizon=20,
        dt=0.05,
        num_samples=150,
        num_elites=15,
        num_iterations=8,
        init_std=5.0,
        control_low=-3.0,
        control_high=3.0,
    )
    result = solver.optimize(initial_state)

    assert torch.all(result.controls >= -3.0 - 1e-6)
    assert torch.all(result.controls <= 3.0 + 1e-6)


def test_icem_with_cbf_barrier_keeps_trajectory_farther_from_obstacle() -> None:
    dynamics = CartpoleDynamics()
    cost = QuadraticCost(
        Q=torch.diag(torch.tensor([10.0, 1.0, 5.0, 1.0])),
        R=torch.diag(torch.tensor([0.001])),
        target_state=torch.zeros(4),
    )
    obstacle = ObstacleAvoidanceCBF(torch.tensor([[0.0]]), safe_radius=0.3, position_indices=(0,))
    initial_state = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])

    common_kwargs = dict(
        horizon=40,
        dt=0.05,
        num_samples=500,
        num_elites=50,
        num_iterations=25,
        init_std=2.0,
    )

    torch.manual_seed(4)
    solver_no_barrier = ICEM(dynamics, cost.running_cost, cost.terminal_cost, **common_kwargs)
    result_no_barrier = solver_no_barrier.optimize(initial_state)

    torch.manual_seed(4)
    solver_with_barrier = ICEM(
        dynamics,
        cost.running_cost,
        cost.terminal_cost,
        barrier=obstacle,
        barrier_weight=500.0,
        **common_kwargs,
    )
    result_with_barrier = solver_with_barrier.optimize(initial_state)

    final_x_no_barrier = result_no_barrier.states[:, -1, 0].abs()
    final_x_with_barrier = result_with_barrier.states[:, -1, 0].abs()

    assert torch.all(final_x_with_barrier > final_x_no_barrier)


def test_icem_rejects_invalid_elite_configuration() -> None:
    dynamics = CartpoleDynamics()
    cost = _make_stabilization_cost()

    with pytest.raises(ValueError):
        ICEM(dynamics, cost.running_cost, cost.terminal_cost, horizon=10, dt=0.05, num_samples=50, num_elites=50)

    with pytest.raises(ValueError):
        ICEM(dynamics, cost.running_cost, cost.terminal_cost, horizon=10, dt=0.05, num_samples=50, num_elites=0)
