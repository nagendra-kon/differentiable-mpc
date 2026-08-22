"""Unit/integration tests for the batched iLQR trajectory optimizer."""

import pytest
import torch

from src.constraints.cbf import ObstacleAvoidanceCBF
from src.dynamics.cartpole import CartpoleDynamics
from src.optimizers.ilqr import ILQR, QuadraticCost


def _make_stabilization_cost() -> QuadraticCost:
    Q = torch.diag(torch.tensor([10.0, 1.0, 50.0, 1.0]))
    R = torch.diag(torch.tensor([0.01]))
    Q_terminal = torch.diag(torch.tensor([50.0, 5.0, 200.0, 5.0]))
    return QuadraticCost(Q=Q, R=R, Q_terminal=Q_terminal, target_state=torch.zeros(4))


def test_ilqr_stabilizes_cartpole_from_offset_state() -> None:
    dynamics = CartpoleDynamics()
    cost = _make_stabilization_cost()
    initial_state = torch.tensor([[0.0, 0.0, 0.2, 0.0]])  # ~11.5 degree pole offset

    solver = ILQR(
        dynamics,
        cost.running_cost,
        cost.terminal_cost,
        horizon=30,
        dt=0.05,
        max_iterations=50,
    )
    result = solver.optimize(initial_state)

    # Cost must decrease substantially from the (uncontrolled) initial rollout.
    initial_cost = result.cost_history[0]
    final_cost = result.cost_history[-1]
    assert torch.all(final_cost < initial_cost * 0.1)

    # Cost is monotonically non-increasing: every accepted step only improves.
    for prev, nxt in zip(result.cost_history[:-1], result.cost_history[1:]):
        assert torch.all(nxt <= prev + 1e-9)

    assert result.converged

    # The final state should be close to the upright, centered target.
    final_state = result.states[:, -1, :]
    assert torch.all(final_state[:, 0].abs() < 0.15)  # cart near center
    assert torch.all(final_state[:, 2].abs() < 0.05)  # pole near upright


def test_ilqr_batched_solves_independently() -> None:
    dynamics = CartpoleDynamics()
    cost = _make_stabilization_cost()
    initial_state = torch.tensor(
        [
            [0.0, 0.0, 0.15, 0.0],
            [0.0, 0.0, -0.15, 0.0],
        ]
    )

    solver = ILQR(
        dynamics,
        cost.running_cost,
        cost.terminal_cost,
        horizon=20,
        dt=0.05,
        max_iterations=30,
    )
    result = solver.optimize(initial_state)

    assert result.states.shape == (2, 21, 4)
    assert result.controls.shape == (2, 20, 1)

    # Symmetric initial offsets should produce (roughly) mirrored trajectories.
    assert torch.allclose(result.states[0, :, 2], -result.states[1, :, 2], atol=1e-3)

    # Solving each batch element alone must match solving them together
    # (batched iLQR uses a shared line-search/regularization schedule, so
    # this checks the vectorized backward/forward pass math is per-sample
    # correct, not just that shapes line up).
    solver_single = ILQR(
        dynamics,
        cost.running_cost,
        cost.terminal_cost,
        horizon=20,
        dt=0.05,
        max_iterations=30,
    )
    result_single = solver_single.optimize(initial_state[0:1])
    assert torch.allclose(result.states[0:1], result_single.states, atol=1e-4)
    assert torch.allclose(result.controls[0:1], result_single.controls, atol=1e-4)


def test_ilqr_result_iterations_and_convergence_flag() -> None:
    dynamics = CartpoleDynamics()
    cost = _make_stabilization_cost()
    initial_state = torch.tensor([[0.0, 0.0, 0.1, 0.0]])

    solver = ILQR(
        dynamics,
        cost.running_cost,
        cost.terminal_cost,
        horizon=15,
        dt=0.05,
        max_iterations=40,
    )
    result = solver.optimize(initial_state)

    assert result.iterations > 0
    assert result.iterations <= 40
    assert isinstance(result.converged, bool)


def test_ilqr_with_cbf_barrier_keeps_trajectory_farther_from_obstacle() -> None:
    dynamics = CartpoleDynamics()
    # Target coincides with the obstacle location so the unconstrained
    # optimum drives straight through it; a barrier should push back.
    cost = QuadraticCost(
        Q=torch.diag(torch.tensor([10.0, 1.0, 5.0, 1.0])),
        R=torch.diag(torch.tensor([0.001])),
        target_state=torch.zeros(4),
    )
    obstacle = ObstacleAvoidanceCBF(torch.tensor([[0.0]]), safe_radius=0.3, position_indices=(0,))
    initial_state = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])

    solver_no_barrier = ILQR(
        dynamics, cost.running_cost, cost.terminal_cost, horizon=40, dt=0.05, max_iterations=50
    )
    result_no_barrier = solver_no_barrier.optimize(initial_state)

    solver_with_barrier = ILQR(
        dynamics,
        cost.running_cost,
        cost.terminal_cost,
        horizon=40,
        dt=0.05,
        max_iterations=50,
        barrier=obstacle,
        barrier_weight=500.0,
    )
    result_with_barrier = solver_with_barrier.optimize(initial_state)

    final_x_no_barrier = result_no_barrier.states[:, -1, 0].abs()
    final_x_with_barrier = result_with_barrier.states[:, -1, 0].abs()

    assert torch.all(final_x_with_barrier > final_x_no_barrier)


def test_ilqr_building_blocks_remain_differentiable_for_optimized_controls() -> None:
    """The solve itself detaches between iterations, but the returned
    controls can be replayed through a differentiable rollout — verifying
    the underlying dynamics/cost building blocks retain full autograd
    compatibility, per the "autograd flows through rollouts" requirement.
    """
    dynamics = CartpoleDynamics()
    cost = _make_stabilization_cost()
    initial_state = torch.tensor([[0.0, 0.0, 0.1, 0.0]])

    solver = ILQR(dynamics, cost.running_cost, cost.terminal_cost, horizon=15, dt=0.05, max_iterations=20)
    result = solver.optimize(initial_state)

    diff_initial_state = initial_state.clone().requires_grad_(True)
    diff_controls = result.controls.clone().requires_grad_(True)

    replayed_states = dynamics.rollout(diff_initial_state, diff_controls, dt=0.05)
    loss = replayed_states.pow(2).sum()
    loss.backward()

    assert diff_initial_state.grad is not None
    assert diff_controls.grad is not None
    assert torch.any(diff_controls.grad != 0)


def test_ilqr_handles_degenerate_zero_cost_without_crashing() -> None:
    # Sanity check that a degenerate (zero) cost still produces a well-formed,
    # non-crashing result rather than a singular Q_uu blowing up the solver.
    dynamics = CartpoleDynamics()

    def zero_running_cost(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)

    def zero_terminal_cost(state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)

    initial_state = torch.tensor([[0.0, 0.0, 0.1, 0.0]])
    solver = ILQR(dynamics, zero_running_cost, zero_terminal_cost, horizon=5, dt=0.05, max_iterations=5)
    result = solver.optimize(initial_state)

    assert result.states.shape == (1, 6, 4)
    assert result.controls.shape == (1, 5, 1)
