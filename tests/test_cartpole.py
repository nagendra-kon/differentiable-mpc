"""Unit tests for the batch-capable, differentiable cartpole dynamics."""

import pytest
import torch

from src.dynamics.cartpole import CartpoleDynamics


@pytest.fixture
def dynamics() -> CartpoleDynamics:
    return CartpoleDynamics()


def test_continuous_dynamics_output_shape(dynamics: CartpoleDynamics) -> None:
    batch_size = 8
    state = torch.zeros(batch_size, 4, dtype=torch.float64)
    control = torch.zeros(batch_size, 1, dtype=torch.float64)

    deriv = dynamics.continuous_dynamics(state, control)

    assert deriv.shape == (batch_size, 4)


def test_upright_equilibrium_has_zero_derivative(dynamics: CartpoleDynamics) -> None:
    state = torch.zeros(4, 4, dtype=torch.float64)
    control = torch.zeros(4, 1, dtype=torch.float64)

    deriv = dynamics.continuous_dynamics(state, control)

    assert torch.allclose(deriv, torch.zeros_like(deriv), atol=1e-10)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_step_output_shape(dynamics: CartpoleDynamics, method: str) -> None:
    batch_size = 5
    state = torch.randn(batch_size, 4, dtype=torch.float64) * 0.1
    control = torch.randn(batch_size, 1, dtype=torch.float64)

    next_state = dynamics.step(state, control, dt=0.02, method=method)

    assert next_state.shape == state.shape


def test_rk4_more_accurate_than_euler(dynamics: CartpoleDynamics) -> None:
    state = torch.tensor([[0.0, 0.0, 0.2, 0.0]], dtype=torch.float64)
    control = torch.tensor([[1.0]], dtype=torch.float64)
    dt_coarse = 0.1

    # Fine-grained RK4 sub-stepping as an accurate reference trajectory.
    n_fine = 1000
    reference = state.clone()
    for _ in range(n_fine):
        reference = dynamics.step(reference, control, dt=dt_coarse / n_fine, method="rk4")

    euler_result = dynamics.step(state, control, dt=dt_coarse, method="euler")
    rk4_result = dynamics.step(state, control, dt=dt_coarse, method="rk4")

    euler_error = torch.norm(euler_result - reference)
    rk4_error = torch.norm(rk4_result - reference)

    assert rk4_error < euler_error


def test_rollout_shape_and_batch_independence(dynamics: CartpoleDynamics) -> None:
    batch_size, horizon = 4, 10
    initial_state = torch.zeros(batch_size, 4, dtype=torch.float64)
    initial_state[:, 2] = torch.tensor([0.05, 0.1, -0.05, -0.1], dtype=torch.float64)
    controls = torch.zeros(batch_size, horizon, 1, dtype=torch.float64)

    trajectory = dynamics.rollout(initial_state, controls, dt=0.02)

    assert trajectory.shape == (batch_size, horizon + 1, 4)
    assert torch.equal(trajectory[:, 0, :], initial_state)

    single = dynamics.rollout(initial_state[0:1], controls[0:1], dt=0.02)
    assert torch.allclose(trajectory[0:1], single)


def test_gradients_flow_through_rollout(dynamics: CartpoleDynamics) -> None:
    batch_size, horizon = 3, 5
    initial_state = torch.zeros(batch_size, 4, dtype=torch.float64, requires_grad=True)
    controls = torch.randn(batch_size, horizon, 1, dtype=torch.float64, requires_grad=True)

    trajectory = dynamics.rollout(initial_state, controls, dt=0.02)
    loss = trajectory.pow(2).sum()
    loss.backward()

    assert initial_state.grad is not None
    assert controls.grad is not None
    assert torch.any(controls.grad != 0)


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_gradients_flow_through_step(dynamics: CartpoleDynamics, method: str) -> None:
    state = torch.tensor([[0.0, 0.0, 0.1, 0.0]], dtype=torch.float64, requires_grad=True)
    control = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)

    next_state = dynamics.step(state, control, dt=0.02, method=method)
    next_state.sum().backward()

    assert state.grad is not None
    assert control.grad is not None
    assert torch.any(control.grad != 0)


def test_invalid_state_shape_raises(dynamics: CartpoleDynamics) -> None:
    bad_state = torch.zeros(3, 5, dtype=torch.float64)
    control = torch.zeros(3, 1, dtype=torch.float64)

    with pytest.raises(ValueError):
        dynamics.continuous_dynamics(bad_state, control)


def test_invalid_control_shape_raises(dynamics: CartpoleDynamics) -> None:
    state = torch.zeros(3, 4, dtype=torch.float64)
    bad_control = torch.zeros(3, 2, dtype=torch.float64)

    with pytest.raises(ValueError):
        dynamics.continuous_dynamics(state, bad_control)
