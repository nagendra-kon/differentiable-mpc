"""Batch-capable, differentiable analytical dynamics for the 2D cartpole system.

State convention: ``[x, x_dot, theta, theta_dot]`` where ``theta`` is the pole
angle from the upward vertical (0 = upright, unstable equilibrium), in radians.
Control convention: ``[force]``, the horizontal force applied to the cart.

The continuous-time equations of motion follow the standard frictionless
cartpole model (e.g. Florian, "Correct equations for the dynamics of the
cart-pole system", and OpenAI Gym's CartPole physics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

STATE_DIM = 4
CONTROL_DIM = 1

IntegrationMethod = Literal["euler", "rk4"]


@dataclass(frozen=True)
class CartpoleParams:
    """Physical parameters of the cartpole system."""

    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_half_length: float = 0.5
    gravity: float = 9.81


class CartpoleDynamics:
    """Differentiable, batch-capable cartpole dynamics.

    All methods are pure functions of their input tensors: the device and
    dtype of outputs follow the inputs, so callers control placement (e.g.
    move ``state``/``control`` to CUDA/MPS before calling).
    """

    state_dim: int = STATE_DIM
    control_dim: int = CONTROL_DIM

    def __init__(self, params: CartpoleParams | None = None) -> None:
        self.params = params or CartpoleParams()

    def continuous_dynamics(self, state: Tensor, control: Tensor) -> Tensor:
        """Compute the state derivative ``d(state)/dt`` for the cartpole ODE.

        Args:
            state: Tensor of shape ``[batch_size, 4]``: ``[x, x_dot, theta, theta_dot]``.
            control: Tensor of shape ``[batch_size, 1]``: ``[force]``.

        Returns:
            Tensor of shape ``[batch_size, 4]``, the time derivative of state.
        """
        self._check_shape(state, self.state_dim, "state")
        self._check_shape(control, self.control_dim, "control")

        m_c = self.params.cart_mass
        m_p = self.params.pole_mass
        length = self.params.pole_half_length
        g = self.params.gravity
        total_mass = m_c + m_p

        x_dot = state[:, 1]
        theta = state[:, 2]
        theta_dot = state[:, 3]
        force = control[:, 0]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        temp = (force + m_p * length * theta_dot**2 * sin_theta) / total_mass
        theta_ddot = (g * sin_theta - cos_theta * temp) / (
            length * (4.0 / 3.0 - m_p * cos_theta**2 / total_mass)
        )
        x_ddot = temp - (m_p * length * theta_ddot * cos_theta) / total_mass

        return torch.stack([x_dot, x_ddot, theta_dot, theta_ddot], dim=-1)

    def step(
        self,
        state: Tensor,
        control: Tensor,
        dt: float,
        method: IntegrationMethod = "rk4",
    ) -> Tensor:
        """Advance the cartpole state by one discrete timestep.

        Args:
            state: Tensor of shape ``[batch_size, 4]``.
            control: Tensor of shape ``[batch_size, 1]``, held constant over ``dt``.
            dt: Integration timestep in seconds.
            method: ``"rk4"`` (default, 4th-order Runge-Kutta) or ``"euler"``.

        Returns:
            Next state, tensor of shape ``[batch_size, 4]``.
        """
        if method == "euler":
            return state + dt * self.continuous_dynamics(state, control)
        if method == "rk4":
            k1 = self.continuous_dynamics(state, control)
            k2 = self.continuous_dynamics(state + 0.5 * dt * k1, control)
            k3 = self.continuous_dynamics(state + 0.5 * dt * k2, control)
            k4 = self.continuous_dynamics(state + dt * k3, control)
            return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        raise ValueError(f"Unknown integration method: {method!r}")

    def rollout(
        self,
        initial_state: Tensor,
        controls: Tensor,
        dt: float,
        method: IntegrationMethod = "rk4",
    ) -> Tensor:
        """Roll out a control sequence from an initial state.

        The horizon dimension is inherently sequential (each state depends on
        the previous one), so it is advanced with a Python loop over ``step``.

        Args:
            initial_state: Tensor of shape ``[batch_size, 4]``.
            controls: Tensor of shape ``[batch_size, horizon, 1]``.
            dt: Integration timestep in seconds.
            method: ``"rk4"`` (default) or ``"euler"``.

        Returns:
            Tensor of shape ``[batch_size, horizon + 1, 4]``: the initial
            state followed by the state after each control input.
        """
        self._check_shape(initial_state, self.state_dim, "initial_state")
        if controls.dim() != 3 or controls.shape[-1] != self.control_dim:
            raise ValueError(
                f"controls must have shape [batch, horizon, {self.control_dim}], "
                f"got {tuple(controls.shape)}"
            )

        horizon = controls.shape[1]
        states = [initial_state]
        state = initial_state
        for t in range(horizon):
            state = self.step(state, controls[:, t, :], dt, method=method)
            states.append(state)
        return torch.stack(states, dim=1)

    def _check_shape(self, tensor: Tensor, last_dim: int, name: str) -> None:
        if tensor.dim() != 2 or tensor.shape[-1] != last_dim:
            raise ValueError(f"{name} must have shape [batch, {last_dim}], got {tuple(tensor.shape)}")
