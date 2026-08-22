"""Batched, differentiable iLQR (iterative LQR) trajectory optimizer.

Solves, for each batch element independently but vectorized together:

    min_{u_0..u_{T-1}}  sum_t running_cost(x_t, u_t) + terminal_cost(x_T)
    s.t.                x_{t+1} = dynamics.step(x_t, u_t, dt)

via the standard iLQR/DDP-style backward Riccati recursion (using only the
first-order dynamics expansion, i.e. iLQR rather than full second-order DDP)
followed by a forward line search. Local Jacobians/Hessians are obtained
from the user-supplied ``dynamics``/cost callables via PyTorch autograd
(double backward for cost Hessians), so any differentiable dynamics or cost
module (e.g. ``src.dynamics.cartpole.CartpoleDynamics``) can be plugged in
without hand-derived derivatives.

The optimizer detaches between outer iterations (standard iLQR practice —
each iteration solves a fresh local linearization, not an unrolled
computation graph spanning all iterations). Every individual building block
(dynamics step, cost evaluation, linearization) is nonetheless a pure
differentiable PyTorch operation: if gradients through the *found* optimal
trajectory are needed (e.g. w.r.t. the initial state or cost parameters),
re-run ``dynamics.rollout(initial_state, result.controls, dt)`` under
``requires_grad`` using the optimized controls as a warm start.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import torch
from torch import Tensor

from src.constraints.cbf import BarrierFunction

IntegrationMethod = Literal["euler", "rk4"]


class DynamicsModel(Protocol):
    """Structural interface an ILQR-compatible dynamics model must satisfy."""

    state_dim: int
    control_dim: int

    def step(self, state: Tensor, control: Tensor, dt: float, method: IntegrationMethod = ...) -> Tensor:
        """Advance ``state`` by one timestep under ``control``. See e.g. CartpoleDynamics.step."""
        ...


def _batch_jacobian(outputs: Tensor, inputs: Tensor) -> Tensor:
    """Batched Jacobian of ``outputs`` w.r.t. ``inputs`` via per-output-dim autograd.

    Assumes ``outputs`` is batch-elementwise in ``inputs`` (batch element b of
    the output depends only on batch element b of the input — true for every
    dynamics/cost module in this codebase, which operate row-wise on
    ``[batch, dim]`` tensors). Under that assumption, summing an output
    column over the batch before calling ``autograd.grad`` yields exactly
    the per-sample gradient for that column, with no cross-batch mixing.

    Args:
        outputs: Tensor of shape ``[batch, out_dim]``, connected to ``inputs``
            in the autograd graph.
        inputs: Tensor of shape ``[batch, in_dim]``.

    Returns:
        Tensor of shape ``[batch, out_dim, in_dim]``.
    """
    out_dim = outputs.shape[-1]
    rows = []
    for i in range(out_dim):
        (grad_i,) = torch.autograd.grad(
            outputs[:, i].sum(), inputs, retain_graph=True, create_graph=False, allow_unused=True
        )
        # allow_unused: `outputs` may be analytically independent of `inputs`
        # for some cost/dynamics functions (e.g. a separable cost has zero
        # cross-term Hessian l_ux); autograd reports that as "unused" rather
        # than a zero-valued gradient.
        rows.append(grad_i if grad_i is not None else torch.zeros_like(inputs))
    return torch.stack(rows, dim=1)


class QuadraticCost:
    """Time-invariant quadratic tracking cost: ``(x-x*)^T Q (x-x*) + u^T R u``."""

    def __init__(
        self,
        Q: Tensor,
        R: Tensor,
        Q_terminal: Tensor | None = None,
        target_state: Tensor | None = None,
    ) -> None:
        """
        Args:
            Q: Running state-cost weight matrix, shape ``[state_dim, state_dim]``.
            R: Running control-cost weight matrix, shape ``[control_dim, control_dim]``.
            Q_terminal: Terminal state-cost weight matrix; defaults to ``Q``.
            target_state: Target state, shape ``[state_dim]``; defaults to the origin.
        """
        self.Q = Q
        self.R = R
        self.Q_terminal = Q_terminal if Q_terminal is not None else Q
        self.target_state = target_state

    def running_cost(self, state: Tensor, control: Tensor) -> Tensor:
        dx = state - self._target(state)
        state_cost = torch.einsum("bi,ij,bj->b", dx, self.Q.to(dx.dtype), dx)
        control_cost = torch.einsum("bi,ij,bj->b", control, self.R.to(control.dtype), control)
        return state_cost + control_cost

    def terminal_cost(self, state: Tensor) -> Tensor:
        dx = state - self._target(state)
        return torch.einsum("bi,ij,bj->b", dx, self.Q_terminal.to(dx.dtype), dx)

    def _target(self, state: Tensor) -> Tensor:
        if self.target_state is None:
            return torch.zeros_like(state)
        return self.target_state.to(dtype=state.dtype, device=state.device)


@dataclass
class ILQRResult:
    """Output of :meth:`ILQR.optimize`."""

    states: Tensor
    """Optimized state trajectory, shape ``[batch, horizon + 1, state_dim]``."""

    controls: Tensor
    """Optimized control sequence, shape ``[batch, horizon, control_dim]``."""

    cost_history: list[Tensor] = field(default_factory=list)
    """Total trajectory cost after each accepted iteration, each ``[batch]``."""

    converged: bool = False
    """Whether the line-search-accepted cost improvement fell below the convergence threshold."""

    iterations: int = 0
    """Number of outer iLQR iterations performed."""


class ILQR:
    """Batched iterative LQR trajectory optimizer."""

    _MAX_REGULARIZATION_ATTEMPTS = 10
    _PD_EPS = 1e-9

    def __init__(
        self,
        dynamics: DynamicsModel,
        running_cost: Callable[[Tensor, Tensor], Tensor],
        terminal_cost: Callable[[Tensor], Tensor],
        horizon: int,
        dt: float,
        integration_method: IntegrationMethod = "rk4",
        barrier: BarrierFunction | None = None,
        barrier_weight: float = 0.0,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-6,
        initial_regularization: float = 1e-6,
        regularization_scaling: float = 10.0,
        max_regularization: float = 1e10,
        line_search_steps: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625),
    ) -> None:
        """
        Args:
            dynamics: Object exposing ``state_dim``, ``control_dim``, and
                ``step(state, control, dt, method)`` (e.g. ``CartpoleDynamics``).
            running_cost: ``(state, control) -> Tensor[batch]``.
            terminal_cost: ``(state) -> Tensor[batch]``.
            horizon: Number of control steps ``T``.
            dt: Integration timestep passed to ``dynamics.step``.
            integration_method: ``"rk4"`` or ``"euler"``, passed to ``dynamics.step``.
            barrier: Optional CBF (or combined barrier) added to the running
                cost as a squared-hinge penalty on constraint violation.
            barrier_weight: Penalty weight for ``barrier``; ignored if ``barrier`` is None.
            max_iterations: Maximum outer iLQR iterations.
            convergence_threshold: Stop when the accepted cost improvement is below this.
            initial_regularization: Initial Levenberg-Marquardt-style ``Q_uu`` regularization.
            regularization_scaling: Multiplicative factor used to grow/shrink regularization.
            max_regularization: Regularization ceiling; solver gives up above this.
            line_search_steps: Backtracking step sizes tried in the forward pass, largest first.
        """
        self.dynamics = dynamics
        self.state_dim = dynamics.state_dim
        self.control_dim = dynamics.control_dim
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.horizon = horizon
        self.dt = dt
        self.integration_method: IntegrationMethod = integration_method
        self.barrier = barrier
        self.barrier_weight = barrier_weight
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.initial_regularization = initial_regularization
        self.regularization_scaling = regularization_scaling
        self.max_regularization = max_regularization
        self.line_search_steps = tuple(line_search_steps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, initial_state: Tensor, initial_controls: Tensor | None = None) -> ILQRResult:
        """Run iLQR from ``initial_state``.

        Args:
            initial_state: Tensor of shape ``[batch, state_dim]``.
            initial_controls: Optional warm-start controls, shape
                ``[batch, horizon, control_dim]``; defaults to zeros.

        Returns:
            An :class:`ILQRResult`.
        """
        batch_size = initial_state.shape[0]
        controls = (
            initial_controls.clone()
            if initial_controls is not None
            else torch.zeros(
                batch_size, self.horizon, self.control_dim, dtype=initial_state.dtype, device=initial_state.device
            )
        )

        with torch.no_grad():
            states = self._rollout(initial_state, controls)
            cost = self._trajectory_cost(states, controls)

        cost_history = [cost.clone()]
        mu = self.initial_regularization
        converged = False
        iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            k, K, backward_ok = None, None, False
            for _ in range(self._MAX_REGULARIZATION_ATTEMPTS):
                k, K, backward_ok = self._backward_pass(states, controls, mu)
                if backward_ok:
                    break
                mu = min(mu * self.regularization_scaling, self.max_regularization)
            if not backward_ok:
                break

            accepted = False
            new_states = new_controls = new_cost = None
            for alpha in self.line_search_steps:
                new_states, new_controls, new_cost = self._forward_pass(states, controls, k, K, alpha)
                if torch.all(new_cost <= cost):
                    accepted = True
                    break

            if not accepted:
                mu = min(mu * self.regularization_scaling, self.max_regularization)
                if mu >= self.max_regularization:
                    break
                continue

            prev_cost = cost
            states, controls = new_states.detach(), new_controls.detach()
            cost = new_cost.detach()
            cost_history.append(cost.clone())
            mu = max(mu / self.regularization_scaling, self.initial_regularization * 1e-3)

            if torch.all(prev_cost - cost < self.convergence_threshold):
                converged = True
                break

        return ILQRResult(
            states=states,
            controls=controls,
            cost_history=cost_history,
            converged=converged,
            iterations=iteration,
        )

    # ------------------------------------------------------------------
    # Cost (with optional CBF penalty)
    # ------------------------------------------------------------------

    def _augmented_running_cost(self, state: Tensor, control: Tensor) -> Tensor:
        cost = self.running_cost(state, control)
        if self.barrier is not None and self.barrier_weight > 0:
            violation = torch.clamp(-self.barrier.value(state), min=0.0)
            cost = cost + self.barrier_weight * violation.pow(2).sum(dim=-1)
        return cost

    def _trajectory_cost(self, states: Tensor, controls: Tensor) -> Tensor:
        batch_size = states.shape[0]
        total = torch.zeros(batch_size, dtype=states.dtype, device=states.device)
        for t in range(self.horizon):
            total = total + self._augmented_running_cost(states[:, t], controls[:, t])
        return total + self.terminal_cost(states[:, -1])

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout(self, initial_state: Tensor, controls: Tensor) -> Tensor:
        states = [initial_state]
        state = initial_state
        for t in range(self.horizon):
            state = self.dynamics.step(state, controls[:, t], self.dt, method=self.integration_method)
            states.append(state)
        return torch.stack(states, dim=1)

    # ------------------------------------------------------------------
    # Local linearization / quadratic expansion via autograd
    # ------------------------------------------------------------------

    def _linearize_dynamics(self, state: Tensor, control: Tensor) -> tuple[Tensor, Tensor]:
        state = state.detach().clone().requires_grad_(True)
        control = control.detach().clone().requires_grad_(True)
        next_state = self.dynamics.step(state, control, self.dt, method=self.integration_method)
        f_x = _batch_jacobian(next_state, state)
        f_u = _batch_jacobian(next_state, control)
        return f_x.detach(), f_u.detach()

    def _running_cost_expansion(
        self, state: Tensor, control: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        state = state.detach().clone().requires_grad_(True)
        control = control.detach().clone().requires_grad_(True)
        cost = self._augmented_running_cost(state, control)
        batch_size, n, m = state.shape[0], state.shape[-1], control.shape[-1]
        if not cost.requires_grad:
            # Cost is a constant w.r.t. (state, control) — no graph to
            # differentiate; the correct expansion is identically zero.
            dtype, device = state.dtype, state.device
            return (
                cost.detach(),
                torch.zeros(batch_size, n, dtype=dtype, device=device),
                torch.zeros(batch_size, m, dtype=dtype, device=device),
                torch.zeros(batch_size, n, n, dtype=dtype, device=device),
                torch.zeros(batch_size, m, m, dtype=dtype, device=device),
                torch.zeros(batch_size, m, n, dtype=dtype, device=device),
            )
        l_x, l_u = torch.autograd.grad(cost.sum(), (state, control), create_graph=True)
        l_xx = _batch_jacobian(l_x, state)
        l_uu = _batch_jacobian(l_u, control)
        l_ux = _batch_jacobian(l_u, state)
        return (
            cost.detach(),
            l_x.detach(),
            l_u.detach(),
            l_xx.detach(),
            l_uu.detach(),
            l_ux.detach(),
        )

    def _terminal_cost_expansion(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        state = state.detach().clone().requires_grad_(True)
        cost = self.terminal_cost(state)
        batch_size, n = state.shape[0], state.shape[-1]
        if not cost.requires_grad:
            dtype, device = state.dtype, state.device
            return (
                cost.detach(),
                torch.zeros(batch_size, n, dtype=dtype, device=device),
                torch.zeros(batch_size, n, n, dtype=dtype, device=device),
            )
        (l_x,) = torch.autograd.grad(cost.sum(), state, create_graph=True)
        l_xx = _batch_jacobian(l_x, state)
        return cost.detach(), l_x.detach(), l_xx.detach()

    # ------------------------------------------------------------------
    # Backward pass (Riccati recursion)
    # ------------------------------------------------------------------

    def _backward_pass(self, states: Tensor, controls: Tensor, mu: float) -> tuple[Tensor, Tensor, bool]:
        batch_size = states.shape[0]
        n, m = self.state_dim, self.control_dim
        dtype, device = states.dtype, states.device

        _, V_x, V_xx = self._terminal_cost_expansion(states[:, -1])

        k = torch.zeros(batch_size, self.horizon, m, dtype=dtype, device=device)
        K = torch.zeros(batch_size, self.horizon, m, n, dtype=dtype, device=device)
        eye_m = torch.eye(m, dtype=dtype, device=device)

        for t in reversed(range(self.horizon)):
            x_t, u_t = states[:, t], controls[:, t]
            f_x, f_u = self._linearize_dynamics(x_t, u_t)
            _, l_x, l_u, l_xx, l_uu, l_ux = self._running_cost_expansion(x_t, u_t)

            Q_x = l_x + torch.einsum("bij,bi->bj", f_x, V_x)
            Q_u = l_u + torch.einsum("bij,bi->bj", f_u, V_x)
            Q_xx = l_xx + f_x.transpose(-1, -2) @ V_xx @ f_x
            f_u_T_Vxx = f_u.transpose(-1, -2) @ V_xx  # [batch, m, n]
            Q_uu = l_uu + f_u_T_Vxx @ f_u
            Q_ux = l_ux + f_u_T_Vxx @ f_x

            Q_uu_reg = Q_uu + mu * eye_m
            Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose(-1, -2))

            eigvals = torch.linalg.eigvalsh(Q_uu_reg)
            if torch.any(eigvals <= self._PD_EPS):
                return k, K, False

            k_t = -torch.linalg.solve(Q_uu_reg, Q_u.unsqueeze(-1)).squeeze(-1)
            K_t = -torch.linalg.solve(Q_uu_reg, Q_ux)
            k[:, t] = k_t
            K[:, t] = K_t

            V_x = (
                Q_x
                + (K_t.transpose(-1, -2) @ Q_uu_reg @ k_t.unsqueeze(-1)).squeeze(-1)
                + (K_t.transpose(-1, -2) @ Q_u.unsqueeze(-1)).squeeze(-1)
                + (Q_ux.transpose(-1, -2) @ k_t.unsqueeze(-1)).squeeze(-1)
            )
            V_xx = (
                Q_xx
                + K_t.transpose(-1, -2) @ Q_uu_reg @ K_t
                + K_t.transpose(-1, -2) @ Q_ux
                + Q_ux.transpose(-1, -2) @ K_t
            )
            V_xx = 0.5 * (V_xx + V_xx.transpose(-1, -2))

        return k, K, True

    # ------------------------------------------------------------------
    # Forward pass (line search rollout)
    # ------------------------------------------------------------------

    def _forward_pass(
        self, states: Tensor, controls: Tensor, k: Tensor, K: Tensor, alpha: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = states.shape[0]
        x_hat = states[:, 0]
        new_states = [x_hat]
        new_controls = []
        total_cost = torch.zeros(batch_size, dtype=states.dtype, device=states.device)

        for t in range(self.horizon):
            dx = x_hat - states[:, t]
            du = alpha * k[:, t] + torch.einsum("bij,bj->bi", K[:, t], dx)
            u_hat = controls[:, t] + du
            total_cost = total_cost + self._augmented_running_cost(x_hat, u_hat)
            x_hat = self.dynamics.step(x_hat, u_hat, self.dt, method=self.integration_method)
            new_states.append(x_hat)
            new_controls.append(u_hat)

        total_cost = total_cost + self.terminal_cost(x_hat)
        return torch.stack(new_states, dim=1), torch.stack(new_controls, dim=1), total_cost
