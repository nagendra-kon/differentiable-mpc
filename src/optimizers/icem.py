"""Batched iCEM (Iterative Cross-Entropy Method) trajectory optimizer.

Solves, for each batch element independently but vectorized together:

    min_{u_0..u_{T-1}}  sum_t running_cost(x_t, u_t) + terminal_cost(x_T)
    s.t.                x_{t+1} = dynamics.step(x_t, u_t, dt)

by iteratively refining a diagonal Gaussian over control sequences:
sample a population, roll every sample out through ``dynamics``, score it
with the cost (plus an optional CBF penalty), keep the lowest-cost "elite"
fraction, and refit the sampling distribution's mean/std from those elites.
This is a zeroth-order, gradient-free optimizer — it never needs autograd,
only forward evaluation of ``dynamics``/``running_cost``/``terminal_cost``,
so the whole search runs under ``torch.no_grad()``.

Both the outer batch dimension (independent problems, e.g. different
cartpole initial states) and the inner population dimension (samples per
problem) are flattened into one mega-batch for the rollout, so sampling,
simulation, and elite selection are all fully vectorized — no Python loop
over samples, only over the (small) planning horizon and CEM iterations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

import torch
from torch import Tensor

from src.constraints.cbf import BarrierFunction

IntegrationMethod = Literal["euler", "rk4"]


class DynamicsModel(Protocol):
    """Structural interface an iCEM-compatible dynamics model must satisfy."""

    state_dim: int
    control_dim: int

    def step(self, state: Tensor, control: Tensor, dt: float, method: IntegrationMethod = ...) -> Tensor:
        """Advance ``state`` by one timestep under ``control``. See e.g. CartpoleDynamics.step."""
        ...


@dataclass
class ICEMResult:
    """Output of :meth:`ICEM.optimize`."""

    states: Tensor
    """Rollout of the best-found control sequence, shape ``[batch, horizon + 1, state_dim]``."""

    controls: Tensor
    """Best-found (all-time elite) control sequence, shape ``[batch, horizon, control_dim]``."""

    mean: Tensor
    """Final sampling-distribution mean, shape ``[batch, horizon, control_dim]``."""

    std: Tensor
    """Final sampling-distribution std, shape ``[batch, horizon, control_dim]``."""

    cost_history: list[Tensor] = field(default_factory=list)
    """Best-so-far cost after each iteration, each ``[batch]``."""

    iterations: int = 0
    """Number of CEM iterations performed."""


class ICEM:
    """Batched Iterative Cross-Entropy Method trajectory optimizer."""

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
        num_samples: int = 200,
        num_elites: int = 20,
        num_iterations: int = 10,
        init_std: float = 1.0,
        alpha: float = 0.1,
        min_std: float = 1e-3,
        control_low: Tensor | float | None = None,
        control_high: Tensor | float | None = None,
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
            num_samples: Population size sampled per iteration, per batch element.
            num_elites: Number of lowest-cost samples used to refit the distribution.
            num_iterations: Number of sample/evaluate/refit iterations.
            init_std: Initial per-dimension standard deviation of the sampling Gaussian.
            alpha: Distribution update smoothing in ``[0, 1)``; ``new = alpha*old + (1-alpha)*elite_stat``.
            min_std: Std floor, prevents premature distribution collapse.
            control_low: Optional elementwise lower bound applied to sampled/returned controls.
            control_high: Optional elementwise upper bound applied to sampled/returned controls.
        """
        if num_elites >= num_samples:
            raise ValueError(f"num_elites ({num_elites}) must be less than num_samples ({num_samples})")
        if num_elites < 1:
            raise ValueError(f"num_elites must be >= 1, got {num_elites}")

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
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.init_std = init_std
        self.alpha = alpha
        self.min_std = min_std
        self.control_low = control_low
        self.control_high = control_high

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def optimize(self, initial_state: Tensor, init_mean: Tensor | None = None) -> ICEMResult:
        """Run iCEM from ``initial_state``.

        Args:
            initial_state: Tensor of shape ``[batch, state_dim]``.
            init_mean: Optional warm-start mean, shape ``[batch, horizon, control_dim]``;
                defaults to zeros.

        Returns:
            An :class:`ICEMResult`.
        """
        batch_size = initial_state.shape[0]
        dtype, device = initial_state.dtype, initial_state.device
        n, m, T = self.state_dim, self.control_dim, self.horizon

        mean = (
            init_mean.clone()
            if init_mean is not None
            else torch.zeros(batch_size, T, m, dtype=dtype, device=device)
        )
        std = torch.full((batch_size, T, m), self.init_std, dtype=dtype, device=device)

        best_cost = torch.full((batch_size,), float("inf"), dtype=dtype, device=device)
        best_controls = mean.clone()
        cost_history = []

        flat_initial_state = (
            initial_state.unsqueeze(1).expand(batch_size, self.num_samples, n).reshape(batch_size * self.num_samples, n)
        )

        iteration = 0
        for iteration in range(1, self.num_iterations + 1):
            noise = torch.randn(batch_size, self.num_samples, T, m, dtype=dtype, device=device)
            samples = mean.unsqueeze(1) + std.unsqueeze(1) * noise
            samples = self._clamp_controls(samples)

            flat_samples = samples.reshape(batch_size * self.num_samples, T, m)
            flat_states = self._rollout(flat_initial_state, flat_samples)
            flat_costs = self._trajectory_cost(flat_states, flat_samples)
            costs = flat_costs.reshape(batch_size, self.num_samples)

            elite_idx = torch.topk(costs, k=self.num_elites, dim=1, largest=False).indices
            gather_idx = elite_idx.view(batch_size, self.num_elites, 1, 1).expand(-1, -1, T, m)
            elite_samples = torch.gather(samples, dim=1, index=gather_idx)

            elite_mean = elite_samples.mean(dim=1)
            elite_std = elite_samples.std(dim=1, unbiased=False)

            mean = self.alpha * mean + (1.0 - self.alpha) * elite_mean
            std = self.alpha * std + (1.0 - self.alpha) * elite_std
            std = torch.clamp(std, min=self.min_std)

            iter_best_cost, iter_best_local_idx = costs.min(dim=1)
            improved = iter_best_cost < best_cost
            if torch.any(improved):
                iter_best_controls = samples[torch.arange(batch_size, device=device), iter_best_local_idx]
                best_controls = torch.where(improved.view(-1, 1, 1), iter_best_controls, best_controls)
                best_cost = torch.where(improved, iter_best_cost, best_cost)

            cost_history.append(best_cost.clone())

        best_states = self._rollout(initial_state, best_controls)
        return ICEMResult(
            states=best_states,
            controls=best_controls,
            mean=mean,
            std=std,
            cost_history=cost_history,
            iterations=iteration,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _clamp_controls(self, controls: Tensor) -> Tensor:
        if self.control_low is not None:
            controls = torch.clamp(controls, min=self.control_low)
        if self.control_high is not None:
            controls = torch.clamp(controls, max=self.control_high)
        return controls

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

    def _rollout(self, initial_state: Tensor, controls: Tensor) -> Tensor:
        states = [initial_state]
        state = initial_state
        for t in range(self.horizon):
            state = self.dynamics.step(state, controls[:, t], self.dt, method=self.integration_method)
            states.append(state)
        return torch.stack(states, dim=1)
