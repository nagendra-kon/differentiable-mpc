"""Differentiable Control Barrier Functions (CBFs) for safety-constrained MPC.

A barrier function ``h(x)`` defines a safe set ``{x : h(x) >= 0}``. All
barrier values here are batched PyTorch tensors that stay part of the
autograd graph (no detaching), so gradients backpropagate through safety
terms in an MPC loss or a CBF-QP constraint.

State/control layout follows ``src.dynamics.cartpole``: state
``[x, x_dot, theta, theta_dot]``, control ``[force]`` — but every class here
is generic and works on any batched tensor of shape ``[batch_size, dim]``.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


class BarrierFunction:
    """Base class for a differentiable barrier function ``h(x) >= 0`` (safe)."""

    def value(self, x: Tensor) -> Tensor:
        """Compute barrier value(s) for a batch of inputs.

        Args:
            x: Tensor of shape ``[batch_size, dim]``.

        Returns:
            Tensor of shape ``[batch_size, num_constraints]``; ``h(x) >= 0``
            means safe for every constraint column.
        """
        raise NotImplementedError

    def min_value(self, x: Tensor) -> Tensor:
        """Worst-case (minimum) barrier value across constraints, per batch element.

        Returns:
            Tensor of shape ``[batch_size]``.
        """
        h = self.value(x)
        return torch.min(h, dim=-1).values

    def is_safe(self, x: Tensor, tol: float = 0.0) -> Tensor:
        """Boolean mask of shape ``[batch_size]``: True where every constraint holds."""
        h = self.value(x)
        return torch.all(h >= -tol, dim=-1)


class ObstacleAvoidanceCBF(BarrierFunction):
    """CBF enforcing a minimum distance from point obstacles.

    ``h_i(x) = ||p(x) - obstacle_i||^2 - safe_radius^2`` (>= 0 is safe).

    Squared distance is used instead of Euclidean distance to avoid the
    gradient singularity of ``sqrt(.)`` as separation approaches zero.
    """

    def __init__(
        self,
        obstacles: Tensor,
        safe_radius: float,
        position_indices: Sequence[int] = (0,),
    ) -> None:
        """
        Args:
            obstacles: Tensor of shape ``[num_obstacles, workspace_dim]``
                giving each obstacle's position.
            safe_radius: Minimum allowed distance to any obstacle.
            position_indices: Indices into the state vector that give the
                workspace position (e.g. ``(0,)`` for cart position ``x``).
        """
        if obstacles.dim() != 2 or obstacles.shape[-1] != len(position_indices):
            raise ValueError(
                f"obstacles must have shape [num_obstacles, {len(position_indices)}], "
                f"got {tuple(obstacles.shape)}"
            )
        if safe_radius <= 0:
            raise ValueError(f"safe_radius must be positive, got {safe_radius}")
        self.obstacles = obstacles
        self.safe_radius = safe_radius
        self.position_indices = list(position_indices)

    def value(self, x: Tensor) -> Tensor:
        position = x[:, self.position_indices]  # [batch, workspace_dim]
        obstacles = self.obstacles.to(dtype=x.dtype, device=x.device)
        diff = position.unsqueeze(1) - obstacles.unsqueeze(0)  # [batch, num_obstacles, workspace_dim]
        sq_dist = diff.pow(2).sum(dim=-1)  # [batch, num_obstacles]
        return sq_dist - self.safe_radius**2


class BoxConstraintCBF(BarrierFunction):
    """CBF enforcing elementwise bounds ``lower <= x <= upper``.

    For each finitely-bounded dimension this contributes one barrier value
    per active side: ``h_lower = x - lower`` and ``h_upper = upper - x``,
    both ``>= 0`` when safe. Use ``float("-inf")``/``float("inf")`` to leave
    a dimension's lower/upper side unconstrained.
    """

    def __init__(self, lower: Tensor, upper: Tensor) -> None:
        if lower.shape != upper.shape or lower.dim() != 1:
            raise ValueError(
                f"lower and upper must be 1D tensors of the same shape, "
                f"got {tuple(lower.shape)} and {tuple(upper.shape)}"
            )
        if torch.any(upper < lower):
            raise ValueError("upper bound must be >= lower bound for every dimension")
        self.lower = lower
        self.upper = upper
        self._lower_idx = torch.isfinite(lower).nonzero(as_tuple=True)[0]
        self._upper_idx = torch.isfinite(upper).nonzero(as_tuple=True)[0]
        if self._lower_idx.numel() == 0 and self._upper_idx.numel() == 0:
            raise ValueError("at least one dimension must have a finite lower or upper bound")

    def value(self, x: Tensor) -> Tensor:
        if x.dim() != 2 or x.shape[-1] != self.lower.shape[0]:
            raise ValueError(f"x must have shape [batch, {self.lower.shape[0]}], got {tuple(x.shape)}")
        lower = self.lower.to(dtype=x.dtype, device=x.device)
        upper = self.upper.to(dtype=x.dtype, device=x.device)

        components = []
        if self._lower_idx.numel() > 0:
            components.append(x[:, self._lower_idx] - lower[self._lower_idx])
        if self._upper_idx.numel() > 0:
            components.append(upper[self._upper_idx] - x[:, self._upper_idx])
        return torch.cat(components, dim=-1)


class StateConstraintCBF(BoxConstraintCBF):
    """Box-constraint CBF wrapper for state bounds (e.g. cart position, pole angle)."""


class InputConstraintCBF(BoxConstraintCBF):
    """Box-constraint CBF wrapper for control/input bounds (e.g. force limits)."""


class CombinedBarrier(BarrierFunction):
    """Concatenates several barrier functions into a single constraint set."""

    def __init__(self, barriers: Sequence[BarrierFunction]) -> None:
        if not barriers:
            raise ValueError("barriers must be non-empty")
        self.barriers = list(barriers)

    def value(self, x: Tensor) -> Tensor:
        return torch.cat([b.value(x) for b in self.barriers], dim=-1)


def discrete_cbf_condition(h_current: Tensor, h_next: Tensor, alpha: float = 1.0) -> Tensor:
    """Discrete-time CBF forward-invariance condition.

    Enforcing ``discrete_cbf_condition(...) >= 0`` at every timestep keeps a
    trajectory inside the safe set ``{h >= 0}``, given ``0 < alpha <= 1``:

        ``h(x_{t+1}) - h(x_t) + alpha * h(x_t) >= 0``

    Args:
        h_current: Barrier value(s) at time t.
        h_next: Barrier value(s) at time t + 1, same shape as ``h_current``.
        alpha: Class-K decay rate in ``(0, 1]``; ``alpha=1`` requires ``h``
            to stay non-negative outright, smaller values allow more
            transient decay before recovering.

    Returns:
        Tensor the same shape as ``h_current``; non-negative means safe.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return h_next - (1.0 - alpha) * h_current
