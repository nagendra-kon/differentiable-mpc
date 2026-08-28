"""Diagnostic plotting utilities for MPC controller comparison runs.

Renders headless (Agg backend, no display required) matplotlib figures
comparing one or more closed-loop controller runs: state trajectories,
control inputs, and CBF barrier values over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import Tensor


class DiagnosticRun(Protocol):
    """Structural interface a plotted run must satisfy."""

    name: str
    states: Tensor
    """``[T + 1, state_dim]``: state at each control cycle boundary."""
    controls: Tensor
    """``[T, control_dim]``: applied control at each cycle."""
    barrier_values: Tensor
    """``[T + 1, num_constraints]``: CBF barrier value(s) at each state."""


def plot_diagnostics(results: dict[str, DiagnosticRun], dt: float, save_path: str | Path) -> Path:
    """Render a multi-panel comparison figure and save it as a PNG.

    Panels (top to bottom): cart position vs time, pole angle vs time,
    control input vs time, and the minimum CBF barrier value vs time — one
    line per controller in ``results``, all sharing a time axis. The
    barrier panel's zero line marks the safety boundary (``h(x) >= 0`` is
    safe).

    Args:
        results: Mapping of controller name -> run exposing ``states``
            ``[T+1, state_dim]``, ``controls`` ``[T, control_dim]``, and
            ``barrier_values`` ``[T+1, num_constraints]``.
        dt: Control cycle duration, used to build the time axis.
        save_path: Output PNG path; parent directories are created if needed.

    Returns:
        The resolved ``save_path``.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)

    for name, result in results.items():
        states = result.states.detach().cpu()
        controls = result.controls.detach().cpu()
        barrier_values = result.barrier_values.detach().cpu()

        time_states = torch.arange(states.shape[0]) * dt
        time_controls = torch.arange(controls.shape[0]) * dt
        min_barrier = barrier_values.min(dim=-1).values

        axes[0].plot(time_states, states[:, 0], label=name)
        axes[1].plot(time_states, states[:, 2], label=name)
        axes[2].plot(time_controls, controls[:, 0], label=name)
        axes[3].plot(time_states, min_barrier, label=name)

    axes[0].set_ylabel("cart position x (m)")
    axes[0].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("pole angle theta (rad)")
    axes[1].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("control force (N)")
    axes[3].set_ylabel("min CBF barrier value")
    axes[3].axhline(0.0, color="red", linewidth=0.8, linestyle="--", label="safety boundary")
    axes[3].set_xlabel("time (s)")

    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("iLQR vs iCEM: Closed-Loop Cartpole Stabilization Under State Noise")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path
