"""Headless benchmark comparing iLQR and iCEM as closed-loop MPC controllers.

Both optimizers plan from an identical offset cartpole initial state, in a
receding-horizon (replan-every-cycle) loop: at each control cycle, solve for
a control sequence from the current (noisy) state, apply only the first
action to the true dynamics, inject Gaussian state noise to emulate sensing
uncertainty, and replan from the resulting state.

Run directly:

    ./venv/bin/python experiments/benchmark_mpc.py

Prints a latency/cost/tracking-error comparison table and saves a
multi-panel diagnostic PNG to ``experiments/runs/``.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from experiments.plot_diagnostics import plot_diagnostics
from src.constraints.cbf import BarrierFunction, CombinedBarrier, ObstacleAvoidanceCBF, StateConstraintCBF
from src.dynamics.cartpole import CartpoleDynamics
from src.optimizers.icem import ICEM
from src.optimizers.ilqr import ILQR, QuadraticCost

RUNS_DIR = Path(__file__).resolve().parent / "runs"
TARGET_STATE = torch.zeros(4)


@dataclass
class ControllerRunResult:
    """Closed-loop evaluation results for one controller."""

    name: str
    states: Tensor
    """``[num_cycles + 1, state_dim]``: true state at each cycle boundary."""
    controls: Tensor
    """``[num_cycles, control_dim]``: applied (first-action) control per cycle."""
    barrier_values: Tensor
    """``[num_cycles + 1, num_constraints]``: CBF barrier value(s) at each true state."""
    solve_times: list[float]
    """Wall-clock seconds spent in ``optimize()`` per control cycle."""
    total_cost: float
    """Sum of running_cost over the closed-loop trajectory plus terminal_cost at the end."""
    final_tracking_error: float
    """Euclidean distance from the target state at the final cycle."""


def make_stabilization_cost() -> QuadraticCost:
    Q = torch.diag(torch.tensor([10.0, 1.0, 50.0, 1.0]))
    R = torch.diag(torch.tensor([0.01]))
    Q_terminal = torch.diag(torch.tensor([50.0, 5.0, 200.0, 5.0]))
    return QuadraticCost(Q=Q, R=R, Q_terminal=Q_terminal, target_state=TARGET_STATE)


def make_safety_barrier() -> CombinedBarrier:
    """An obstacle plus state bounds, combined into one CBF for diagnostics."""
    obstacle = ObstacleAvoidanceCBF(torch.tensor([[1.5]]), safe_radius=0.3, position_indices=(0,))
    state_bounds = StateConstraintCBF(
        torch.tensor([-2.5, -float("inf"), -0.7, -float("inf")]),
        torch.tensor([2.5, float("inf"), 0.7, float("inf")]),
    )
    return CombinedBarrier([obstacle, state_bounds])


def _shift_warm_start(controls: Tensor) -> Tensor:
    """Shift a ``[batch, horizon, control_dim]`` plan by one step, repeating the last action."""
    return torch.cat([controls[:, 1:, :], controls[:, -1:, :]], dim=1)


def _solve(solver: ILQR | ICEM, state: Tensor, warm_start: Tensor | None):
    if isinstance(solver, ILQR):
        return solver.optimize(state, initial_controls=warm_start)
    if isinstance(solver, ICEM):
        return solver.optimize(state, init_mean=warm_start)
    raise TypeError(f"Unsupported solver type: {type(solver)!r}")


def run_closed_loop(
    name: str,
    solver: ILQR | ICEM,
    dynamics: CartpoleDynamics,
    cost: QuadraticCost,
    barrier: BarrierFunction,
    initial_state: Tensor,
    num_cycles: int,
    dt: float,
    process_noise_std: float,
    seed: int,
) -> ControllerRunResult:
    """Run one controller in closed loop for ``num_cycles`` steps under state noise."""
    torch.manual_seed(seed)

    state = initial_state.clone()
    states = [state.squeeze(0).clone()]
    controls_log = []
    barrier_values = [barrier.value(state).squeeze(0).clone()]
    solve_times = []
    warm_start: Tensor | None = None

    for _ in range(num_cycles):
        start_time = time.perf_counter()
        result = _solve(solver, state, warm_start)
        solve_times.append(time.perf_counter() - start_time)

        u0 = result.controls[:, 0, :]
        warm_start = _shift_warm_start(result.controls)

        state = dynamics.step(state, u0, dt, method="rk4")
        if process_noise_std > 0:
            state = state + process_noise_std * torch.randn_like(state)

        states.append(state.squeeze(0).clone())
        controls_log.append(u0.squeeze(0).clone())
        barrier_values.append(barrier.value(state).squeeze(0).clone())

    states_t = torch.stack(states, dim=0)
    controls_t = torch.stack(controls_log, dim=0)
    barrier_t = torch.stack(barrier_values, dim=0)

    total_cost = 0.0
    for t in range(num_cycles):
        total_cost += cost.running_cost(states_t[t : t + 1], controls_t[t : t + 1]).item()
    total_cost += cost.terminal_cost(states_t[-1:]).item()

    final_tracking_error = torch.norm(states_t[-1] - TARGET_STATE).item()

    return ControllerRunResult(
        name=name,
        states=states_t,
        controls=controls_t,
        barrier_values=barrier_t,
        solve_times=solve_times,
        total_cost=total_cost,
        final_tracking_error=final_tracking_error,
    )


def print_summary(results: dict[str, ControllerRunResult]) -> None:
    header = (
        f"{'Controller':<10} {'Latency mean (ms)':>18} {'Latency std (ms)':>18} "
        f"{'Final track err':>16} {'Total cost':>14}"
    )
    print(header)
    print("-" * len(header))
    for result in results.values():
        times_ms = [t * 1000.0 for t in result.solve_times]
        mean_ms = statistics.mean(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        print(
            f"{result.name:<10} {mean_ms:>18.3f} {std_ms:>18.3f} "
            f"{result.final_tracking_error:>16.4f} {result.total_cost:>14.2f}"
        )


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    dynamics = CartpoleDynamics()
    cost = make_stabilization_cost()
    barrier = make_safety_barrier()

    initial_state = torch.tensor([[0.0, 0.0, 0.3, 0.0]])
    horizon = 20
    dt = 0.05
    num_cycles = 40
    process_noise_std = 0.01
    barrier_weight = 200.0
    seed = 0

    ilqr_solver = ILQR(
        dynamics,
        cost.running_cost,
        cost.terminal_cost,
        horizon=horizon,
        dt=dt,
        barrier=barrier,
        barrier_weight=barrier_weight,
        max_iterations=20,
    )
    icem_solver = ICEM(
        dynamics,
        cost.running_cost,
        cost.terminal_cost,
        horizon=horizon,
        dt=dt,
        barrier=barrier,
        barrier_weight=barrier_weight,
        num_samples=400,
        num_elites=40,
        num_iterations=10,
        init_std=2.0,
    )

    results = {}
    for name, solver in (("iLQR", ilqr_solver), ("iCEM", icem_solver)):
        results[name] = run_closed_loop(
            name, solver, dynamics, cost, barrier, initial_state, num_cycles, dt, process_noise_std, seed
        )

    print_summary(results)

    save_path = RUNS_DIR / "mpc_comparison.png"
    plot_diagnostics(results, dt=dt, save_path=save_path)
    print(f"\nSaved diagnostic plot to {save_path}")


if __name__ == "__main__":
    main()
