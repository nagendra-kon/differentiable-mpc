"""Smoke tests for the headless iLQR-vs-iCEM benchmark runner and plotting utility."""

from pathlib import Path

import torch

from experiments.benchmark_mpc import make_safety_barrier, make_stabilization_cost, run_closed_loop
from experiments.plot_diagnostics import plot_diagnostics
from src.dynamics.cartpole import CartpoleDynamics
from src.optimizers.icem import ICEM
from src.optimizers.ilqr import ILQR


def test_run_closed_loop_ilqr_produces_well_formed_result() -> None:
    dynamics = CartpoleDynamics()
    cost = make_stabilization_cost()
    barrier = make_safety_barrier()
    initial_state = torch.tensor([[0.0, 0.0, 0.1, 0.0]])
    num_cycles = 3

    solver = ILQR(dynamics, cost.running_cost, cost.terminal_cost, horizon=5, dt=0.05, max_iterations=5)
    result = run_closed_loop(
        "iLQR", solver, dynamics, cost, barrier, initial_state, num_cycles, dt=0.05, process_noise_std=0.01, seed=0
    )

    assert result.states.shape == (num_cycles + 1, 4)
    assert result.controls.shape == (num_cycles, 1)
    assert result.barrier_values.shape[0] == num_cycles + 1
    assert len(result.solve_times) == num_cycles
    assert all(t >= 0 for t in result.solve_times)
    assert isinstance(result.total_cost, float)
    assert isinstance(result.final_tracking_error, float)


def test_run_closed_loop_icem_produces_well_formed_result() -> None:
    dynamics = CartpoleDynamics()
    cost = make_stabilization_cost()
    barrier = make_safety_barrier()
    initial_state = torch.tensor([[0.0, 0.0, 0.1, 0.0]])
    num_cycles = 3

    solver = ICEM(
        dynamics, cost.running_cost, cost.terminal_cost, horizon=5, dt=0.05, num_samples=20, num_elites=4, num_iterations=3
    )
    result = run_closed_loop(
        "iCEM", solver, dynamics, cost, barrier, initial_state, num_cycles, dt=0.05, process_noise_std=0.01, seed=0
    )

    assert result.states.shape == (num_cycles + 1, 4)
    assert result.controls.shape == (num_cycles, 1)
    assert result.barrier_values.shape[0] == num_cycles + 1
    assert len(result.solve_times) == num_cycles


def test_process_noise_perturbs_the_trajectory() -> None:
    dynamics = CartpoleDynamics()
    cost = make_stabilization_cost()
    barrier = make_safety_barrier()
    initial_state = torch.tensor([[0.0, 0.0, 0.1, 0.0]])

    solver_a = ILQR(dynamics, cost.running_cost, cost.terminal_cost, horizon=5, dt=0.05, max_iterations=5)
    result_a = run_closed_loop(
        "iLQR", solver_a, dynamics, cost, barrier, initial_state, 3, dt=0.05, process_noise_std=0.5, seed=0
    )

    solver_b = ILQR(dynamics, cost.running_cost, cost.terminal_cost, horizon=5, dt=0.05, max_iterations=5)
    result_b = run_closed_loop(
        "iLQR", solver_b, dynamics, cost, barrier, initial_state, 3, dt=0.05, process_noise_std=0.5, seed=1
    )

    # Different noise seeds under a large process_noise_std should diverge.
    assert not torch.allclose(result_a.states, result_b.states)


def test_plot_diagnostics_saves_png(tmp_path: Path) -> None:
    dynamics = CartpoleDynamics()
    cost = make_stabilization_cost()
    barrier = make_safety_barrier()
    initial_state = torch.tensor([[0.0, 0.0, 0.1, 0.0]])

    solver = ILQR(dynamics, cost.running_cost, cost.terminal_cost, horizon=5, dt=0.05, max_iterations=5)
    result = run_closed_loop(
        "iLQR", solver, dynamics, cost, barrier, initial_state, 3, dt=0.05, process_noise_std=0.0, seed=0
    )

    save_path = tmp_path / "diagnostics.png"
    returned_path = plot_diagnostics({"iLQR": result}, dt=0.05, save_path=save_path)

    assert returned_path == save_path
    assert save_path.exists()
    assert save_path.stat().st_size > 0
