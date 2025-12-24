import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class ButcherTableau:
    def __init__(
        self,
        a: NDArray[np.float64] | None = None,
        b: NDArray[np.float64] | None = None,
        c: NDArray[np.float64] | None = None,
    ) -> None:
        if a is None:
            a = np.array([[0]])
        if b is None:
            b = np.array([1])
        if c is None:
            c = np.array([0])
        self.a = a
        self.b = b
        self.c = c
        self.stages = len(a)


# 1. Forward Euler (Explicit, 1st Order)
euler_tableau = ButcherTableau(a=np.array([[0]]), b=np.array([1]), c=np.array([0]))
# Note: This is the default if no arguments are given to ButcherTableau

# 2. Midpoint Method (Explicit Runge-Kutta, 2nd Order)
midpoint_tableau = ButcherTableau(
    a=np.array([[0, 0], [1 / 2, 0]]), b=np.array([0, 1]), c=np.array([0, 1 / 2])
)

# 2nd order Runge-Kutta method
rk2_tableu = ButcherTableau(
    a=np.array([[0, 0], [1 / 2, 0]]),
    b=np.array([1 / 2, 1 / 2]),
    c=np.array([0, 1]),
)

# 3. Heun's Method (Explicit Runge-Kutta, 2nd Order, aka Trapezoidal Rule)
heun_tableau = ButcherTableau(
    a=np.array([[0, 0], [1, 0]]), b=np.array([1 / 2, 1 / 2]), c=np.array([0, 1])
)

# 4. Ralston's Method (Explicit Runge-Kutta, 2nd Order, minimum error bound)
ralston_tableau = ButcherTableau(
    a=np.array([[0, 0], [2 / 3, 0]]), b=np.array([1 / 4, 3 / 4]), c=np.array([0, 2 / 3])
)

# 5. Classic Runge-Kutta (RK4, Explicit, 4th Order)
rk4_tableau = ButcherTableau(
    a=np.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]]),
    b=np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
    c=np.array([0, 1 / 2, 1 / 2, 1]),
)


class NumericalSolver:
    def __init__(
        self,
        butcherTableau: ButcherTableau,
        d_function: Callable[[np.float64, NDArray[np.float64]], NDArray[np.float64]],
        profiling: bool = False,
    ) -> None:
        self.butcher = butcherTableau
        self.d_func = d_function
        self.profiling = profiling
        if self.profiling:
            self.time_get_ks_total: float = 0.0
            self.time_update_total: float = 0.0
            self.time_solve_total: float = 0.0
            self.solve_calls: int = 0
            self.steps_total: int = 0

    def get_ks(
        self, t_k: float, x_k: NDArray[np.float64], dt: float
    ) -> NDArray[np.float64]:
        k_stages = np.zeros((self.butcher.stages, x_k.shape[0]))

        for i in range(len(k_stages)):
            x_increment = sum(a[i] * k_stages[j] for j, a in enumerate(self.butcher.a))
            k = self.d_func(t_k + dt * self.butcher.c[i], x_k + dt * x_increment)
            k_stages[i] = k

        return k_stages

    def print_execution_time(self):
        """Prints the execution time of the different stages of the solving process."""
        if not self.profiling or self.solve_calls == 0:
            print("Profiling was not enabled or solve() has not been called.")
            return

        avg_total_time = self.time_solve_total / self.solve_calls
        avg_get_ks_time = self.time_get_ks_total / self.solve_calls
        avg_update_time = self.time_update_total / self.solve_calls
        avg_steps = self.steps_total / self.solve_calls

        print("\n--- Solver Profiling ---")
        print(f"Total calls to solve(): {self.solve_calls}")
        print(f"Average steps per call: {avg_steps:.2f}")
        print(f"Average total time per solve() call: {avg_total_time:.6f} seconds")
        print(f"  Average time in get_ks() per call:   {avg_get_ks_time:.6f} seconds")
        print(f"  Average time in update step per call: {avg_update_time:.6f} seconds")

        if self.steps_total > 0:
            avg_get_ks_per_step = self.time_get_ks_total / self.steps_total
            avg_update_per_step = self.time_update_total / self.steps_total
            print("\nAverage time per step:")
            print(f"  get_ks(): {avg_get_ks_per_step:.8f} seconds")
            print(f"  update:   {avg_update_per_step:.8f} seconds")
        print("------------------------\n")

    def solve(
        self,
        initial_state: NDArray[np.float64],
        start_time: float,
        end_time: float,
        dt: float,
    ) -> NDArray[np.float64]:
        solve_start_time = time.perf_counter() if self.profiling else 0
        local_time_get_ks = 0.0
        local_time_update = 0.0

        n_steps = int((end_time - start_time) / dt)
        timeSeries = np.zeros((n_steps + 1, len(initial_state)))
        timeSeries[0] = initial_state

        for i in range(n_steps):
            t_k = start_time + i * dt
            x_k = timeSeries[i]

            ks_start_time = time.perf_counter() if self.profiling else 0

            k_stages = self.get_ks(t_k, x_k, dt)

            ks_end_time = time.perf_counter() if self.profiling else 0
            if self.profiling:
                local_time_get_ks += ks_end_time - ks_start_time

            update_start_time = time.perf_counter() if self.profiling else 0
            x_k1 = x_k + dt * sum(b * k for b, k in zip(self.butcher.b, k_stages))
            timeSeries[i + 1] = x_k1
            update_end_time = time.perf_counter() if self.profiling else 0
            if self.profiling:
                local_time_update += update_end_time - update_start_time

        solve_end_time = time.perf_counter() if self.profiling else 0
        if self.profiling:
            self.time_get_ks_total += local_time_get_ks
            self.time_update_total += local_time_update
            self.time_solve_total += solve_end_time - solve_start_time
            self.solve_calls += 1
            self.steps_total += n_steps

        return timeSeries
