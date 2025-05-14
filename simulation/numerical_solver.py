import time
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
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


def spring_pendulum(t, x):
    """
    Calculates the time derivative of the state vector for the coupled system.

    Assumes the first equation was corrected to have ddz (z double dot).

    Args:
      t: Current time (float). Although not explicitly used in the equations,
         it's standard for ODE solver functions.
      x: Current state vector [x1, x2, x3, x4] as a NumPy array, where:
         x1 = z (position)
         x2 = z_dot (velocity)
         x3 = theta (angle)
         x4 = theta_dot (angular velocity)
      params: A dictionary containing the system parameters:
              'm1': Mass 1 (float)
              'm2': Mass 2 (float)
              'L': Length (float)
              'k': Spring constant (float)
              'g': Acceleration due to gravity (float)

    Returns:
      dxdt: The derivative of the state vector [dx1/dt, dx2/dt, dx3/dt, dx4/dt]
            as a NumPy array.
    """

    params = {
        "m1": 1.0,  # kg
        "m2": 0.1,  # kg
        "L": 2,  # m
        "k": 2.0,  # N/m
        "g": 20.81,  # m/s^2
    }

    # Unpack parameters
    m1 = params["m1"]
    m2 = params["m2"]
    L = params["L"]
    k = params["k"]
    g = params["g"]

    # Unpack state variables
    x1, x2, x3, x4 = x

    # Pre-calculate trigonometric terms and powers for efficiency and readability
    sin_x3 = np.sin(x3)
    cos_x3 = np.cos(x3)
    sin2_x3 = sin_x3**2  # More efficient than np.sin(x3)**2 inside loop
    x4_squared = x4**2

    # Calculate the determinant D = det(M)
    # D = m2 * L * ((m1 + m2) - m2 * L * sin(x3)**2)
    D = m2 * L * ((m1 + m2) - m2 * L * sin2_x3)

    # --- Check for singularity ---
    # If D is close to zero, the system is singular (cannot invert matrix M)
    # This might happen if m2 or L is zero, or if (m1+m2) = m2*L*sin^2(x3)
    if np.abs(D) < 1e-10:  # Use a small tolerance
        # Handle singularity: e.g., raise an error, return NaNs, or a large number
        # Returning NaNs is often preferred by solvers.
        print(
            f"Warning: Determinant D is close to zero ({D:.2e}) at t={t:.2f}, x3={x3:.2f}. System might be singular."
        )
        return np.array([np.nan, np.nan, np.nan, np.nan])
        # Alternatively, raise ValueError("Determinant D is near zero, system singular.")

    # Calculate dx2/dt (which is z_ddot)
    # dx2/dt = (m2 * L / D) * (-m2 * L * cos(x3) * x4**2 - (m1 + m2) * g - k * x1 + m2 * g * L * sin(x3)**2)
    term1_dx2 = -m2 * L * cos_x3 * x4_squared
    term2_dx2 = -(m1 + m2) * g
    term3_dx2 = -k * x1
    term4_dx2 = m2 * g * L * sin2_x3
    dx2dt = (m2 * L / D) * (term1_dx2 + term2_dx2 + term3_dx2 + term4_dx2)

    # Calculate dx4/dt (which is theta_ddot)
    # dx4/dt = (L * m2 * sin(x3) / D) * (m2 * L * cos(x3) * x4**2 + k * x1)
    term1_dx4 = m2 * L * cos_x3 * x4_squared
    term2_dx4 = k * x1
    dx4dt = (L * m2 * sin_x3 / D) * (term1_dx4 + term2_dx4)

    # Assemble the derivative vector
    dxdt = np.array(
        [
            x2,  # dx1/dt = x2 (z_dot)
            dx2dt,  # dx2/dt = z_ddot
            x4,  # dx3/dt = x4 (theta_dot)
            dx4dt,  # dx4/dt = theta_ddot
        ]
    )

    return dxdt


def beam_ball(t: float, state: NDArray[np.float64]):
    M = 10
    g = 9.81
    R = 10

    x, theta = state

    td = np.sqrt(-g * np.sin(theta) / (x * M))

    xd = (
        M * g * (R * np.sin(theta) - x * np.cos(theta))
        - 200 * theta
        - 70 * td
        + 200 * x
    ) / -70

    return np.array([xd, td])


def calculate_system_expressions(t, state):
    """
    Calculates the values of the two expressions derived from the system equations.

    Args:
      state: A list or NumPy array containing the state variables in the order:
             [x, xdot, theta, thetadot]
             - x: state variable 'x'
             - xdot: time derivative of 'x'
             - theta: state variable 'theta' (in radians)
             - thetadot: time derivative of 'theta'
      params: A dictionary or similar object containing the parameters:
              'M': Mass (or relevant parameter M)
              'g': Acceleration due to gravity (or relevant parameter g)
              'R': Radius (or relevant parameter R)

    Returns:
      A NumPy array containing the calculated values of the two expressions.
      [expression_1, expression_2]
    """
    # Unpack state variables
    # x, theta = state

    # params = {
    #     "M": 2.0,  # kg
    #     "g": 9.81,  # m/s^2
    #     "R": 0.5,  # m
    # }

    # # Unpack parameters
    # M = params["M"]
    # g = params["g"]
    # R = params["R"]

    # Pre-calculate trigonometric functions for efficiency
    # sin_theta = np.sin(theta)
    # cos_theta = np.cos(theta)

    # Calculate Expression 1
    # expr1 = -1.0 * M * (g * sin_theta + thetadot**2 * x)

    # Calculate Expression 2
    # term1_expr2 = M * g * (R * sin_theta - x * cos_theta)
    # term2_expr2 = -200.0 * theta
    # # term3_expr2 = -70.0 * thetadot
    # term4_expr2 = 200.0 * x
    # term5_expr2 = 70.0 * xdot
    # expr2 = term1_expr2 + term2_expr2 + term3_expr2 + term4_expr2 + term5_expr2

    # Return the results as a NumPy array
    # return np.array([expr1, expr2])


def dae_rod_on_cart_rhs(
    t: float, state: NDArray[np.float64], params: dict
) -> NDArray[np.float64]:
    """
    RHS function for the DAE system describing a rod connected to a moving cart.
    The system equations are:
        J * θ_ddot = τ + λ*R*x*sin(θ)
        m * x_ddot = F - m*g + λx - λ*R*cos(θ)
    Algebraic constraint:
        C(x, θ) = 0.5 * (x^2 + R^2 - L^2 - 2*x*R*cos(θ)) = 0

    This function calculates [dθ/dt, dω/dt, dx/dt, dv/dt]
    where ω = dθ/dt (theta_dot), v = dx/dt (x_dot).
    The Lagrange multiplier λ is determined by ensuring d²C/dt² = 0.
    The interpretation lambda_x = λ*x is used, making the second equation:
        m * x_ddot = F - m*g + λ*(x - R*cos(θ))

    Args:
        t (float): Current time.
        state (NDArray[np.float64]): Current state [theta, omega, x, v].
        params (dict): Dictionary of system parameters:
            'J': Moment of inertia of the rod.
            'm': Mass of the cart.
            'R': A characteristic length (e.g., distance from pivot on cart to a point on rod, or rod length if pivot is at one end).
            'L': Length parameter from the algebraic constraint.
            'tau': Applied torque on the rod (can be float or callable f(t)).
            'F': Applied external force on the cart (can be float or callable f(t)).
            'g': Acceleration due to gravity.

    Returns:
        NDArray[np.float64]: Derivatives [d_theta_dt, d_omega_dt, d_x_dt, d_v_dt].
    """
    theta, omega, x, v = state

    # Unpack parameters
    J = params["J"]
    m_cart = params["m"]
    R_len = params["R"]  # Using R_len to distinguish from R in equations if needed
    # L_len = params['L'] # L is for constraint consistency, not directly in lambda derivation here
    g_accel = params["g"]

    # Evaluate time-dependent inputs if they are callable
    current_tau = params["tau"](t) if callable(params["tau"]) else params["tau"]
    current_F = params["F"](t) if callable(params["F"]) else params["F"]

    # Pre-calculate trigonometric terms
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Calculate terms for d²C/dt² to find lambda.
    # d²C/dt² = (x_ddot * (x - R_len*cos_theta) + theta_ddot * (x*R_len*sin_theta) +
    #            v**2 + 2*R_len*v*omega*sin_theta + x*R_len*omega**2*cos_theta) = 0
    # Let C_tt_known = v**2 + 2*R_len*v*omega*sin_theta + x*R_len*omega**2*cos_theta
    C_tt_known = (
        v**2 + 2 * R_len * v * omega * sin_theta + x * R_len * omega**2 * cos_theta
    )

    # Substitute expressions for theta_ddot and x_ddot into d²C/dt² = 0:
    # theta_ddot = (current_tau + lambda_val * R_len * x * sin_theta) / J
    # x_ddot = (current_F - m_cart*g_accel + lambda_val * (x - R_len*cos_theta)) / m_cart

    # (x - R_len*cos_theta) * [ (current_F - m_cart*g_accel)/m_cart + lambda_val*(x-R_len*cos_theta)/m_cart ] +
    # (x*R_len*sin_theta) * [ current_tau/J + lambda_val*(R_len*x*sin_theta)/J ] + C_tt_known = 0
    # Solve for lambda_val:
    # lambda_val * [ (x-R_len*cos_theta)²/m_cart + (x*R_len*sin_theta)²/J ] =
    #   -C_tt_known - (x-R_len*cos_theta)*(current_F-m_cart*g_accel)/m_cart - (x*R_len*sin_theta)*current_tau/J

    coeff_A_sq_m = (x - R_len * cos_theta) ** 2 / m_cart
    coeff_B_sq_J = (x * R_len * sin_theta) ** 2 / J
    lambda_denominator = coeff_A_sq_m + coeff_B_sq_J

    if np.abs(lambda_denominator) < 1e-12:  # Singularity check
        print(
            f"Warning: Singularity at t={t:.4f}. Denominator for lambda is {lambda_denominator:.3e}."
        )
        # This can occur at specific configurations (e.g., x = R*cos(theta) and x*R*sin(theta) = 0 simultaneously)
        # or if J or m_cart are zero.
        # Return NaNs for accelerations as the system is ill-defined here.
        return np.array([omega, np.nan, v, np.nan])

    num_term1 = -C_tt_known
    num_term2 = -(x - R_len * cos_theta) * (current_F - m_cart * g_accel) / m_cart
    num_term3 = -(x * R_len * sin_theta) * current_tau / J
    lambda_numerator = num_term1 + num_term2 + num_term3

    lambda_val = lambda_numerator / lambda_denominator

    # Calculate accelerations (d_omega_dt and d_v_dt) using the determined lambda_val
    d_omega_dt = (current_tau + lambda_val * R_len * x * sin_theta) / J
    d_v_dt = (
        current_F - m_cart * g_accel + lambda_val * (x - R_len * cos_theta)
    ) / m_cart

    # Derivatives of state variables
    d_theta_dt = omega
    d_x_dt = v

    return np.array([d_theta_dt, d_omega_dt, d_x_dt, d_v_dt])


def mass_on_cart_system() -> NDArray[np.float64]:
    """
    Simulates the mass on a cart system using the NumericalSolver class.
    The system is described by the DAE equations for a rod connected to a moving cart.
    The simulation uses the RK4 method for numerical integration.
    Returns:
        [
            theta (angle),
            omega (angular velocity),
            x (position of the cart),
            v (velocity of the cart)
        ]
    """

    system_params = {
        "J": 0.1,  # Moment of inertia of the rod
        "m": 1.0,  # Mass of the cart
        "R": 0.5,  # Length parameter (e.g., distance from pivot to a point on the rod)
        "L": 1.0,  # Length parameter for the algebraic constraint
        "tau": 0.0,  # Applied torque (can be a function of time)
        "F": 0.0,  # Applied external force (can be a function of time)
        "g": 9.81,  # Acceleration due to gravity
    }

    wrapped_dae_rhs = partial(dae_rod_on_cart_rhs, params=system_params)

    # initial_state = np.array([1, 0.0, np.pi / 6, 0.5])
    initial_state = np.array(
        [
            0.5,  # Initial angle (theta)
            1,  # Initial angular velocity (omega)
            5,  # Initial position (x)
            0.0,  # Initial velocity (v)
        ]
    )

    solver = NumericalSolver(rk4_tableau, wrapped_dae_rhs, profiling=True)
    start_time = np.float64(0)
    end_time = 20
    time_step = 0.01
    solution = solver.solve(initial_state, start_time, end_time, time_step)

    return solution


if __name__ == "__main__":
    # initial_state = np.array([1, 0.0, np.pi / 6, 0.5])
    initial_state = np.array(
        [
            0.5,  # Initial angle (theta)
            1,  # Initial angular velocity (omega)
            5,  # Initial position (x)
            0.0,  # Initial velocity (v)
        ]
    )

    solver = NumericalSolver(rk4_tableau, spring_pendulum, profiling=True)
    start_time = np.float64(0)
    end_time = 20
    time_step = 0.01
    solution = solver.solve(initial_state, start_time, end_time, time_step)
    time_series = np.linspace(start_time, end_time, len(solution))

    solver.print_execution_time()  # Print results

    plt.plot(time_series, solution)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Numerical Solver Example")
    plt.show()
