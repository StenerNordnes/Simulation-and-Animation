import numpy as np
from functools import partial
from numpy.typing import NDArray
from solver.numerical_solver import (
    NumericalSolver,
    rk4_tableau,
)
from animation.objects import ( 
    Circle,
    ConnectiveRod,
    Rectangle,
    animate_objects,
)


def dae_rod_on_cart_rhs(
    t: float, state: NDArray[np.float64], params: dict
) -> NDArray[np.float64]:
    """
    RHS function for the DAE system describing a rod connected to a rotating wheel.
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


def main():
    N = 200
    h = 0.01
    T = 5

    t = np.linspace(0, T, N)

    sol = mass_on_cart_system()
    indices = np.linspace(0, int(T / h) - 1, N, dtype=int)
    theta = sol[indices, 1] + np.pi
    x = sol[indices, 2]

    x21 = t * 0
    y21 = x
    trajectory2 = np.vstack((x21, y21)).T
    plane = Rectangle(trajectory2, length=2, width=2)

    x1 = np.cos(theta) * 0.5
    y1 = np.sin(theta) * 0.5
    trajectory1 = np.vstack((x1, y1)).T

    circle = Circle(trajectory1, radius=0.5)
    rod = ConnectiveRod(circle, plane, width=0.1)

    animate_objects(
        [plane, circle, rod],  # Rod is last
        interval=h * 1000,
        title="Object Animation",
        xlabel="X Position",
        ylabel="Y Position",
    )

    rod.plot_length()


if __name__ == "__main__":
    main()
