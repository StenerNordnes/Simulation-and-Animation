import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root_dir = os.path.dirname(parent_dir)  # Get the project root directory
sys.path.append(project_root_dir)  # Add the project root to sys.path

from animatableObjects import (  # noqa: E402
    Circle,
    ConnectiveRod,
    ConnectiveSpring,
    Rectangle,
    Triangle,
    animate_objects,
)
from numerical_solver import (  # noqa: E402
    NumericalSolver,
    rk4_tableau,
)


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


def profiling():
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


if __name__ == "__main__":
    N = 300
    h = 0.01
    T = 10 * np.pi

    t = np.linspace(0, T, N)

    profiling()

    initial_state = np.array([0.5, 1, np.pi / 6, 0.1])
    solver = NumericalSolver(rk4_tableau, spring_pendulum, profiling=True)
    solution = solver.solve(initial_state, 0, T, h)

    # Truncate the solution to the desired number of points
    indices = np.linspace(0, int(T / h) - 1, N, dtype=int)
    angle = solution[indices, 2] + np.pi
    z = solution[indices, 0]

    L = 5
    x2 = L * np.sin(angle)
    sim_y = z + L * np.cos(angle)
    trajectory2 = np.vstack((x2, sim_y)).T
    triangleRotationAngle = -angle
    triangle = Triangle(
        trajectory2, angle=triangleRotationAngle, side_length=2, show_trajectory=True
    )

    x1 = t * 0
    y1 = z
    trajectory1 = np.vstack((x1, y1)).T
    rectangle = Circle(trajectory1, radius=1)

    x21 = t * 0
    y21 = np.ones(t.shape) * 5
    trajectory2 = np.vstack((x21, y21)).T
    staticRectangle = Rectangle(trajectory2, length=5, width=1)

    rod = ConnectiveRod(rectangle, triangle, width=0.1)
    spring2 = ConnectiveSpring(
        staticRectangle,
        rectangle,
        num_coils=5,
        len_end_straight=1,
        len_start_straight=1,
    )

    animate_objects(
        [triangle, rectangle, staticRectangle, rod, spring2],  # Rod is last
        interval=40,
        title="Object Animation",
        xlabel="X Position",
        ylabel="Y Position",
    )
