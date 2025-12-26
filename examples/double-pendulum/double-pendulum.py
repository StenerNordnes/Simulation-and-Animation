import numpy as np
import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    project_root_dir = os.path.dirname(parent_dir)  # Get the project root directory
    sys.path.append(project_root_dir)  # Add the project root to sys.path
from animation.objects import animate_objects, Circle, ConnectiveRod
from solver.numerical_solver import NumericalSolver, rk4_tableau

save_path = os.path.abspath(__file__).replace(".py", ".gif")

NUM_STATES = 4
G = 9.81
L1 = 50
L2 = 50


def make_system(masses: list[int]):
    """
    :param masses: Masses attached to the double pendulum
    :type masses: list[int]
    """

    m1 = masses[0]
    m2 = masses[1]

    def f(t, state: np.ndarray):
        theta1, theta2, theta1_dot, theta2_dot = state

        theta1_dot_dot = (
            -G * (2 * m1 + m2) * np.sin(theta1)
            - m2 * G * np.sin(theta1 - 2 * theta2)
            - 2
            * np.sin(theta1 - theta2)
            * m2
            * (theta2_dot**2 * L2 + theta1_dot**2 * L1 * np.cos(theta1 - theta2))
        ) / (L1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
        theta2_dot_dot = (
            2
            * np.sin(theta1 - theta2)
            * (
                theta1_dot**2 * L1 * (m1 + m2)
                + G * (m1 + m2) * np.cos(theta1)
                + theta2_dot**2 * L2 * m2 * np.cos(theta1 - theta2)
            )
        ) / (L2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))

        return np.array([theta1_dot, theta2_dot, theta1_dot_dot, theta2_dot_dot])

    return f


def main():
    N = 200
    t = np.linspace(0, 50, N)
    h = t[1] - t[0]

    masses = [1, 1]
    initial_state = np.array([np.pi / 2, np.pi / 2, 0, 0])

    solver = NumericalSolver(rk4_tableau, make_system(masses))
    state = solver.solve(initial_state, t[0], t[-1], h)

    x1 = L1 * np.sin(state[:, 0])
    y1 = -L1 * np.cos(state[:, 0])
    trajectory1 = np.vstack((x1, y1)).T
    circle1 = Circle(trajectory1, 2)

    x2 = x1 + L2 * np.sin(state[:, 1])
    y2 = y1 - L1 * np.cos(state[:, 1])
    trajectory2 = np.vstack((x2, y2)).T
    circle2 = Circle(trajectory2, 2, show_trajectory=True)

    origin = Circle(trajectory1 * 0, 0.1, show_trajectory=False)

    rod1 = ConnectiveRod(origin, circle1, width=1)
    rod2 = ConnectiveRod(circle1, circle2, width=1)

    animate_objects(
        [circle1, circle2, rod1, rod2],
        legend=False,
        show_grid=False,
    )


if __name__ == "__main__":
    main()
