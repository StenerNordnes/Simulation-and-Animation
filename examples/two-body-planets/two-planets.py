import numpy as np
import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    project_root_dir = os.path.dirname(parent_dir)  # Get the project root directory
    sys.path.append(project_root_dir)  # Add the project root to sys.path
from animation.trajectories import animate_trajectory
from solver.numerical_solver import NumericalSolver, rk4_tableau


# G = 6.6743e-11
G = 1
save_path = os.path.abspath(__file__).replace(".py", ".gif")


def gravityForce(
    attraction_pos: np.ndarray,
    planet_pos: np.ndarray,
    attraction_mass: float,
    planet_mass: float,
):
    r_vec = attraction_pos - planet_pos
    dist = np.linalg.norm(r_vec)
    direction = r_vec / dist
    attraction_force = G * attraction_mass * planet_mass / dist**2 * direction

    return attraction_force


def f(t, state: np.ndarray):
    x1, y1, xvel1, yvel1, x2, y2, xvel2, yvel2 = state
    pos1 = state[:2]
    pos2 = state[4:6]

    planet_mass1 = 10
    planet_mass2 = 1

    gravity_force1 = gravityForce(pos2, pos1, planet_mass2, planet_mass1)
    gravity1 = gravity_force1 / planet_mass1

    gravity_force2 = gravityForce(pos1, pos2, planet_mass1, planet_mass2)
    gravity2 = gravity_force2 / planet_mass2

    return np.array(
        [xvel1, yvel1, gravity1[0], gravity1[1], xvel2, yvel2, gravity2[0], gravity2[1]]
    )


def main():
    N = 100
    t = np.linspace(0, 2 * np.pi, N)
    h = t[1] - t[0]

    # x1, y1, xvel1, yvel1, x2, y2, xvel2, yvel2
    initial_state = np.array([5, 5, 1, 0, 8, 10, 0.3, 0])

    solver = NumericalSolver(rk4_tableau, f)
    state = solver.solve(initial_state, t[0], t[-1], h)

    trajectory1 = state[:, :2]
    trajectory2 = state[:, 4:6]
    animate_trajectory(
        [trajectory1, trajectory2], show_grid=False, legend=False, save_path=save_path
    )


if __name__ == "__main__":
    main()
