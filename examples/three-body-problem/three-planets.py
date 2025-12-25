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
from dataclasses import dataclass


@dataclass
class Planet:
    mass: float
    initial_state: list


G = 6.6743e-11
SOFTENING = 1e-5  # Prevents division by zero collisions
save_path = os.path.abspath(__file__).replace(".py", ".gif")
NUM_STATES = 4
NUM_DIM = 2


def get_pos(index: int):
    return slice(index * NUM_STATES, index * NUM_STATES + NUM_DIM)


def get_vel(index: int):
    return slice(index * NUM_STATES + NUM_DIM, index * NUM_STATES + NUM_STATES)


def get_state(index: int):
    return slice(index * NUM_STATES, index * NUM_STATES + NUM_STATES)


def gravityForce(
    attraction_pos: np.ndarray,
    planet_pos: np.ndarray,
    attraction_mass: float,
    planet_mass: float,
):
    r_vec = attraction_pos - planet_pos
    dist = np.linalg.norm(r_vec)
    direction = r_vec / dist
    attraction_force = (
        G * attraction_mass * planet_mass / (dist**2 + SOFTENING**2) * direction
    )

    return attraction_force


def make_system(planet_masses: np.ndarray):
    def f(t, state: np.ndarray):
        state_dot = np.zeros(len(planet_masses) * NUM_STATES)

        for i, curr_mass in enumerate(planet_masses):
            total_accel = np.zeros(NUM_DIM)

            for j, attr_mass in enumerate(planet_masses):
                if i == j:
                    continue

                curr_pos = state[get_pos(i)]
                attr_pos = state[get_pos(j)]

                gravity_force = gravityForce(attr_pos, curr_pos, attr_mass, curr_mass)
                gravity_accel: np.ndarray = gravity_force / curr_mass

                total_accel += gravity_accel

            state_dot[get_state(i)] = np.concatenate([state[get_vel(i)], total_accel])

        return state_dot

    return f


def main():
    N = 200
    t = np.linspace(0, 2, N)
    h = t[1] - t[0]

    pos_x = 0.97000436
    pos_y = -0.24308753
    vel_x = 0.46620368
    vel_y = 0.43236573

    planets = [
        Planet(100e8, [pos_x, pos_y, -vel_x / 2, -vel_y / 2]),
        Planet(100e8, [-pos_x, -pos_y, -vel_x / 2, -vel_y / 2]),
        Planet(100e8, [0, 0, vel_x, vel_y]),
    ]

    initial_state = np.array([pl.initial_state for pl in planets]).flatten()
    masses = np.array([pl.mass for pl in planets])

    solver = NumericalSolver(rk4_tableau, make_system(masses))
    state = solver.solve(initial_state, t[0], t[-1], h)

    trajectories = [state[:, get_pos(i)] for i in range(len(planets))]

    animate_trajectory(trajectories, show_grid=False, legend=False)


if __name__ == "__main__":
    main()
