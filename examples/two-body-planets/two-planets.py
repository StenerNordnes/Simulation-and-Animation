import numpy as np
from animation.trajectories import animate_trajectory
from solver.numerical_solver import NumericalSolver, rk4_tableau
from dataclasses import dataclass


@dataclass
class AttractionPoint:
    x: float
    y: float


CV_MODEL = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])


def gravityForce(
    attraction_pos: np.ndarray,
    planet_pos: np.ndarray,
    attraction_mass: float,
    planet_mass: float,
):
    r_vec = attraction_pos - planet_pos
    dist = np.linalg.norm(r_vec)
    direction = r_vec / dist
    attraction_force = attraction_mass * planet_mass / dist**2 * direction

    return attraction_force


def f(t, state: np.ndarray):
    x, y, xvel, yvel = state
    pos = state[:2]

    planet_mass = 10
    attraction_mass = 100
    a_point = np.array([0, 0])

    gravity_force = gravityForce(a_point, pos, attraction_mass, planet_mass)
    gravity_acceleration = gravity_force / planet_mass

    return np.array([xvel, yvel, gravity_acceleration[0], gravity_acceleration[1]])


def main():
    N = 200
    t = np.linspace(0, 2 * np.pi, N)
    h = t[1] - t[0]

    initial_state = np.array([5, 5, 1, 0])

    solver = NumericalSolver(rk4_tableau, f)
    pos1 = solver.solve(initial_state, t[0], t[-1], h)

    trajectory1 = pos1[:, :2]
    animate_trajectory([trajectory1], show_grid=True, legend=True)


if __name__ == "__main__":
    main()
