import numpy as np
import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    project_root_dir = os.path.dirname(parent_dir)  # Get the project root directory
    sys.path.append(project_root_dir)  # Add the project root to sys.path
from animation.surface import animate_surface
from solver.numerical_solver import NumericalSolver, rk4_tableau

save_path = os.path.abspath(__file__).replace(".py", ".gif")

M = 1
H_BAR = 1


def laplacian(grid: np.ndarray, dx) -> np.ndarray:
    # Shift grid to get neighbors
    up = np.roll(grid, -1, axis=0)
    down = np.roll(grid, 1, axis=0)
    left = np.roll(grid, -1, axis=1)
    right = np.roll(grid, 1, axis=1)
    center = grid

    # Finite difference formula: (f(x+h) + f(x-h) + f(y+h) + f(y-h) - 4f(x,y)) / h^2
    lap = (up + down + left + right - 4 * center) / (dx**2)
    return lap


def make_system(grid_size, dx):
    wall_thickness = 10
    wall_height = 5.0

    V = np.zeros((grid_size, grid_size))
    V[:wall_thickness, :] = wall_height  # Top
    V[-wall_thickness:, :] = wall_height  # Bottom
    V[:, :wall_thickness] = wall_height  # Left
    V[:, -wall_thickness:] = wall_height  # Right

    def f(t, psi_flattened: np.ndarray):
        mid_point = len(psi_flattened) // 2
        psi_real = psi_flattened[:mid_point].reshape((grid_size, grid_size))
        psi_imag = psi_flattened[mid_point:].reshape((grid_size, grid_size))

        psi = psi_real + 1j * psi_imag

        psi_dot = -1j * (-(H_BAR**2) / (2 * M) * laplacian(psi, dx) + V * psi)

        return np.concatenate([psi_dot.real.flatten(), psi_dot.imag.flatten()])

    return f


def get_initial_state(grid_size, dx):
    L = grid_size * dx

    x = np.linspace(0, L, grid_size)
    y = np.linspace(0, L, grid_size)
    X, Y = np.meshgrid(x, y)

    x0, y0 = L / 2, L / 2  # Start position
    kx, ky = 1.5, 0.5  # Momentum (Velocity)
    sigma = L / 15  # Width of the packet

    # Gaussian Envelope * Complex Plane Wave
    psi: np.ndarray = np.exp(
        -((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)
    ) * np.exp(1j * (kx * X + ky * Y))

    norm: np.float64 = np.sqrt(np.sum(np.abs(psi) ** 2) * dx * dx)
    return psi / norm


def main():
    grid_size = 100
    dx = 0.2

    dt = 0.01 * (dx**2)
    T_MAX = 10

    t = np.arange(0, T_MAX, dt)
    h = dt

    initial_state = get_initial_state(grid_size, dx)
    initial_state_split = np.concatenate(
        [initial_state.real.flatten(), initial_state.imag.flatten()]
    )

    solver = NumericalSolver(rk4_tableau, make_system(grid_size, dx))
    state = solver.solve(initial_state_split, t[0], t[-1], h)

    skip = 200
    mid = state.shape[1] // 2
    psi_real = state[::skip, :mid]
    psi_imag = state[::skip, mid:]

    psi_hist = psi_real + 1j * psi_imag

    probability_density = np.abs(psi_hist) ** 2
    probability_density = probability_density.reshape(
        (len(psi_hist), grid_size, grid_size)
    )

    animate_surface(
        probability_density,
        title="Schrödinger Equation (Probability Density)",
        zlabel=r"$|\Psi|^2$",
        zlim=(0, np.max(probability_density) * 0.8),
        interval=10,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
