import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
from matplotlib.artist import Artist


def animate_surface(
    data: np.ndarray,
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
    interval: int = 50,
    title: str = "Surface Animation",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    zlabel: str = "Z-axis",
    zlim: tuple[float, float] | None = None,
    save_path: str | None = None,
    cmap: str = "viridis",
    show_grid: bool = True,
):
    """
    Animates a 3D surface based on a 3D NumPy array (Time, Y, X).

    Args:
        data (np.ndarray): A 3D NumPy array of shape (T, N, M) where T is time steps.
        x_grid (np.ndarray, optional): 2D array for X coordinates. Defaults to np.arange(M).
        y_grid (np.ndarray, optional): 2D array for Y coordinates. Defaults to np.arange(N).
        interval (int): Delay between frames in milliseconds. Defaults to 50.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        zlabel (str): The label for the z-axis.
        zlim (tuple[float, float], optional): Fixed Z-axis limits (min, max). If None, inferred from data.
        save_path (str, optional): If provided, saves the animation to this file path.
        cmap (str): Colormap name. Defaults to "viridis".
        show_grid (bool): Whether to show the grid. Defaults to True.
    """
    # Validation
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a NumPy array.")
    if data.ndim != 3:
        raise ValueError("data must have shape (T, N, M).")

    num_frames, height, width = data.shape

    if x_grid is None or y_grid is None:
        x = np.arange(width)
        y = np.arange(height)
        x_grid, y_grid = np.meshgrid(x, y)

    # Determine Z limits if not provided
    if zlim is None:
        z_min, z_max = np.min(data), np.max(data)
        # Add some padding
        z_range = z_max - z_min
        if z_range == 0:
            z_range = 1.0
        zlim = (z_min - 0.1 * z_range, z_max + 0.1 * z_range)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(frame) -> Iterable[Artist]:
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_zlim(zlim)
        if show_grid:
            ax.grid(True)
        else:
            ax.grid(False)

        surf = ax.plot_surface(x_grid, y_grid, data[frame], cmap=cmap)

        return [surf]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval,
        blit=False,  # blit=False is typically required for 3D surface animations with ax.clear()
    )

    if save_path:
        try:
            print(f"Saving animation to {save_path}...")
            writer_name = (
                "ffmpeg"
                if save_path.endswith(".mp4")
                else "pillow"
                if save_path.endswith(".gif")
                else None
            )
            if writer_name:
                ani.save(save_path, writer=writer_name, dpi=100)
                print("Save complete.")
            else:
                print(
                    f"Warning: Unknown file extension for saving. Could not save to {save_path}."
                )
                plt.show()
        except Exception as e:
            print(f"Error saving animation: {e}")
            print(
                "Make sure you have the required writer installed (e.g., 'pip install Pillow' for GIF, or install ffmpeg)."
            )
            print("Displaying animation instead.")
            plt.show()
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage: Heat equation simulation (similar to user request)
    L = 20
    T = 300  # Reduced frames for quick test
    alpha = 4
    dx = 1
    dt = (dx**2) / (4 * alpha)
    gamma = (alpha * dt) / (dx**2)

    u = np.zeros((T, L, L))

    # Initial condition: Ramp
    for i in range(L):
        for j in range(L):
            u[0, i, j] = i * dx / L

    # Explicit finite difference method
    for k in range(0, T - 1):
        for i in range(1, L - 1):
            for j in range(1, L - 1):
                u[k + 1, i, j] = (
                    gamma
                    * (
                        u[k, i + 1, j]
                        + u[k, i - 1, j]
                        + u[k, i, j + 1]
                        + u[k, i, j - 1]
                        - 4 * u[k, i, j]
                    )
                    + u[k, i, j]
                )

    animate_surface(
        u,
        title="Heat Equation (Surface Animation)",
        zlabel="Temperature",
    )
