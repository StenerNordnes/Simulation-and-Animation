import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Default colors for trajectories if not specified
DEFAULT_TRAJECTORY_COLORS = [
    "blue",
    "green",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "purple",
    "orange",
    "brown",
]


def animate_trajectory(
    trajectories: list[np.ndarray] | np.ndarray,
    interval: int = 20,
    title: str = "Trajectories Animation",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    save_path: str | None = None,
    legend: bool = True,
    show_grid=True,
    trajectory_labels: list[str] | None = None,
    trajectory_colors: list[str] | None = None,
):
    """
    Animates the trajectories of multiple dots based on a list of NumPy arrays.

    Args:
        trajectories (list[np.ndarray]): A list of NumPy arrays. Each array
                                         should have shape (N_i, 2) where N_i is
                                         the number of time steps for trajectory i,
                                         and each row represents the (x, y) coordinates.
        interval (int): Delay between frames in milliseconds. Defaults to 20.
        title (str): The title of the plot. Defaults to 'Trajectories Animation'.
        xlabel (str): The label for the x-axis. Defaults to 'X-axis'.
        ylabel (str): The label for the y-axis. Defaults to 'Y-axis'.
        save_path (str, optional): If provided, saves the animation to this
                                   file path (e.g., 'animation.gif' or 'animation.mp4').
                                   Requires appropriate writers installed. Defaults to None.
        legend (bool): Whether to display the legend. Defaults to True.
        trajectory_labels (list[str], optional): A list of labels for each trajectory.
                                                 If None, default labels are used.
        trajectory_colors (list[str], optional): A list of colors for each trajectory.
                                                 If None, default colors are used.
    """
    if not isinstance(trajectories, list):
        # raise TypeError("trajectories must be a list of NumPy arrays.")
        trajectories = [trajectories]
    if not trajectories:
        print("Warning: No trajectories provided. Nothing to animate.")
        return

    valid_trajectories = []
    for i, traj in enumerate(trajectories):
        if not isinstance(traj, np.ndarray):
            print(f"Warning: Trajectory {i} is not a NumPy array. Skipping.")
            continue
        if traj.ndim != 2 or traj.shape[1] != 2:
            print(
                f"Warning: Trajectory {i} has incorrect shape {traj.shape}. Expected (N, 2). Skipping."
            )
            continue
        if len(traj) == 0:
            print(f"Warning: Trajectory {i} is empty. Skipping.")
            continue
        valid_trajectories.append(traj)

    if not valid_trajectories:
        print("Warning: No valid trajectories to animate after filtering.")
        return
    trajectories = valid_trajectories

    max_frames = 0
    for traj in trajectories:
        max_frames = max(max_frames, len(traj))

    if max_frames == 0:
        print("Warning: All trajectories are effectively empty. Nothing to animate.")
        return

    fig, ax = plt.subplots()

    # Determine plot limits with a small margin
    all_points = np.vstack(trajectories)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    range_vals = max_vals - min_vals
    margin = range_vals * 0.1  # 10% margin
    margin[range_vals == 0] = 1.0  # Handle cases where range is zero

    ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
    ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(show_grid)

    dot_artists = []
    line_artists = []

    colors_to_use = (
        trajectory_colors if trajectory_colors else DEFAULT_TRAJECTORY_COLORS
    )

    for i, traj in enumerate(trajectories):
        color = colors_to_use[i % len(colors_to_use)]
        label = (
            trajectory_labels[i]
            if trajectory_labels and i < len(trajectory_labels)
            else f"Trajectory {i + 1}"
        )

        (dot,) = ax.plot([], [], "o", ms=6, color=color)  # Dot for current position
        (line,) = ax.plot(
            [], [], "-", lw=1, color=color, label=label
        )  # Line for path taken
        dot_artists.append(dot)
        line_artists.append(line)

    if legend:
        ax.legend()

    all_artists_tuple = tuple(dot_artists + line_artists)

    def init():
        for dot in dot_artists:
            dot.set_data([], [])
        for line in line_artists:
            line.set_data([], [])
        return all_artists_tuple

    def update(frame):
        for i, traj in enumerate(trajectories):
            if frame < len(traj):
                current_pos = traj[frame]
                dot_artists[i].set_data([current_pos[0]], [current_pos[1]])

                path_so_far = traj[: frame + 1]
                line_artists[i].set_data(path_so_far[:, 0], path_so_far[:, 1])
            # If frame >= len(traj), the trajectory has ended; its artists remain as they were.
        return all_artists_tuple

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max_frames,
        init_func=init,
        interval=interval,
        blit=True,
        repeat=False,
    )

    if save_path:
        try:
            print(f"Saving animation to {save_path}...")
            writer_name = (
                "ffmpeg"
                if save_path.endswith(".mp4")
                else "pillow" if save_path.endswith(".gif") else None
            )
            if writer_name:
                ani.save(save_path, writer=writer_name, dpi=150)
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


def animate_trajectory_3d(
    trajectory: np.ndarray,
    interval: int = 20,
    title: str = "3D Dot Trajectory Animation",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    zlabel: str = "Z-axis",
    save_path: str | None = None,
    legend: bool = True,
):
    """
    Animates the 3D trajectory of a dot based on a NumPy array.

    Args:
        trajectory (np.ndarray): A NumPy array of shape (N, 3) where N is
                                 the number of time steps, and each row
                                 represents the (x, y, z) coordinates of the dot.
        interval (int): Delay between frames in milliseconds. Defaults to 20.
        title (str): The title of the plot. Defaults to '3D Dot Trajectory Animation'.
        xlabel (str): The label for the x-axis. Defaults to 'X-axis'.
        ylabel (str): The label for the y-axis. Defaults to 'Y-axis'.
        zlabel (str): The label for the z-axis. Defaults to 'Z-axis'.
        save_path (str, optional): If provided, saves the animation to this
                                   file path (e.g., 'animation.gif' or 'animation.mp4').
                                   Requires appropriate writers installed (e.g., Pillow for GIF,
                                   ffmpeg for MP4). Defaults to None (displays animation).
        legend (bool): Whether to display the legend. Defaults to True.
    """
    if not isinstance(trajectory, np.ndarray):
        raise TypeError("trajectory must be a NumPy array.")
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory must have shape (N, 3).")
    if len(trajectory) == 0:
        print("Warning: Trajectory is empty. Nothing to animate.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Determine plot limits with a small margin
    min_vals = np.min(trajectory, axis=0)
    max_vals = np.max(trajectory, axis=0)
    range_vals = max_vals - min_vals
    margin = range_vals * 0.1  # 10% margin
    # Handle cases where range is zero
    margin[range_vals == 0] = 1.0

    ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
    ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
    ax.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])  # type: ignore

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)  # type: ignore
    ax.set_title(title)
    # Optional: Set aspect ratio if needed, can be tricky in 3D
    # ax.set_box_aspect([np.ptp(trajectory[:,0]), np.ptp(trajectory[:,1]), np.ptp(trajectory[:,2])]) # Aspect ratio based on data range
    # ax.set_box_aspect([1,1,1]) # Equal aspect ratio

    ax.grid(True)

    # Initialize the plot elements: the dot and the path trace
    (dot,) = ax.plot([], [], [], "bo", ms=6, label="Current Position")  # Blue dot
    (line,) = ax.plot([], [], [], "r-", lw=1, label="Path Taken")  # Red line for path

    if legend:
        ax.legend()

    # Initialization function: plot the background of each frame
    def init():
        dot.set_data_3d([], [], [])  # type: ignore
        line.set_data_3d([], [], [])  # type: ignore
        return dot, line

    # Animation function: this is called sequentially
    def update(frame):
        # Update the dot's position
        current_pos = trajectory[frame]
        dot.set_data_3d([current_pos[0]], [current_pos[1]], [current_pos[2]])  # type: ignore

        # Update the path trace up to the current frame
        path_so_far = trajectory[: frame + 1]
        line.set_data_3d(path_so_far[:, 0], path_so_far[:, 1], path_so_far[:, 2])  # type: ignore

        return dot, line

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        init_func=init,
        interval=interval,
        blit=True,  # Blitting might have issues in some 3D backends, set to False if needed
        repeat=False,
    )

    # Save or show the animation
    if save_path:
        try:
            print(f"Saving animation to {save_path}...")
            # You might need to install ffmpeg or pillow
            # For GIF: writer='pillow' (might be slow/large for 3D)
            # For MP4: writer='ffmpeg' (recommended for 3D)
            writer_name = (
                "ffmpeg"
                if save_path.endswith(".mp4")
                else "pillow" if save_path.endswith(".gif") else None
            )
            if writer_name:
                # Lower dpi might be needed for performance with 3D
                ani.save(save_path, writer=writer_name, dpi=100)
                print("Save complete.")
            else:
                print(
                    f"Warning: Unknown file extension for saving. Could not save to {save_path}."
                )
                plt.show()  # Fallback to showing
        except Exception as e:
            print(f"Error saving animation: {e}")
            print(
                "Make sure you have the required writer installed (e.g., 'pip install Pillow' for GIF, or install ffmpeg)."
            )
            print("Displaying animation instead.")
            plt.show()  # Fallback to showing if saving fails
    else:
        plt.show()


# Example Usage (optional, for testing)
if __name__ == "__main__":
    # Generate some sample 3D trajectory data (e.g., a helix)
    t = np.linspace(0, 10 * np.pi, 500)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (5 * np.pi)
    trajectory_3d_data = np.vstack((x, y, z)).T

    # Animate the 3D trajectory
    animate_trajectory_3d(
        trajectory_3d_data,
        interval=10,
        title="Helix Trajectory",
        xlabel="X Position",
        ylabel="Y Position",
        zlabel="Z Position",
        # save_path='helix_animation.mp4' # Uncomment to save
    )

    t_2d = np.linspace(0, 2 * np.pi, 100)
    x_2d = np.cos(t_2d)
    y_2d = np.sin(t_2d) * 0.5  # Ellipse
    trajectory_2d_data = np.vstack((x_2d, y_2d)).T
    animate_trajectory([trajectory_2d_data], title="Ellipse Trajectory")
