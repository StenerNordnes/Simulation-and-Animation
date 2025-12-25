import math
from abc import ABC, abstractmethod

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.path as mpath  # Import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from utils import Rotation2D


class AnimatableObject(ABC):
    def __init__(self, trajectory: np.ndarray, show_trajectory: bool = False):
        if not isinstance(trajectory, np.ndarray):
            raise TypeError("trajectory must be a NumPy array.")
        # Allow for objects without independent trajectories (like ConnectiveRod)
        if trajectory is not None:
            if trajectory.ndim != 2 or trajectory.shape[1] != 2:
                raise ValueError("trajectory must have shape (N, 2).")
            if len(trajectory) == 0:
                raise ValueError("Trajectory cannot be empty.")
            self.x, self.y = trajectory[0, :]
        else:
            # Objects like ConnectiveRod might not have their own base trajectory
            self.x, self.y = 0, 0  # Default or calculate based on connected objects

        self.trajectory = trajectory
        self.patch: patches.Patch | None = None  # Generic patch attribute
        self.show_trajectory = show_trajectory

    def update(self, frame_index: int, ax: plt.Axes):  # type: ignore
        """
        Updates the object's state based on the frame index and updates the patch.
        Args:
            frame_index (int): The index of the current frame in the trajectory.
            ax (plt.Axes): The axes object to draw/update on.
        """
        # Update position if the object has its own trajectory
        if self.trajectory is not None:
            if frame_index < 0 or frame_index >= len(self.trajectory):
                raise IndexError("frame_index out of bounds for the trajectory.")
            self.x, self.y = self.trajectory[frame_index, :]

        # Call the unified method to handle patch creation and updates
        self._update_patch(ax, frame_index)

    @abstractmethod
    def _update_patch(
        self,
        ax: plt.Axes,  # type: ignore
        frame_index: int,
    ):  # Added ax and frame_index parameters
        """
        Abstract method to create (if needed) and update the object's patch
        based on the current state (self.x, self.y) and potentially frame_index.
        This method is responsible for adding the patch to the axes initially.
        Args:
            ax (matplotlib.axes.Axes): The axis to draw on.
            frame_index (int): The current frame index, useful for objects
                               whose state depends on others (like ConnectiveRod).
        """
        pass

    @abstractmethod
    def get_longest_offset(self) -> float:
        """
        Returns the longest offset from the object center
        """
        pass

    def get_coords(self) -> np.ndarray:
        return np.array([self.x, self.y])


class Rectangle(AnimatableObject):
    def __init__(
        self,
        trajectory: np.ndarray,
        angles: NDArray[np.floating] | None = None,
        length: float = 20,
        width: float = 10,
        show_trajectory: bool = False,
    ):
        super().__init__(trajectory, show_trajectory)
        self.length = length
        self.width = width
        self.angles: NDArray[np.floating] = (
            angles
            if isinstance(angles, np.ndarray)
            else np.zeros(len(trajectory), dtype=float)
        )
        # self.patch is inherited

    def _update_patch(self, ax: plt.Axes, frame_index: int):  # type: ignore
        """
        Creates or updates the rectangle patch's position.
        """
        center_coords = np.array([self.x, self.y])

        if self.patch is not None:
            self.patch.remove()

        current_angle = self.angles[frame_index]
        rot_matrix = Rotation2D(current_angle)

        v1 = center_coords + rot_matrix @ np.array([-self.length / 2, -self.width / 2])
        v2 = center_coords + rot_matrix @ np.array([-self.length / 2, self.width / 2])
        v3 = center_coords + rot_matrix @ np.array([self.length / 2, self.width / 2])
        v4 = center_coords + rot_matrix @ np.array([self.length / 2, -self.width / 2])

        vertices = [v1, v2, v3, v4]

        self.patch = patches.Polygon(
            vertices,
            closed=True,
            fc="blue",
            ec="black",
        )
        ax.add_patch(self.patch)

    def get_longest_offset(self) -> float:
        return math.sqrt(
            (self.width / 2) ** 2 + (self.length / 2) ** 2
        )  # Diagonal offset


class Triangle(AnimatableObject):
    def __init__(
        self,
        trajectory: np.ndarray,
        side_length: float,
        angle: np.ndarray | None = None,
        show_trajectory: bool = False,
    ):
        super().__init__(trajectory, show_trajectory)
        self.side_length = side_length
        self.height = side_length * np.sin(np.pi / 3)
        self.angles = (
            angle
            if isinstance(angle, np.ndarray)
            else np.zeros(len(trajectory), dtype=float)
        )
        # self.patch is inherited

    def _get_vertices(self, frame_index: int) -> np.ndarray:
        """Calculates vertices based on current self.x, self.y"""
        center_point = np.array([self.x, self.y])
        # Vertices relative to the center (0,0)
        current_angle = self.angles[frame_index]

        rot_mat = Rotation2D(current_angle)
        h, s = self.height, self.side_length
        v1_rel = rot_mat @ np.array([0, h * 2 / 3])
        v2_rel = rot_mat @ np.array([-s / 2, -h / 3])
        v3_rel = rot_mat @ np.array([s / 2, -h / 3])

        # Translate vertices by the center point
        vertices = np.array(
            [center_point + v1_rel, center_point + v2_rel, center_point + v3_rel]
        )
        return vertices

    def _update_patch(self, ax: plt.Axes, frame_index: int):  # type: ignore
        """
        Creates or updates the triangle patch's vertices.
        """
        current_vertices = self._get_vertices(frame_index)

        if self.patch is None:
            self.patch = patches.Polygon(
                current_vertices,
                closed=True,
                fc="gray",
                ec="black",
            )
            ax.add_patch(self.patch)  # Add patch to the axes
        elif isinstance(self.patch, patches.Polygon):
            self.patch.set_xy(current_vertices)  # Update vertices
        else:
            print("Warning: Patch is not a Polygon instance.")

    def get_longest_offset(self) -> float:
        # The longest offset from the center is to any vertex
        # e.g., distance to top vertex
        return self.height / 2


class Circle(AnimatableObject):
    def __init__(
        self,
        trajectory: np.ndarray,
        radius: float,
        show_trajectory: bool = False,
    ):
        super().__init__(trajectory, show_trajectory)
        self.radius = radius

    def _update_patch(self, ax: plt.Axes, frame_index: int):  # type: ignore
        """
        Creates or updates the triangle patch's vertices.
        """
        if self.patch is not None:
            self.patch.remove()

        self.patch = patches.CirclePolygon(
            (self.x, self.y),
            radius=self.radius,
            fc="gray",
            ec="black",
        )
        ax.add_patch(self.patch)  # Add patch to the axes

    def get_longest_offset(self) -> float:
        # The longest offset from the center is to any vertex
        # e.g., distance to top vertex
        return self.radius


class ConnectiveRod(AnimatableObject):
    def __init__(
        self,
        object1: AnimatableObject,
        object2: AnimatableObject,
        width: float = 2,
        show_trajectory: bool = False,
    ):
        trajectory = np.zeros(object1.trajectory.shape)
        super().__init__(trajectory, show_trajectory)
        self.width = width
        self.object1 = object1
        self.object2 = object2

        self.lengths = []

    def _make_rod(self) -> patches.Patch:
        diff_vec = self.object1.get_coords() - self.object2.get_coords()
        obj1_coords = self.object1.get_coords()

        offset_vec = Rotation2D(np.pi / 2) @ diff_vec
        self.x, self.y = obj1_coords - diff_vec / 2
        offset_vec = offset_vec * self.width / np.linalg.norm(offset_vec)

        v_1 = obj1_coords - offset_vec
        v_2 = obj1_coords + offset_vec

        v_3 = v_2 - diff_vec
        v_4 = v_1 - diff_vec

        vertices = [v_1, v_2, v_3, v_4]

        patch = patches.Polygon(
            vertices,
            closed=True,
            fc="green",
            ec="black",
        )

        return patch

    def _update_patch(self, ax: plt.Axes, frame_index: int):  # type: ignore
        """
        Creates or updates the connective rod patch (a rotated rectangle).
        """
        self._capture_length()
        if self.patch is not None:
            self.patch.remove()

        self.patch = self._make_rod()
        self.trajectory[frame_index, :] = np.array([self.x, self.y])
        ax.add_patch(self.patch)  # Add patch to the axes

    def get_longest_offset(self) -> float:
        return 0

    def _capture_length(self):
        """Calculates the length of the rod based on the distance between the two objects."""
        self.lengths.append(
            np.linalg.norm(self.object1.get_coords() - self.object2.get_coords())
        )

    def plot_length(self):
        """Plots the length of the rod over time."""
        if len(self.lengths) == 0:
            print("No lengths captured yet.")
            return

        # Create a new figure for the length plot
        fig, ax = plt.subplots()
        ax.plot(self.lengths, label="Rod Length")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Length")
        ax.set_title("Rod Length Over Time")
        ax.legend()
        plt.show()


class ConnectiveSpring(AnimatableObject):
    def __init__(
        self,
        object1: AnimatableObject,
        object2: AnimatableObject,
        num_coils: int = 10,  # Number of full zigzag cycles
        amplitude: float = 1.0,  # Amplitude of the zigzag perpendicular to the spring axis
        len_start_straight: float = 0.5,  # Length of straight section at object1 end
        len_end_straight: float = 0.5,  # Length of straight section at object2 end
        line_width: float = 1.0,  # Thickness of the spring line
        color: str = "green",  # Color of the spring
        show_trajectory: bool = False,
    ):
        # Spring doesn't have its own independent trajectory, calculated based on objects
        # We still need a placeholder trajectory for the AnimatableObject structure
        placeholder_trajectory = (
            np.zeros_like(object1.trajectory)
            if object1.trajectory is not None
            else np.zeros((1, 2))
        )
        if object1.trajectory is not None and object2.trajectory is not None:
            if len(object1.trajectory) != len(object2.trajectory):
                raise ValueError(
                    "Connected objects must have trajectories of the same length."
                )
            placeholder_trajectory = np.zeros_like(object1.trajectory)

        super().__init__(placeholder_trajectory, show_trajectory)  # Pass None initially

        self.object1 = object1
        self.object2 = object2
        self.num_coils = max(1, num_coils)  # Ensure at least one coil
        self.amplitude = amplitude
        self.len_start_straight = len_start_straight
        self.len_end_straight = len_end_straight
        self.line_width = line_width
        self.color = color

        # Patch will be a PathPatch
        # self.patch is inherited from AnimatableObject (patches.Patch | None)
        # A PathPatch is a Patch, so this is fine.

    def _make_spring_path(self) -> mpath.Path:
        """Creates the Matplotlib Path for the spring shape."""
        p1 = self.object1.get_coords()
        p2 = self.object2.get_coords()
        diff_vec = p1 - p2

        L_spring = np.linalg.norm(diff_vec)
        if L_spring < 1e-6:  # Avoid division by zero if objects coincide
            # Return a path with just two coincident points
            return mpath.Path([p1, p1], [mpath.Path.MOVETO, mpath.Path.LINETO])

        u_dir = diff_vec / L_spring  # Unit vector from p2 towards p1
        # u_perp = RotationMatrix(np.pi / 2) @ u_dir # Unit vector perpendicular
        u_perp = np.array([-u_dir[1], u_dir[0]])  # More direct calculation

        # Calculate length available for coils
        L_coils = L_spring - self.len_start_straight - self.len_end_straight

        vertices = [p1]  # Start at object1 center
        codes = [mpath.Path.MOVETO]

        # Start straight section
        p_start_coil = p1 - self.len_start_straight * u_dir
        if self.len_start_straight > 0:
            vertices.append(p_start_coil)
            codes.append(mpath.Path.LINETO)

        # Coils section
        if L_coils > 1e-6 and self.num_coils > 0:
            # Length of one full zigzag along the spring axis
            coil_segment_len = L_coils / self.num_coils
            # Number of points per half-coil (straight segment in zigzag)
            # points_per_coil = 4

            current_pos = p_start_coil
            for i in range(self.num_coils):
                # Points within one coil cycle relative to current_pos
                # Point 1 (Peak 1)
                p_coil1 = (
                    current_pos
                    - (coil_segment_len / 4) * u_dir
                    + self.amplitude * u_perp
                )
                # Point 2 (Peak 2)
                p_coil2 = (
                    current_pos
                    - (3 * coil_segment_len / 4) * u_dir
                    - self.amplitude * u_perp
                )
                # Point 3 (End of this coil segment on axis)
                p_coil_end = current_pos - coil_segment_len * u_dir

                vertices.extend([p_coil1, p_coil2, p_coil_end])
                codes.extend([mpath.Path.LINETO] * 3)
                current_pos = p_coil_end  # Update position for next coil

            # Ensure the last point of coils connects smoothly to the end straight section
            p_end_coil = p_start_coil - L_coils * u_dir
            # If the last calculated point isn't exactly p_end_coil, adjust
            if np.linalg.norm(vertices[-1] - p_end_coil) > 1e-6:
                vertices[-1] = p_end_coil  # Force it to the correct position

        else:
            # If no space for coils, just draw a straight line for the coil part
            p_end_coil = p_start_coil  # Effectively zero coil length
            # vertices.append(p_end_coil) # This point might already be there
            # codes.append(mpath.Path.LINETO)

        # End straight section
        # The target is p2. p_end_coil should be L_end_straight away from p2.
        if self.len_end_straight > 0:
            # Check if the last vertex added is already p_end_coil
            if len(vertices) > 1 and np.linalg.norm(vertices[-1] - p_end_coil) < 1e-6:
                vertices.append(p2)  # Add the final point p2
                codes.append(mpath.Path.LINETO)
            elif len(vertices) == 1:  # Only p1 exists
                vertices.append(p2)  # Draw straight line p1 to p2
                codes.append(mpath.Path.LINETO)
            else:  # Should have coil points ending at p_end_coil
                # It means p_end_coil wasn't added explicitly if L_coils was small
                # Add it now if it's different from the last vertex
                if np.linalg.norm(vertices[-1] - p_end_coil) > 1e-6:
                    vertices.append(p_end_coil)
                    codes.append(mpath.Path.LINETO)
                vertices.append(p2)
                codes.append(mpath.Path.LINETO)

        elif len(vertices) > 0 and np.linalg.norm(vertices[-1] - p2) > 1e-6:
            # If no end straight section, ensure the last vertex is p2
            vertices.append(p2)
            codes.append(mpath.Path.LINETO)

        # Calculate the spring's midpoint for the trajectory attribute (optional)
        self.x, self.y = (p1 + p2) / 2

        return mpath.Path(np.array(vertices), codes)

    def _update_patch(self, ax: plt.Axes, frame_index: int):  # type: ignore
        """
        Creates or updates the connective spring patch using PathPatch.
        """
        new_path = self._make_spring_path()

        if self.patch is None:
            # Create patch for the first time
            self.patch = patches.PathPatch(
                new_path,
                fill=False,  # Don't fill the shape
                ec=self.color,  # Edge color
                lw=self.line_width,  # Line width
            )
            ax.add_patch(self.patch)
            # Update placeholder trajectory (optional, represents midpoint)
            if self.trajectory is not None and frame_index < len(self.trajectory):
                self.trajectory[frame_index, :] = np.array([self.x, self.y])

        elif isinstance(self.patch, patches.PathPatch):
            # Update existing patch by setting its path
            self.patch.set_path(new_path)
            # Update placeholder trajectory (optional)
            if self.trajectory is not None and frame_index < len(self.trajectory):
                self.trajectory[frame_index, :] = np.array([self.x, self.y])
        else:
            print("Warning: Patch is not a PathPatch instance.")

    def get_longest_offset(self) -> float:
        # The longest offset is roughly half the length + the amplitude,
        # but it's complex. Returning 0 as it's not a rigid body.
        # Or estimate based on amplitude.
        return self.amplitude  # A rough estimate for axis limits


def animate_objects(
    objects: list[AnimatableObject],
    interval: float = 20,
    title: str = "Animation",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    save_path: str | None = None,
    legend: bool = False,
    show_grid: bool = True,
):
    """
    Animates a list of objects based on their trajectories.
    Ensures objects are updated in order (important for dependencies like ConnectiveRod).
    Plots path lines based on the object's actual (x, y) position each frame.
    """
    fig, ax = plt.subplots()

    # Determine plot limits based on all objects' potential movement
    # Need to consider the full trajectories for limits
    all_x_traj = []
    all_y_traj = []
    max_offset = 0
    num_frames = 0
    for obj in objects:
        if obj.trajectory is not None:
            all_x_traj.append(obj.trajectory[:, 0])
            all_y_traj.append(obj.trajectory[:, 1])
            if len(obj.trajectory) > num_frames:
                num_frames = len(obj.trajectory)
        # Estimate max offset - might need refinement for dynamic shapes like Rod
        try:
            # Calculate offset based on initial state for estimation
            initial_offset = obj.get_longest_offset()
            max_offset = max(max_offset, initial_offset)
        except Exception:  # Catch potential errors during initial calculation
            pass  # Use default padding

    # If num_frames is still 0 (e.g., only objects without trajectories),
    # try to infer from connected objects if possible, or default.
    if num_frames == 0 and objects:
        # Check if any object is a ConnectiveRod to get frame count from its connections
        for obj in objects:
            if isinstance(obj, ConnectiveRod):
                if obj.object1.trajectory is not None:
                    num_frames = max(num_frames, len(obj.object1.trajectory))
                if obj.object2.trajectory is not None:
                    num_frames = max(num_frames, len(obj.object2.trajectory))
        if num_frames == 0:  # Still zero? Default to 1 frame.
            num_frames = 1

    if not all_x_traj or not all_y_traj:
        # Handle case where no objects have explicit trajectories for limits
        print(
            "Warning: No explicit trajectories found to determine plot limits automatically."
        )
        # Try to estimate limits from initial positions if possible
        initial_xs = [obj.x for obj in objects]
        initial_ys = [obj.y for obj in objects]
        if initial_xs and initial_ys:
            min_x, max_x = min(initial_xs), max(initial_xs)
            min_y, max_y = min(initial_ys), max(initial_ys)
        else:
            min_x, max_x, min_y, max_y = -10, 10, -10, 10  # Fallback default
        if not num_frames:
            num_frames = 1  # At least one frame
    else:
        all_x_flat = np.concatenate(all_x_traj)
        all_y_flat = np.concatenate(all_y_traj)
        min_x, max_x = np.min(all_x_flat), np.max(all_x_flat)
        min_y, max_y = np.min(all_y_flat), np.max(all_y_flat)

    padding = 5 + max_offset  # Use calculated max offset

    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(show_grid)

    # Lines to trace paths
    lines = [
        ax.plot([], [], "-", lw=1, label=f"Object {i} Path")[0]
        for i in range(len(objects))
    ]

    # Store path history based on actual (x, y) updates
    path_histories = [[] for _ in objects]

    if legend:
        # Create legend only for artists with labels
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles=handles, labels=labels)

    # Initialization function: plot the background of each frame
    def init():
        artists = []
        for i, line in enumerate(lines):
            line.set_data([], [])
            artists.append(line)
            path_histories[i] = []  # Clear history on init
        # Patches will be created/added in the first update call
        return artists

    # Animation function: this is called sequentially
    def update(frame: int):
        artists_to_update = []
        # Update objects IN ORDER - important for dependencies
        for i, obj in enumerate(objects):
            # Update object state (obj.x, obj.y) and its patch
            obj.update(frame, ax)  # Pass ax here

            # Store the updated position for path tracing
            path_histories[i].append((obj.x, obj.y))

            # Update path trace using the history
            line = lines[i]
            # Convert list of tuples to separate x and y lists/arrays
            if path_histories[i] and obj.show_trajectory:
                hist_x, hist_y = zip(*path_histories[i])
                line.set_data(hist_x, hist_y)
                artists_to_update.append(line)

            # Add the object's patch to the list of artists to update
            if obj.patch:
                artists_to_update.append(obj.patch)

        # If using blit=True, MUST return all modified artists
        # If using blit=False, returning is less critical but good practice
        return artists_to_update

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,  # Use determined number of frames
        init_func=init,
        interval=interval,
        blit=False,  # Set blit=False is MUCH simpler with patches being added/modified
        repeat=False,
    )

    # ... (rest of save/show logic remains the same) ...

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
