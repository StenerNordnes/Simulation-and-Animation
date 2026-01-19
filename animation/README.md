# Animation Module Documentation

This module provides a comprehensive framework for creating 2D and 3D animations of physical systems. It is split into three main components:

1. **Object Animation (`objects.py`)**: For animating complex shapes (rectangles, circles, springs) and their interactions.
2. **Trajectory Animation (`trajectories.py`)**: For visualizing raw particle paths or data points in 2D and 3D space.
3. **Surface Animation (`surface.py`)**: For animating 3D surfaces over time, ideal for visualizing PDEs and field simulations.

---

## 1. Animatable Objects (`objects.py`)

This submodule defines a system of classes representing physical objects that can be animated over time. It handles the drawing, updating, and synchronization of these objects.

### Core Function: `animate_objects`

The primary entry point for animating a collection of objects.

```python
def animate_objects(
    objects: list[AnimatableObject],
    interval: float = 20,
    title: str = "Animation",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    save_path: str | None = None,
    legend: bool = False
)
```

- **objects**: A list of `AnimatableObject` instances to render. Order matters for layering (last item is drawn on top).
- **interval**: Delay between frames in milliseconds.
- **save_path**: If provided, saves the animation to a file (e.g., `.mp4`, `.gif`).

### Classes

#### `AnimatableObject` (Abstract Base Class)

The parent class for all animated entities.

- **`trajectory`**: A NumPy array of shape $(N, 2)$ representing $(x, y)$ positions over time.
- **`show_trajectory`**: Boolean flag to draw a trace line of the object's path.

#### Geometric Shapes

| Class           | Constructor Parameters                            | Description                                                    |
| :-------------- | :------------------------------------------------ | :------------------------------------------------------------- |
| **`Rectangle`** | `trajectory`, `angles` (array), `length`, `width` | A rotating rectangle. `angles` controls orientation per frame. |
| **`Triangle`**  | `trajectory`, `side_length`, `angle` (array)      | A rotating equilateral triangle.                               |
| **`Circle`**    | `trajectory`, `radius`                            | A simple circle moving along a path.                           |

#### Connectors

These objects do not necessarily have their own independent trajectory but connect two other `AnimatableObject` instances.

**`ConnectiveRod`**
Draws a rigid rod between two objects.

```python
ConnectiveRod(object1, object2, width=2, show_trajectory=False)
```

**`ConnectiveSpring`**
Draws a dynamic spring connecting two objects. It stretches and compresses automatically.

```python
ConnectiveSpring(
    object1,
    object2,
    num_coils=10,
    amplitude=1.0,
    len_start_straight=0.5,
    len_end_straight=0.5
)
```

---

## 2. Trajectory Animation (`trajectories.py`)

This submodule is designed for quick visualization of data points or particle simulations without the overhead of defining complex shapes.

### 2D Animation: `animate_trajectory`

Animates one or more points moving in 2D space.

```python
def animate_trajectory(
    trajectories: list[np.ndarray] | np.ndarray,
    interval: int = 20,
    title: str = "Trajectories Animation",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    save_path: str | None = None,
    legend: bool = True,
    show_grid: bool = True,
    trajectory_labels: list[str] | None = None,
    trajectory_colors: list[str] | None = None
)
```

- **trajectories**: A list of $(N, 2)$ NumPy arrays. Each array is a separate path.
- **trajectory_colors**: Custom colors for each path. Defaults to a standard cycle.

### 3D Animation: `animate_trajectory_3d`

Animates a single point moving in 3D space.

```python
def animate_trajectory_3d(
    trajectory: np.ndarray,
    interval: int = 20,
    title: str = "3D Dot Trajectory Animation",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    zlabel: str = "Z-axis",
    save_path: str | None = None
)
```

- **trajectory**: A NumPy array of shape $(N, 3)$ representing $(x, y, z)$ coordinates.

---

## 3. Surface Animation (`surface.py`)

This submodule provides functionality for animating 3D surfaces that evolve over time. It is particularly useful for visualizing solutions to partial differential equations (PDEs), heat diffusion, wave propagation, and other field-based simulations.

### Function: `animate_surface`

Animates a 3D surface based on time-series data.

```python
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
    show_grid: bool = True
)
```

**Parameters:**

- **data**: A 3D NumPy array of shape $(T, N, M)$ where $T$ is the number of time steps, $N$ is the grid height, and $M$ is the grid width.
- **x_grid, y_grid**: Optional 2D arrays for custom X and Y coordinate grids. If `None`, defaults to integer indices.
- **interval**: Delay between frames in milliseconds.
- **zlim**: Fixed Z-axis limits as a tuple `(min, max)`. If `None`, automatically inferred from data.
- **cmap**: Colormap name for the surface (e.g., `"viridis"`, `"plasma"`, `"coolwarm"`).
- **save_path**: If provided, saves the animation to a file (e.g., `.mp4`, `.gif`).
- **show_grid**: Boolean flag to show/hide the grid lines.

---

## Usage Examples

### Example 1: Simple Pendulum (Object Animation)

```python
import numpy as np
from animation.objects import Circle, ConnectiveRod, animate_objects

# 1. Generate Data
t = np.linspace(0, 10, 100)
r = 50
x = r * np.sin(t)
y = -r * np.cos(t)
trajectory = np.vstack((x, y)).T

# 2. Create Objects
# A fixed point (pivot)
pivot_traj = np.zeros((100, 2))
pivot = Circle(pivot_traj, radius=0.1)

# The pendulum bob
bob = Circle(trajectory, radius=4, show_trajectory=True)

# The rod connecting them
rod = ConnectiveRod(pivot, bob, width=1)

# 3. Animate
animate_objects([pivot, bob, rod], title="Simple Pendulum")
```

### Example 2: Particle Trace (Trajectory Animation)

```python
import numpy as np
from animation.trajectories import animate_trajectory

# Generate a spiral path
theta = np.linspace(0, 4*np.pi, 200)
r = theta
x = r * np.cos(theta)
y = r * np.sin(theta)
path = np.vstack((x, y)).T

animate_trajectory([path], title="Spiral", trajectory_colors=['red'])
```

### Example 3: Heat Diffusion (Surface Animation)

```python
import numpy as np
from animation.surface import animate_surface

# Simulate heat diffusion on a 2D grid
T, N, M = 100, 50, 50
data = np.zeros((T, N, M))
for t in range(T):
    x = np.linspace(-5, 5, M)
    y = np.linspace(-5, 5, N)
    X, Y = np.meshgrid(x, y)
    data[t] = np.exp(-0.1*t) * np.exp(-(X**2 + Y**2) / (4 + 0.1*t))

animate_surface(data, title="Heat Diffusion", cmap="hot")
```
