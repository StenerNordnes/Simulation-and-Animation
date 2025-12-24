# Animation Module Documentation

This module provides a comprehensive framework for creating 2D and 3D animations of physical systems. It is split into two main components:

1. **Object Animation (`objects.py`)**: For animating complex shapes (rectangles, circles, springs) and their interactions.
2. **Trajectory Animation (`trajectories.py`)**: For visualizing raw particle paths or data points in 2D and 3D space.

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
