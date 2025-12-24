# Two-Body Planet Simulation

This example simulates the gravitational interaction between two bodies (planets) in 2D space using the `NumericalSolver` and visualizes the trajectories.

## Mathematical Model

The simulation is governed by Newton's Law of Universal Gravitation and Newton's Second Law of Motion.

### Gravitational Force

The force $\vec{F}_{12}$ exerted on body 1 by body 2 is given by:

$$
\vec{F}_{12} = G \frac{m_1 m_2}{|\vec{r}|^2} \hat{r}
$$

Where:

- $m_1, m_2$ are the masses of the two bodies.
- $\vec{r} = \vec{r}_2 - \vec{r}_1$ is the position vector pointing from body 1 to body 2.
- $|\vec{r}|$ is the distance between the bodies.
- $\hat{r} = \frac{\vec{r}}{|\vec{r}|}$ is the unit vector pointing from body 1 to body 2.
- $G$ is the gravitational constant.

### Equations of Motion

The system is described by a system of coupled Ordinary Differential Equations (ODEs). For each body $i$, the acceleration is determined by the total force acting on it:

$$
\begin{aligned}
\frac{d\vec{r}_i}{dt} &= \vec{v}_i \\
\frac{d\vec{v}_i}{dt} &= \frac{\vec{F}_{net}}{m_i}
\end{aligned}
$$

For this two-body system:

- **Body 1 Acceleration**: $\vec{a}_1 = \frac{\vec{F}_{12}}{m_1}$
- **Body 2 Acceleration**: $\vec{a}_2 = \frac{-\vec{F}_{12}}{m_2}$ (Newton's Third Law)

## Implementation Details

### State Vector

The state of the system is represented by a single 8-dimensional vector containing positions and velocities for both bodies:

$$
\mathbf{S} = [x_1, y_1, v_{x1}, v_{y1}, x_2, y_2, v_{x2}, v_{y2}]
$$

### Parameters

The simulation uses the following default parameters (defined in `two-planets.py`):

- **Planet 1**: Mass $m_1 = 10$
- **Planet 2**: Mass $m_2 = 1$
- **Initial State**:
  - Planet 1 starts at $(5, 5)$ with velocity $(1, 0)$.
  - Planet 2 starts at $(8, 10)$ with velocity $(0.3, 0)$.

### Example Preview

<video src="two-planets.mp4" controls title="Two Planets Simulation" width="100%"></video>

## Usage

You can run this example directly to see the animation:

```bash
python examples/two-body-planets/two-planets.py
```

Or run it via the main project entry point by modifying `main.py`:

```python
from examples import two_planets_main

def main():
    two_planets_main()
```
