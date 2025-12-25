# Simulation and Animation Framework

A modular Python framework for simulating physical systems using numerical methods and visualizing them with 2D/3D animations. This project combines a flexible ODE/DAE solver with an object-oriented animation engine.

## Features

- **Numerical Solver**: Custom implementation of Explicit Runge-Kutta methods (RK4, Euler, Heun, etc.) for solving Ordinary Differential Equations (ODEs) and Differential-Algebraic Equations (DAEs).
- **Animation Engine**:
  - **Object Animation**: Animate shapes like circles, rectangles, springs, and connecting rods.
  - **Trajectory Animation**: Visualize particle paths in 2D and 3D space.
- **Modular Design**: Separated logic for solvers, animation objects, and example implementations.

## Documentation

Detailed documentation for the core modules can be found here:

- **[Animation Module](animation/README.md)**: Guide to `AnimatableObject`, shapes, and trajectory visualization.
- **[Solver Module](solver/README.md)**: Guide to `NumericalSolver`, `ButcherTableau`, and implementing custom integration methods.

## Setup and Installation

This project uses **[uv](https://github.com/astral-sh/uv)** for Python package management.

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd Simulation-and-Animation
   ```

2. **Install dependencies:**
   Initialize the virtual environment and install packages using `uv`:

   ```bash
   uv sync
   ```

   Or manually:

   ```bash
   uv venv
   uv pip install -r pyproject.toml
   ```

   _Note: If you don't use `uv`, you can install dependencies via pip:_

   ```bash
   pip install numpy matplotlib
   ```

## Usage

### Running the Main Application

The project includes a `main.py` entry point that runs a selected example.

```bash
uv run main.py
```

Or with standard python:

```bash
python main.py
```

### Running Examples

The `examples/` directory contains various physical simulations. You can run them by importing them in `main.py` or running them directly (ensure the project root is in your PYTHONPATH).

Available examples include:

- **Spring Pendulum**: A pendulum attached to a spring (`examples/pendulum-on-spring`).
- **Mass on Wheel**: A rod connected to a rotating wheel (`examples/mass-wheel-rotation`).
- **Two Planets**: A two-body gravitational simulation (`examples/two-body-planets`).

To switch examples in `main.py`:

```python
from examples import spring_pendulum_main, mass_wheel_main, two_planets_main

def main():
    # spring_pendulum_main()
    mass_wheel_main()  # Run the mass-wheel example
```

## Project Structure

```
.
├── animation/              # Animation engine
│   ├── objects.py          # Classes for shapes (Circle, Rectangle, Spring, etc.)
│   ├── trajectories.py     # 2D/3D path visualization
│   └── README.md        # Animation docs
├── solver/                 # Numerical methods
│   ├── numerical_solver.py # RK4 and other solvers
│   └── README.md           # Solver docs
├── examples/               # Simulation examples
│   ├── pendulum-on-spring/
│   ├── mass-wheel-rotation/
│   └── two-body-planets/
├── main.py                 # Entry point
├── pyproject.toml          # Dependencies
└── README.md               # Project documentation
```
