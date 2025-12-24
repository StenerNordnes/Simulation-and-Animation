# Numerical Solver Module Documentation

This module implements a flexible **Explicit Runge-Kutta (ERK)** solver for Ordinary Differential Equations (ODEs). It is designed to be modular, allowing users to define custom methods via Butcher Tableaus or use pre-defined standard methods (such as Forward Euler or RK4).

## Mathematical Background

The solver finds numerical approximations for initial value problems of the form:

$$
\frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0
$$

It uses the generalized Runge-Kutta method. For a step size $h$ (denoted as `dt` in the code), the next value $y_{n+1}$ is calculated as:

$$
y_{n+1} = y_n + h \sum_{i=1}^{s} b_i k_i
$$

Where $k_i$ are the intermediate slopes (stages) calculated as:

$$
k_i = f(t_n + c_i h, \ y_n + h \sum_{j=1}^{s} a_{ij} k_j)
$$

### Butcher Tableau

The coefficients $A$ (matrix), $b$ (weights), and $c$ (nodes) are organized into a **Butcher Tableau**. This allows the solver to switch between different integration methods (e.g., Euler vs. RK4) simply by swapping the configuration object.

The tableau is often represented as:

$$
\begin{array}{c|c}
\mathbf{c} & A \\
\hline
& \mathbf{b}^T
\end{array}
$$

---

## Class Documentation

### 1. `ButcherTableau`

A data class that defines the coefficients for a specific Runge-Kutta method.

**Constructor:**

```python
ButcherTableau(a=None, b=None, c=None)
```

**Parameters:**

- `a` (`NDArray`): The Runge-Kutta matrix (size $s \times s$). If `None`, defaults to Euler `[[0]]`.
- `b` (`NDArray`): The weights vector (size $s$). If `None`, defaults to Euler `[1]`.
- `c` (`NDArray`): The nodes vector (size $s$). If `None`, defaults to Euler `[0]`.

**Attributes:**

- `self.stages`: Automatically derived from the length of `a` (represents $s$).

#### Pre-defined Solvers

The module instantiates several common tableaus globally:

| Variable Name      | Method              | Order | Description                                 |
| :----------------- | :------------------ | :---- | :------------------------------------------ |
| `euler_tableau`    | Forward Euler       | 1st   | Simplest explicit method.                   |
| `midpoint_tableau` | Midpoint Method     | 2nd   | RK2 with a step at $t + 0.5dt$.             |
| `rk2_tableu`       | Generic RK2         | 2nd   | Standard 2nd order RK.                      |
| `heun_tableau`     | Heun's Method       | 2nd   | Also known as the Trapezoidal Rule.         |
| `ralston_tableau`  | Ralston's Method    | 2nd   | RK2 with minimum error bound.               |
| `rk4_tableau`      | Classic Runge-Kutta | 4th   | The standard, highly accurate "RK4" method. |

---

### 2. `NumericalSolver`

The core engine that performs the time-stepping integration.

**Constructor:**

```python
NumericalSolver(butcherTableau, d_function, profiling=False)
```

**Parameters:**

- `butcherTableau` (`ButcherTableau`): The configuration object defining the integration method.
- `d_function` (`Callable`): The derivative function $f(t, y)$. Must accept `(float, NDArray)` and return `NDArray`.
- `profiling` (`bool`): If `True`, enables performance tracking for execution time.

#### Methods

**`solve(initial_state, start_time, end_time, dt)`**
Main execution method. Iterates from `start_time` to `end_time` using step size `dt`.

- **Returns:** `NDArray` of shape `(steps + 1, state_dimension)` containing the time series of the state.

**`get_ks(t_k, x_k, dt)`**
Internal helper method that calculates the $k$ values (stages) for a single time step based on the provided Butcher Tableau.

**`print_execution_time()`**
Prints a detailed breakdown of the solver's performance. Only functions if `profiling=True` was passed during initialization.

---

## Profiling Features

If `profiling=True` is enabled, the `NumericalSolver` tracks the real-world time spent in different parts of the calculation. Call `solver.print_execution_time()` after solving to view:

- **Total Solve Time:** Aggregate time spent in the `solve` loop.
- **Stage Calculation (`get_ks`):** Time spent calculating intermediate derivatives ($f(t, y)$ calls).
- **Update Step:** Time spent summing the stages to update the state vector.
- **Per-Step Averages:** High-precision timing averaged over the total number of steps taken.

---

## Usage Example

Below is an example of how to use the `NumericalSolver` to solve a simple Harmonic Oscillator (spring-mass system).

$$
\frac{d^2x}{dt^2} = -x \implies \begin{cases} \dot{x} = v \\ \dot{v} = -x \end{cases}
$$

```python
import numpy as np
from your_module import NumericalSolver, rk4_tableau

# 1. Define the derivative function f(t, y)
# state y = [position, velocity]
def harmonic_oscillator(t: float, y: np.ndarray) -> np.ndarray:
    position = y[0]
    velocity = y[1]

    d_position = velocity
    d_velocity = -position

    return np.array([d_position, d_velocity])

# 2. Initialize the solver with RK4
solver = NumericalSolver(
    butcherTableau=rk4_tableau,
    d_function=harmonic_oscillator,
    profiling=True
)

# 3. Define simulation parameters
y0 = np.array([1.0, 0.0]) # Start at position 1, velocity 0
t_start = 0.0
t_end = 10.0
dt = 0.01

# 4. Run the solver
result = solver.solve(y0, t_start, t_end, dt)

# 5. Access results and check profiling
print(f"Final State: {result[-1]}")
solver.print_execution_time()
```
