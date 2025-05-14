# %%
import numpy as np

from simulation.animate import (
    animate_trajectory,
)  # Removed unused imports
from simulation.animate_objects import Rectangle, animate_objects
import sympy as sm
from sympy.core.basic import Basic

# from sympy.core.relational import Equality # Removed unused import
from IPython.display import display_latex


def lambify_function(expr: Basic, t_sym):
    """Lambdifies a SymPy expression."""
    return sm.lambdify(t_sym, expr, "numpy")


# %%
if __name__ == "__main__":
    # Example Usage: Simulate a simple circular trajectory
    t = sm.symbols("t")
    d, k, m = sm.symbols("d k m")
    x: sm.Function = sm.Function("x")(t)  # type: ignore
    x_dot = sm.diff(x, t)  # This calculates dx/dt
    x_ddot = sm.diff(x, t, t)  # This calculates d^2x/dt^2

    equation: sm.Eq = m * x_ddot + d * x_dot + k * x
    subbed_equation = equation.subs({"m": 10, "d": 2, "k": 1000})

    solved = sm.dsolve(
        subbed_equation, x, ics={x.func(0): 17, sm.diff(x, t, t).subs(t, 0): 3}
    )
    display_latex(solved)  # Display the solved equation as well

    solution_expr = solved.rhs  # type: ignore
    func = lambify_function(solution_expr, t)

    x_values = np.linspace(0, 30, 1000)

    example_trajectory = np.column_stack((func(x_values), x_values))

    rectangle = Rectangle(example_trajectory, length=20, width=20, show_trajectory=True)

    animate_objects(
        [rectangle],  # Rod is last
        interval=30,  # Slower interval
        title="Object Animation with Connective Rod",
        xlabel="X Position",
        ylabel="Y Position",
    )

    print("Example finished.")
