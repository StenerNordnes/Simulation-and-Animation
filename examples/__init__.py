"""Examples package - imports main functions from each example."""

import importlib.util
import sys
from pathlib import Path

# Get the examples directory
examples_dir = Path(__file__).parent


def _load_module_from_file(module_name, file_path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None:
        raise ValueError("Spec is None")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    if spec.loader is None:
        raise ValueError("Spec.loader is None")

    spec.loader.exec_module(module)
    return module


# Load spring-pendulum example
spring_pendulum_module = _load_module_from_file(
    "spring_pendulum", examples_dir / "pendulum-on-spring" / "spring-pendulum.py"
)
spring_pendulum_main = spring_pendulum_module.main

# Load mass-wheel example
mass_wheel_module = _load_module_from_file(
    "mass_wheel", examples_dir / "mass-wheel-rotation" / "mass-wheel.py"
)
mass_wheel_main = mass_wheel_module.main

# Load two-planets example
two_planets_module = _load_module_from_file(
    "two_planets", examples_dir / "two-body-planets" / "two-planets.py"
)
two_planets_main = two_planets_module.main


# Load two-planets example
three_body_problem = _load_module_from_file(
    "three_planets", examples_dir / "three-body-problem" / "three-planets.py"
)
three_body_problem_main = three_body_problem.main

__all__ = [
    "spring_pendulum_main",
    "mass_wheel_main",
    "two_planets_main",
    "three_body_problem_main",
]
