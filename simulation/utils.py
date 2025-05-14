import numpy as np


def RotationMatrix(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class Rotation:
    def Z(theta: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    def Y(theta: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

    def X(theta: float) -> np.ndarray:
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )


if __name__ == "__main__":
    # Test the Rotation class
    theta = np.pi / 4  # 45 degrees
    print("Rotation around Z-axis:")
    print(Rotation.Z(theta))
    print("Rotation around Y-axis:")
    print(Rotation.Y(theta))
    print("Rotation around X-axis:")
    print(Rotation.X(theta))
