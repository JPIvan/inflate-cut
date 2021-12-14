import numpy as np

from collections import namedtuple
from matplotlib import pyplot as plt

BoundsTuple = namedtuple("bounds", ["min", "max"])
CoordinateTuple = namedtuple("coordinates", ["x", "y"])

square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

def inflate(V: np.array, c: np.array) -> np.ndarray:
    """Inflate a grid polygon about the center point c.

    Parameters
    ----------
    V : np.array
        (n_vertices, 2) vertices of the polygon
    c : np.array
        (2,) point to inflate around

    Returns
    -------
    V : np.array
        (n_vertices, 2) vertices of new polygon
    """
    p, q = c
    return np.concatenate(
        (
            (V[:, 0] + (V[:, 0] > p))[:, np.newaxis],
            (V[:, 1] + (V[:, 1] > q))[:, np.newaxis]
        ),
        axis=-1
    )

def get_orthogonal_hull(V: np.ndarray) -> CoordinateTuple:
    return CoordinateTuple(
        BoundsTuple(V[:, 0].min(), V[:, 0].max()),
        BoundsTuple(V[:, 1].min(), V[:, 1].max()),
    )

def plot_polygon(V: np.array, ax: plt.Axes):
    V = np.concatenate((V, V[[0], :]), axis=0)
    ax.plot(V[:, 0], V[:, 1], color='white', alpha=0.5)
    ax.grid(linestyle='dotted')
    ax.set_aspect('equal')
    hull = get_orthogonal_hull(V)
    ax.set_xlim(hull.x.min - 1, hull.x.max + 1)
    ax.set_ylim(hull.y.min - 1, hull.y.max + 1)

    X, Y = np.meshgrid(
        np.linspace(hull.x.min - 1, hull.x.max + 1, 41),
        np.linspace(hull.y.min - 1, hull.y.max + 1, 41)
    )
    ax.set_facecolor('black')
    # Z = self.edf_at_point(
    #     np.array([X.flatten(), Y.flatten()]).T
    # )
    # ax.contourf(
    #     X, Y, Z.reshape(X.shape), levels=80,
    #     vmin=self._last_plot_vmin, vmax=self._last_plot_vmax)
    return ax