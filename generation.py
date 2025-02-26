import numpy as np

from collections import namedtuple
from itertools import product
from matplotlib import pyplot as plt
from shapely import geometry
from typing import Tuple

from numpy import s_

BoundsTuple = namedtuple("bounds", ["min", "max"])
CoordinateTuple = namedtuple("coordinates", ["x", "y"])

square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
fun_shape = np.array(
    [
        [0, 0],
        [2, 0],
        [2, 2],
        [-4, 2],
        [-4, 1],
        [-3, 1],
        [-3, 0],
        [-2, 0],
        [-2, 1],
        [0, 1],
    ]
)


def ordered_points_to_cycle(V: np.array) -> np.array:
    """Turns, for example, (a, b, c) to (a, b, c, a)

    Parameters
    ----------
    V : np.array
        (n_points, n_dim) ordered point

    Returns
    -------
    np.array
        (n_points+1, n_dim) output cycle
    """
    return np.concatenate((V, V[[0], :]), axis=0)


def ordered_points_to_vectors(V: np.array) -> np.ndarray:
    """From (a, b, c) produce vectors (b-a, c-b, a-c)

    Parameters
    ----------
    V : np.array
        (n_points, n_dim) ordered points

    Returns
    -------
    np.ndarray
        (n_points, n_dim) vectors
    """
    n = V.shape[0]  # number of points
    return V[(np.arange(n) + 1) % n, :] - V


def project_points_onto_polygon_edge_span(
    V: np.ndarray, p: np.ndarray, open: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """[summary]

    Parameters
    ----------
    V : np.ndarray
        (n_vertices, n_dim)
    p : np.ndarray
        (n_points, n_dim)
    open : bool
        whether being 'on an edge' considers the edge open (or closed)

    Returns
    -------
    proj : np.ndarray
        (n_vertices, n_points, n_dim) projected points
    on_edge : np.ndarray
        (n_vertices, n_points) whether points are actually on edge
    """
    _V = V[:, np.newaxis, :]
    pv1 = p - _V  # (n_vertices, n_points, n_dim)
    # vectors from vertex "1" to point
    v1v2 = ordered_points_to_vectors(V)  # (n_vertices, n_dim)
    # vectors from vertex "1" to "2"
    proj = np.einsum("ijk,ik->ij", pv1, v1v2)  # (n_vertices, n_points)
    # inner product <pv1, v1v2>
    proj = proj / np.power(np.linalg.norm(v1v2, axis=-1)[:, np.newaxis], 2)
    # (n_vertices, n_points) / (n_vertices, 1) ->
    # (n_vertices, n_points)
    # <pv1, v1v2> / <v1v2, v1v2>
    if open:
        on_edge = (0 < proj) & (proj < 1)
    else:
        on_edge = (0 <= proj) & (proj <= 1)
    proj = v1v2[:, np.newaxis, :] * proj[:, :, np.newaxis]
    # (n_vertices, 1, n_dim) * (n_vertices, n_points, 1) ->
    # (n_vertices, n_points, n_dim)
    return proj + _V, on_edge


def points_in_rectangle(V: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Return a boolean array showing if p is in the rectangle defined by V.
    Rectangle considered closed, i.e.
    (1, 1) is in Rectangle((1, 1), (3, 3))

    Parameters
    ----------
    V : np.ndarray
        (4, 2) rectangle
    p : np.ndarray
        (n_points, 2)

    Returns
    -------
    np.ndarray
        (n_points,) if points are in rectangle
    """
    _, on_edge = project_points_onto_polygon_edge_span(V, p, open=False)
    return on_edge.all(0)


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
            (V[:, 1] + (V[:, 1] > q))[:, np.newaxis],
        ),
        axis=-1,
    )


def cut(V: np.array, c: np.array, rng: np.random.Generator) -> Tuple[np.ndarray, bool]:
    """Cut a grid polygon about the center point c.

    Parameters
    ----------
    V : np.array
        (n_vertices, 2) vertices of the polygon
    c : np.array
        (2,) point to cut around
    rng : np.random.Generator
        random number generator for random order of cutting directions

    Returns
    -------
    V : np.array
        (n_vertices, 2) vertices of new polygon
    success : bool
    """
    proj, on_edge = project_points_onto_polygon_edge_span(
        V, c
    )  # project c onto edge span
    proj = proj.squeeze()  # always queried with one point
    on_edge = on_edge.squeeze()
    proj = proj[on_edge]  # filter out projections not on edge
    shuffle_idxs = rng.permutation(proj.shape[0])
    proj = proj[shuffle_idxs]
    v1 = V[on_edge][shuffle_idxs]
    v2 = V[(on_edge.nonzero()[0] + 1) % V.shape[0]][shuffle_idxs]
    for i, s in enumerate(proj):  # iterate over projected points
        if (s == V).all(-1).any():  # if s is a vertex
            continue  # cannot cut here
        shuffle_idxs = rng.permutation(2)
        candidate_vs = np.array([v1[i], v2[i]])[shuffle_idxs]
        for j, vm in enumerate(candidate_vs):
            sp = c + vm - s
            rect = np.array([s, c, sp, vm])
            if points_in_rectangle(rect, V).sum() > 1:
                continue
            # if we get here this is a valid cut
            vi = np.nonzero((vm == V).all(-1))[0].item()
            order = (shuffle_idxs[1] + j) % 2
            newV = np.array([sp, c, s]) if order else np.array([s, c, sp])
            newV = np.concatenate(
                (
                    V[:vi, :],
                    newV,
                    V[vi + 1 :, :],
                ),
                axis=0,
            )
            yield newV
    pass


def get_orthogonal_hull(V: np.ndarray) -> CoordinateTuple:
    return CoordinateTuple(
        BoundsTuple(V[:, 0].min(), V[:, 0].max()),
        BoundsTuple(V[:, 1].min(), V[:, 1].max()),
    )


def contains(V: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Determine if the points in p are contained in V.
    Currently uses shapely, may be slow.

        Parameters
        ----------
        V : np.ndarray
            (n_vertices, 2)
        p : np.ndarray
            (n_points, 2)

        Returns
        -------
        np.ndarray
            Winding number of each point, shape (n_points, )
    """
    shape = geometry.Polygon(V)
    return np.array([shape.contains(geometry.Point(pi)) for pi in p])


def interior_points(V: np.ndarray):
    hull = get_orthogonal_hull(V)
    X, Y = np.meshgrid(
        np.arange(hull.x.min + 1, hull.x.max),
        np.arange(hull.y.min + 1, hull.y.max),
    )
    test_points = np.vstack((X.flatten(), Y.flatten())).T
    return test_points[contains(V, test_points)]


def plot_polygon(V: np.array, ax: plt.Axes):
    V = ordered_points_to_cycle(V)
    ax.plot(V[:, 0], V[:, 1], color="white", alpha=0.5)
    ax.grid(linestyle="dotted")
    ax.set_aspect("equal")
    hull = get_orthogonal_hull(V)
    ax.set_xlim(hull.x.min - 1, hull.x.max + 1)
    ax.set_ylim(hull.y.min - 1, hull.y.max + 1)

    X, Y = np.meshgrid(
        np.linspace(hull.x.min - 1, hull.x.max + 1, 41),
        np.linspace(hull.y.min - 1, hull.y.max + 1, 41),
    )
    ax.set_facecolor("black")
    # Z = self.edf_at_point(
    #     np.array([X.flatten(), Y.flatten()]).T
    # )
    # ax.contourf(
    #     X, Y, Z.reshape(X.shape), levels=80,
    #     vmin=self._last_plot_vmin, vmax=self._last_plot_vmax)
    return ax


def plot_points(p: np.ndarray, ax: plt.Axes) -> plt.Axes:
    """
    Parameters
    ----------
    p : np.ndarray
        (..., 2) points
    ax : plt.Axes

    Returns
    -------
    plt.Axes
    """
    _p = p.reshape(-1, 2)
    ax.scatter(*_p.T, marker="+", c="white", s=5)
    return ax


def inflate_cut(V: np.ndarray, depth: int, rng: np.random.Generator):
    for c in interior_points(V):
        _V = inflate(V, c)
        for d in interior_points(_V):
            newrng = np.random.default_rng(seed=rng.integers(2**16))
            for shape in cut(_V, d, newrng):
                if depth == 0:
                    yield shape
                else:
                    newerrng = np.random.default_rng(seed=rng.integers(2**16))
                    for s in inflate_cut(shape, depth - 1, newerrng):
                        yield s
        # if depth == 0:

        # newrng = np.random.default_rng(seed=rng.integers(2**16))
        # for shape in inflate_cut(_V, depth-1, newrng):
        #     yield shape


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)
    ax = plot_polygon(fun_shape * 2, ax)
    proj, on_edge = project_points_onto_polygon_edge_span(
        fun_shape * 2,
        # np.array([[1, 1], [3.6, 3.6], [-5.9, 0.1]]),
        np.array([[2, 1]]),
    )
    filtered = proj[on_edge, :]
    plot_points(filtered, ax)

    shape_bytes = set()
    shapes = []
    depth = 1
    for s in inflate_cut(square * 2, depth, np.random.default_rng(0)):
        if s.tobytes() not in shape_bytes:
            shape_bytes.add(s.tobytes())
            shapes.append(s)
    print(f"Depth {depth}, generated {len(shapes)} unique shapes.")
    for s in shapes[::50]:
        fig, ax = plt.subplots(1, 1)
        ax = plot_polygon(s, ax)
