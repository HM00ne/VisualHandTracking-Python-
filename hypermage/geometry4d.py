"""
geometry4d.py — real higher-dimensional geometry: shapes in 3, 4, 5 and 6 dimensions, rotations
in every coordinate plane, perspective projection down to 2D, and slicing.

Every shape is a set of N-dimensional vertices plus an edge list. Polytopes also carry their
2D faces (as lists of edge indices) so that slicing can connect the cut points into a proper
cross-section. Everything is vectorised with NumPy; nothing loops per vertex at render time.

Axes: 0=x, 1=y, 2=z, 3=w, 4=v, 5=u.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

PHI = (1 + 5 ** 0.5) / 2
MAX_DIM = 6
DIMENSIONS = (3, 4, 5, 6)                  # the three-finger gesture steps through these

# Every rotation plane of 6D space (15 of them). An N-D object uses the planes whose two axes
# are both < N: 3 planes in 3D, 6 in 4D, 10 in 5D, 15 in 6D.
PLANES = list(itertools.combinations(range(MAX_DIM), 2))
N_PLANES = len(PLANES)
_AXIS = "XYZWVU"
PLANE_NAMES = [_AXIS[i] + _AXIS[j] for i, j in PLANES]
PLANE_INDEX = {p: k for k, p in enumerate(PLANES)}
XY, XZ, XW, YZ, YW, ZW = (PLANE_INDEX[p] for p in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


@dataclass
class Shape4D:
    name: str
    verts: np.ndarray            # (N, D) float32 (D = 3..6), scaled so the farthest vertex is at radius 1
    edges: np.ndarray            # (E, 2) int32
    faces: np.ndarray | None     # (F, K) int32 edge indices, padded with -1; None for curve meshes

    @property
    def dim(self) -> int:
        return self.verts.shape[1]

    @property
    def counts(self) -> str:
        f = 0 if self.faces is None else len(self.faces)
        return f"{len(self.verts)} vertices, {len(self.edges)} edges, {f} faces"


# ======================================================================================
# Construction helpers
# ======================================================================================
def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return (v / np.linalg.norm(v, axis=1).max()).astype(np.float32)


def _shortest_edges(verts: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """Edges of a uniform polytope = all vertex pairs at the minimum distance."""
    d = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    i, j = np.where(np.abs(d - d.min()) < tol)
    keep = i < j
    return np.stack([i[keep], j[keep]], axis=1).astype(np.int32)


def _edge_lookup(edges: np.ndarray) -> dict[tuple[int, int], int]:
    return {(int(min(a, b)), int(max(a, b))): k for k, (a, b) in enumerate(edges)}


def _faces_from_cycles(cycles: list[list[int]], edges: np.ndarray) -> np.ndarray:
    """Turn vertex cycles into rows of edge indices, padded with -1 to equal length."""
    lookup = _edge_lookup(edges)
    width = max(len(c) for c in cycles)
    out = np.full((len(cycles), width), -1, dtype=np.int32)
    for r, cyc in enumerate(cycles):
        for k in range(len(cyc)):
            a, b = cyc[k], cyc[(k + 1) % len(cyc)]
            out[r, k] = lookup[(min(a, b), max(a, b))]
    return out


def _triangle_faces(n: int, edges: np.ndarray) -> np.ndarray:
    """All triangles of the edge graph. For the 16-, 24- and 600-cell these are exactly the 2-faces."""
    nbrs = [set() for _ in range(n)]
    for a, b in edges:
        nbrs[a].add(int(b)); nbrs[b].add(int(a))
    tris = []
    for a, b in edges:
        i, j = sorted((int(a), int(b)))
        for k in nbrs[i] & nbrs[j]:
            if k > j:
                tris.append([i, j, k])
    return _faces_from_cycles(tris, edges)


def _even_permutations(n: int = 4):
    for p in itertools.permutations(range(n)):
        inversions = sum(1 for a in range(n) for b in range(a + 1, n) if p[a] > p[b])
        if inversions % 2 == 0:
            yield p


# ======================================================================================
# The shapes
# ======================================================================================
def hypercube(n: int = 4) -> Shape4D:
    """n-cube: 2^n vertices (±1,...,±1). n=3 cube, 4 tesseract, 5 penteract, 6 hexeract."""
    corners = list(itertools.product((-1, 1), repeat=n))
    verts = np.array(corners, dtype=np.float64)
    edges = _shortest_edges(verts)
    index = {c: k for k, c in enumerate(corners)}
    squares = []
    for a, b in itertools.combinations(range(n), 2):            # the plane of the square
        others = [k for k in range(n) if k not in (a, b)]
        for fixed in itertools.product((-1, 1), repeat=n - 2):   # position of the square
            def corner(sa, sb):
                c = [0] * n
                c[a], c[b] = sa, sb
                for k, v in zip(others, fixed):
                    c[k] = v
                return index[tuple(c)]
            squares.append([corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)])
    return Shape4D(HYPERCUBE_NAMES[n], _normalise(verts), edges, _faces_from_cycles(squares, edges))


def orthoplex(n: int = 4) -> Shape4D:
    """n-orthoplex (cross-polytope): 2n vertices ±e_i. n=3 octahedron, 4 the 16-cell."""
    verts = np.vstack([np.eye(n), -np.eye(n)])
    edges = _shortest_edges(verts)
    return Shape4D(ORTHOPLEX_NAMES[n], _normalise(verts), edges, _triangle_faces(len(verts), edges))


HYPERCUBE_NAMES = {3: "Cube", 4: "Tesseract", 5: "Penteract (5-cube)", 6: "Hexeract (6-cube)"}
ORTHOPLEX_NAMES = {3: "Octahedron", 4: "16-cell", 5: "5-orthoplex", 6: "6-orthoplex"}


def tesseract() -> Shape4D:
    return hypercube(4)


def cell16() -> Shape4D:
    return orthoplex(4)


def cell24() -> Shape4D:
    """24-cell: all permutations of (±1,±1,0,0): 24 vertices, 96 edges, 96 triangles."""
    vs = set()
    for i, j in itertools.combinations(range(4), 2):
        for si, sj in itertools.product((-1, 1), repeat=2):
            v = [0, 0, 0, 0]
            v[i], v[j] = si, sj
            vs.add(tuple(v))
    verts = np.array(sorted(vs), dtype=np.float64)
    edges = _shortest_edges(verts)
    return Shape4D("24-cell", _normalise(verts), edges, _triangle_faces(len(verts), edges))


def cell600() -> Shape4D:
    """600-cell: 120 vertices, 720 edges, 1200 triangles (the 120-cell's 600 vertices are too heavy)."""
    vs = []
    for i in range(4):                                           # 8 of the form (±1,0,0,0)
        for s in (-1, 1):
            v = [0.0] * 4
            v[i] = s
            vs.append(v)
    for signs in itertools.product((-0.5, 0.5), repeat=4):       # 16 of the form (±½,±½,±½,±½)
        vs.append(list(signs))
    base = [PHI / 2, 0.5, 1 / (2 * PHI), 0.0]                   # 96 even perms of ½(±φ,±1,±1/φ,0)
    for perm in _even_permutations():
        for signs in itertools.product((-1, 1), repeat=3):
            v = [0.0] * 4
            vals = [base[0] * signs[0], base[1] * signs[1], base[2] * signs[2], 0.0]
            for src, dst in enumerate(perm):
                v[dst] = vals[src]
            vs.append(v)
    verts = np.array(vs, dtype=np.float64)
    edges = _shortest_edges(verts)
    return Shape4D("600-cell", _normalise(verts), edges, _triangle_faces(len(verts), edges))


def _torus_grid(name: str, p: int, q: int, ring_faces: bool) -> Shape4D:
    """Points (cos a, sin a, cos b, sin b): a p×q duoprism, or a Clifford torus when finely sampled."""
    a = np.arange(p) * 2 * np.pi / p
    b = np.arange(q) * 2 * np.pi / q
    A, B = np.meshgrid(a, b, indexing="ij")
    verts = np.stack([np.cos(A), np.sin(A), np.cos(B), np.sin(B)], axis=-1).reshape(-1, 4)
    idx = lambda i, j: (i % p) * q + (j % q)                      # noqa: E731
    edges = [(idx(i, j), idx(i + 1, j)) for i in range(p) for j in range(q)]
    edges += [(idx(i, j), idx(i, j + 1)) for i in range(p) for j in range(q)]
    edges = np.array([(min(e), max(e)) for e in edges], dtype=np.int32)
    cycles = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
              for i in range(p) for j in range(q)]
    if ring_faces:                                                # the duoprism's p-gon and q-gon cells
        cycles += [[idx(i, j) for i in range(p)] for j in range(q)]
        cycles += [[idx(i, j) for j in range(q)] for i in range(p)]
    return Shape4D(name, _normalise(verts), edges, _faces_from_cycles(cycles, edges))


def duoprism(p: int = 6, q: int = 6) -> Shape4D:
    return _torus_grid(f"{p}x{q} duoprism", p, q, ring_faces=True)


def clifford_torus(n: int = 22) -> Shape4D:
    return _torus_grid("Clifford torus", n, n, ring_faces=False)


def hopf_fibration(rings=(0.22, 0.5, 0.78), fibers_per_ring: int = 9, samples: int = 36) -> Shape4D:
    """
    The 3-sphere filled with Hopf circles. Each circle is
        (cos η cos t, cos η sin t, sin η cos(t+φ), sin η sin(t+φ)),  t ∈ [0, 2π),
    one circle per point (η, φ) of the base 2-sphere. Every pair of circles is linked once.
    """
    t = np.arange(samples) * 2 * np.pi / samples
    verts, edges = [], []
    for ring_k, frac in enumerate(rings):
        eta = frac * np.pi / 2
        for f in range(fibers_per_ring):
            phi = f * 2 * np.pi / fibers_per_ring + ring_k * 0.35
            pts = np.stack([np.cos(eta) * np.cos(t), np.cos(eta) * np.sin(t),
                            np.sin(eta) * np.cos(t + phi), np.sin(eta) * np.sin(t + phi)], axis=1)
            start = len(verts) * samples
            verts.append(pts)
            k = np.arange(samples)
            edges.append(np.stack([start + k, start + (k + 1) % samples], axis=1))
    return Shape4D("Hopf fibration", _normalise(np.vstack(verts)),
                   np.vstack(edges).astype(np.int32), None)


def prism(shape: Shape4D, name: str) -> Shape4D:
    """
    Extrude a D-dimensional shape into D+1 dimensions: two copies at ±h on the new axis, with
    every vertex joined to its copy. Faces: both copies' faces plus one square per original edge.
    """
    n, d = shape.verts.shape
    h = 0.6
    v = np.vstack([np.hstack([shape.verts, np.full((n, 1), -h)]),
                   np.hstack([shape.verts, np.full((n, 1), h)])])
    e = shape.edges
    E = len(e)
    edges = np.vstack([e, e + n, np.stack([np.arange(n), np.arange(n) + n], 1)]).astype(np.int32)
    faces = None
    if shape.faces is not None:
        F = shape.faces
        top = np.where(F >= 0, F + E, -1)
        rungs = 2 * E + e                                          # edge index of vertex -> copy
        squares = np.stack([np.arange(E), rungs[:, 1], np.arange(E) + E, rungs[:, 0]], 1)
        width = max(F.shape[1], 4)
        pad = lambda A: np.hstack([A, np.full((len(A), width - A.shape[1]), -1)])   # noqa: E731
        faces = np.vstack([pad(F), pad(top), pad(squares)]).astype(np.int32)
    return Shape4D(name, _normalise(v), edges, faces)


def shadow_3d(shape: Shape4D, name: str) -> Shape4D:
    """The 3D shadow of a 4D shape: drop the w coordinate (orthogonal projection)."""
    return Shape4D(name, _normalise(shape.verts[:, :3]), shape.edges, shape.faces)


# Order = keyboard 1..7 and the creation cycle.
SHAPE_BUILDERS = [tesseract, cell16, cell24, cell600, duoprism, clifford_torus, hopf_fibration]
SHAPE_NAMES = ["Tesseract", "16-cell", "24-cell", "600-cell", "6x6 duoprism",
               "Clifford torus", "Hopf fibration"]
# Lighter versions used when a heavy shape is extruded to 5D/6D (prisms multiply the edge count)
_LIGHT = {5: lambda: clifford_torus(12), 6: lambda: hopf_fibration(fibers_per_ring=5, samples=24)}
_cache: dict[tuple[int, int], Shape4D] = {}


def get_shape(index: int, dim: int = 4) -> Shape4D:
    """The shape in the requested dimension. Built once and shared by every object of that type."""
    index %= len(SHAPE_BUILDERS)
    key = (index, dim)
    if key in _cache:
        return _cache[key]
    name = SHAPE_NAMES[index]
    if index == 0:
        shape = hypercube(dim)
    elif index == 1:
        shape = orthoplex(dim)
    elif dim == 4:
        shape = SHAPE_BUILDERS[index]()
    elif dim == 3:
        shape = shadow_3d(get_shape(index, 4), f"{name} (3D shadow)")
    else:
        base = _LIGHT[index]() if index in _LIGHT else get_shape(index, 4)
        if dim == 5:
            shape = prism(base, f"{name} prism (5D)")
        else:
            shape = prism(prism(base, ""), f"{name} double prism (6D)")
    _cache[key] = shape
    return shape


def shape_name(index: int, dim: int) -> str:
    return get_shape(index, dim).name


# ======================================================================================
# Rotation, projection, slicing
# ======================================================================================
def rotation_matrix(angles: np.ndarray, n: int) -> np.ndarray:
    """n×n rotation from the plane angles (indexed like PLANES), using only planes inside n-D."""
    R = np.eye(n)
    for (i, j), a in zip(PLANES, angles):
        if j >= n or a == 0.0:
            continue
        c, s = np.cos(a), np.sin(a)
        G = np.eye(n)
        G[i, i] = G[j, j] = c
        G[i, j], G[j, i] = -s, s
        R = G @ R
    return R


def rotate(verts: np.ndarray, angles: np.ndarray) -> np.ndarray:
    return verts @ rotation_matrix(angles, verts.shape[1]).T.astype(np.float32)


def project_nd_to_3d(v: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Perspective-project N-D points down to 3D one axis at a time (u, then v, then w), like a
    chain of cameras. Returns the 3D points and each point's w (0 for 3D shapes), used for colour.
    """
    p = v
    for axis in range(v.shape[1] - 1, 3, -1):                  # extra axes beyond w
        k = d / np.maximum(d - p[:, axis], 0.15)
        p = p[:, :axis] * k[:, None]
    if p.shape[1] == 3:
        return p, np.zeros(len(p), dtype=p.dtype)
    w = p[:, 3]
    k = d / np.maximum(d - w, 0.15)
    return p[:, :3] * k[:, None], w


def project_4d_to_3d(v4: np.ndarray, d4: float) -> tuple[np.ndarray, np.ndarray]:
    return project_nd_to_3d(v4, d4)


def project_3d_to_2d(p3: np.ndarray, d3: float) -> tuple[np.ndarray, np.ndarray]:
    """Perspective along z; returns 2D points (y up) and z."""
    z = p3[:, 2]
    k = d3 / np.maximum(d3 - z, 0.15)
    return p3[:, :2] * k[:, None], z


def slice_at_w(shape: Shape4D, v4: np.ndarray, w0: float):
    """
    Cross-section of the (rotated) N-D shape with the hyperplane w = w0 — what a lower-dimensional
    being sees as the object passes through its world. For a 4D shape the result is 3D; for 5D/6D
    it is 4D/5D (with the w axis removed) and gets projected further by the caller.

    Returns (seg_a, seg_b, points):
      seg_a, seg_b: (S, N-1) endpoints of the cross-section's edges (where a 2-face is cut),
      points:       (P, N-1) every point where an edge pierces the hyperplane.
    """
    e = shape.edges
    a, b = v4[e[:, 0]], v4[e[:, 1]]
    wa, wb = a[:, 3] - w0, b[:, 3] - w0
    crossing = (wa * wb) < 0
    t = np.where(crossing, wa / np.where(crossing, wa - wb, 1.0), 0.0)
    keep = [k for k in range(v4.shape[1]) if k != 3]               # every axis except w
    cut = a[:, keep] + (b[:, keep] - a[:, keep]) * t[:, None]
    points = cut[crossing]
    if shape.faces is None:
        empty = np.zeros((0, len(keep)), np.float32)
        return empty, empty, points

    F = shape.faces
    valid = F >= 0
    hit = np.zeros(F.shape, dtype=bool)
    hit[valid] = crossing[F[valid]]
    # A convex face crossed by the hyperplane is cut along a segment joining exactly 2 edge points.
    rows = hit.sum(axis=1) == 2
    Fs, Hs = F[rows], hit[rows]
    first_two = np.argsort(~Hs, axis=1, kind="stable")[:, :2]
    ea = np.take_along_axis(Fs, first_two[:, :1], axis=1)[:, 0]
    eb = np.take_along_axis(Fs, first_two[:, 1:], axis=1)[:, 0]
    return cut[ea], cut[eb], points


# ======================================================================================
# String figure: strings whose interior bends through 4D
# ======================================================================================
def string_lift(n_strings: int, u: np.ndarray, phase: float) -> np.ndarray:
    """
    4D displacement for every sample of every string: each string follows a (1, 2) torus knot
    wound around the Clifford torus (cos a, sin a, cos b, sin b)/√2, phase-shifted per string.
    Returns (n_strings, len(u), 4).
    """
    s = np.arange(n_strings, dtype=np.float32)[:, None]
    a = 2 * np.pi * u[None, :] + s * 0.45 + phase
    b = 4 * np.pi * u[None, :] + s * 0.3 - phase * 1.3
    return np.stack([np.cos(a), np.sin(a), np.cos(b), np.sin(b)], axis=-1).astype(np.float32) / np.sqrt(2)
