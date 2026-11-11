"""Shared post-processing for VASTO reconstruction runs.

Collects everything that is computed *after* (or alongside) an optimization so
that every notebook calls the same code:

  * boundary-condition diagnostics   -- `inlet_report`, `flux_check`
  * reference-design handling        -- `resample_design`
  * geometry error metrics           -- `geometry_errors`
  * flow error metrics               -- `flow_errors`
  * objective floor / self-consistency -- `floor_test`
  * cost instrumentation             -- `EpochTimer`, `cost_report`
  * run bookkeeping                  -- `RunRecord`

Nothing here imports the notebook's globals. Every function takes what it needs
as an argument, so the same module works for the bifurcation, stenosis and
carotid cases and at any mesh resolution.

Typical use in a notebook::

    import src.postprocessing as _post

    _post.inlet_report(_fluid_mesh, inlet_top_faces, geom.edges[3], "inlet_top")
    ...
    rec = _post.RunRecord(case="bifurcation", nelx=nelx_desired, gt_file=sngrm_file_import)
    timer = _post.EpochTimer()                      # inside optimize_design
    ...
    rec.add(_post.geometry_errors(gamma_rec, gamma_true, dens_filter, beta=4.0))
    rec.add(_post.flow_errors(...))
    rec.add(timer.report(mesh=_fluid_mesh, num_time_steps=num_time_steps))
    rec.save("results/bifurcation_n80.json")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

try:  # jax is optional for the pure-numpy helpers
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None

try:
    import psutil

    _PROC = psutil.Process(os.getpid())
except ImportError:  # pragma: no cover
    psutil = None
    _PROC = None


# ---------------------------------------------------------------------------
# small mesh helpers
# ---------------------------------------------------------------------------
def face_node_coords(mesh, elem: int, face: int) -> np.ndarray:
    """Physical coordinates of the nodes of one element face.

    Args:
      mesh: The mesh object.
      elem: Global element index.
      face: Local face index within the element.

    Returns: Array of (nodes_per_face, num_dim).
    """
    fc = np.asarray(mesh.elem_template.face_connectivity)
    gids = np.asarray(mesh.elem_nodes)[elem][fc[face]]
    return np.asarray(mesh.nodes.coords)[gids]


def elem_velocity(press_vel, mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Element-averaged velocity from the packed (p, u, v) dof vector.

    Args:
      press_vel: Array of (num_dofs,) ordered as (p, u, v) per node.
      mesh: The fluid mesh.

    Returns: Tuple of (u, v), each of shape (num_elems,).
    """
    pv = np.asarray(press_vel)[np.asarray(mesh.elem_dof_mat)]
    return pv[:, 1::3].mean(axis=1), pv[:, 2::3].mean(axis=1)


def elem_pressure(press_vel, mesh) -> np.ndarray:
    """Element-averaged pressure from the packed (p, u, v) dof vector."""
    pv = np.asarray(press_vel)[np.asarray(mesh.elem_dof_mat)]
    return pv[:, 0::3].mean(axis=1)


# ---------------------------------------------------------------------------
# nodal flow field: save and evaluate on another mesh
# ---------------------------------------------------------------------------
def save_nodal_flow(path: str, press_vel, mesh, grid_bounds, nel: Tuple[int, int],
                    **extra) -> None:
    """Save the nodal (p, u, v) field of a `grid_mesh_brep` mesh for re-interpolation.

    Args:
      path: Output .npz file.
      press_vel: Array of (num_dofs,) ordered as (p, u, v) per node.
      mesh: The fluid mesh from `grid_mesh_brep`.
      grid_bounds: (min_x, min_y, max_x, max_y) of the background grid, i.e.
        `geom.geometry.bounds` that was passed to `grid_mesh_brep`.
      nel: (nelx, nely) of the background grid.
      **extra: Any additional arrays to store.
    """
    pv = np.asarray(press_vel).reshape(-1, 3)
    np.savez_compressed(
        path,
        pressure=pv[:, 0],
        u_vel=pv[:, 1],
        v_vel=pv[:, 2],
        node_coords=np.asarray(mesh.nodes.coords),
        elem_nodes=np.asarray(mesh.elem_nodes),
        grid_bounds=np.asarray(grid_bounds, float),
        nel=np.asarray(nel, int),
        **extra,
    )


def save_nodal_conc(path: str, conc_hist, mesh, grid_bounds, nel: Tuple[int, int],
                    **extra) -> None:
    """Save a nodal concentration history of a `grid_mesh_brep` mesh for re-interpolation.

    Args:
      path: Output .npz file.
      conc_hist: Array of (num_time_steps, num_nodes) concentration (1 dof per node).
      mesh: The concentration mesh from `grid_mesh_brep`.
      grid_bounds: (min_x, min_y, max_x, max_y) of the background grid.
      nel: (nelx, nely) of the background grid.
      **extra: Any additional arrays to store (e.g. times).
    """
    np.savez_compressed(
        path,
        concentration=np.asarray(conc_hist).T,  # (num_nodes, num_time_steps)
        node_coords=np.asarray(mesh.nodes.coords),
        elem_nodes=np.asarray(mesh.elem_nodes),
        grid_bounds=np.asarray(grid_bounds, float),
        nel=np.asarray(nel, int),
        **extra,
    )


def interp_nodal_flow(data, points,
                      keys: Sequence[str] = ("pressure", "u_vel", "v_vel")
                      ) -> Dict[str, np.ndarray]:
    """Evaluate saved nodal fields at arbitrary points with the Q1 shape functions.

    The source mesh is a structured grid with some cells removed, so the element
    containing a point is found by index arithmetic. Points that fall in a removed cell
    or outside the grid get NaN.

    Args:
      data: Path to, or the loaded dict of, a file written by `save_nodal_flow` or
        `save_nodal_conc`.
      points: Array of (num_points, 2) physical coordinates.
      keys: Nodal fields to evaluate. Each is (num_nodes, ...); trailing axes (e.g.
        time) are carried through. Use ("concentration",) for `save_nodal_conc` files.

    Returns: Dict with each key as (num_points, ...) and 'inside' (bool mask of points
      that landed in an element).
    """
    if isinstance(data, str):
        data = dict(np.load(data))
    pts = np.asarray(points, float).reshape(-1, 2)
    x0, y0, x1, y1 = data["grid_bounds"]
    nelx, nely = (int(n) for n in data["nel"])
    dx, dy = (x1 - x0) / nelx, (y1 - y0) / nely
    node_coords, elem_nodes = data["node_coords"], data["elem_nodes"]

    # grid cell (iy, ix) -> element index, -1 for removed cells
    bl = node_coords[elem_nodes[:, 0]]
    lut = -np.ones((nely, nelx), dtype=int)
    lut[np.rint((bl[:, 1] - y0) / dy).astype(int),
        np.rint((bl[:, 0] - x0) / dx).astype(int)] = np.arange(len(elem_nodes))

    fx, fy = (pts[:, 0] - x0) / dx, (pts[:, 1] - y0) / dy
    tol = 1e-9
    in_box = (fx >= -tol) & (fx <= nelx + tol) & (fy >= -tol) & (fy <= nely + tol)
    ix = np.clip(np.floor(fx).astype(int), 0, nelx - 1)
    iy = np.clip(np.floor(fy).astype(int), 0, nely - 1)
    elem = np.where(in_box, lut[iy, ix], -1)
    inside = elem >= 0

    # Rect4 local nodes are counter-clockwise from bottom-left, see element.Quad4
    xi, eta = 2.0 * (fx - ix) - 1.0, 2.0 * (fy - iy) - 1.0
    shp = 0.25 * np.stack([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                           (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)], axis=1)
    conn = elem_nodes[np.where(inside, elem, 0)]

    out = {"inside": inside}
    for key in keys:
        vals = np.einsum("pn, pn... -> p...", shp, data[key][conn])
        mask = inside.reshape((-1,) + (1,) * (vals.ndim - 1))
        out[key] = np.where(mask, vals, np.nan)
    return out


# ---------------------------------------------------------------------------
# mesh-convergence comparison: coarse mesh vs a finer reference mesh
# ---------------------------------------------------------------------------
def _fluid_node_mask(node_coords, elem_nodes, design) -> np.ndarray:
    """Nodes touching at least one fluid element (design < 0.5) of the coarse mesh."""
    design = np.asarray(design).reshape(-1)
    if design.size != len(elem_nodes):
        raise ValueError(f"design has {design.size} entries but the coarse mesh has "
                         f"{len(elem_nodes)} elements; pass the coarse-mesh design")
    mask = np.zeros(len(node_coords), bool)
    mask[np.unique(elem_nodes[design < 0.5])] = True
    return mask


def _quad_triangulation(node_coords, elem_nodes):
    """Split each Rect4 element into two triangles for tripcolor."""
    import matplotlib.tri as mtri
    tris = np.vstack([elem_nodes[:, [0, 1, 2]], elem_nodes[:, [0, 2, 3]]])
    return mtri.Triangulation(node_coords[:, 0], node_coords[:, 1], tris), tris


def time_axis(num_states: int, time_step_size) -> np.ndarray:
    """Physical times of a stored history, with or without the initial state."""
    dt = np.asarray(time_step_size).reshape(-1)
    if num_states == len(dt) + 1:
        return np.concatenate([[0.0], np.cumsum(dt)])
    if num_states == len(dt):
        return np.cumsum(dt)
    print(f"warning: {num_states} states vs {len(dt)} time steps, using step index")
    return np.arange(num_states, dtype=float)


def mesh_velocity_errors(coarse_path: str, ref_path: str, coarse_design,
                         labels: Tuple[str, str] = ("coarse", "ref"),
                         verbose: bool = True) -> Dict:
    """Velocity of a coarse mesh vs a finer reference, evaluated at the coarse nodes.

    Args:
      coarse_path: `save_nodal_flow` file of the coarse mesh.
      ref_path: `save_nodal_flow` file of the reference (finer) mesh.
      coarse_design: Element design of the coarse mesh, (num_coarse_elems,). Fluid is
        design < 0.5.
      labels: Names used in the printout and plots, e.g. ("120", "160").

    Returns: Dict with the metrics and the nodal arrays needed for plotting.
    """
    gt = dict(np.load(coarse_path))
    q = interp_nodal_flow(ref_path, gt["node_coords"])
    ok = q["inside"]
    fl = ok & _fluid_node_mask(gt["node_coords"], gt["elem_nodes"], coarse_design)

    du, dv = gt["u_vel"] - q["u_vel"], gt["v_vel"] - q["v_vel"]
    err_mag = np.sqrt(du**2 + dv**2)
    ref_mag = np.sqrt(q["u_vel"]**2 + q["v_vel"]**2)

    out = {
        "labels": labels,
        "rel_L2_all": float(np.linalg.norm(err_mag[ok]) / np.linalg.norm(ref_mag[ok])),
        "rel_L2_fluid": float(np.linalg.norm(err_mag[fl]) / np.linalg.norm(ref_mag[fl])),
        "max_abs": float(np.nanmax(err_mag[ok])),
        "num_inside": int(ok.sum()), "num_fluid": int(fl.sum()),
        "node_coords": gt["node_coords"], "elem_nodes": gt["elem_nodes"], "inside": ok,
        "speed_coarse": np.sqrt(gt["u_vel"]**2 + gt["v_vel"]**2), "speed_ref": ref_mag,
        "err_mag": err_mag,
    }
    if verbose:
        print(f"[velocity {labels[0]} vs {labels[1]}] nodes: {out['num_inside']} inside, "
              f"{out['num_fluid']} fluid")
        print(f"  rel L2 (all) = {out['rel_L2_all']:.3e}, rel L2 (fluid) = "
              f"{out['rel_L2_fluid']:.3e}, max |Δv| = {out['max_abs']:.3e}")
    return out


def plot_mesh_velocity(res: Dict, savepath: str | None = None):
    """Speed on both meshes and their difference (coarse − ref), from `mesh_velocity_errors`."""
    import matplotlib.pyplot as plt
    tri, tris = _quad_triangulation(res["node_coords"], res["elem_nodes"])
    tri.set_mask(~np.all(res["inside"][tris], axis=1))
    a, b = res["labels"]
    vc, vr = res["speed_coarse"], np.nan_to_num(res["speed_ref"])
    diff = np.nan_to_num(res["speed_coarse"] - res["speed_ref"])
    vmax, lim = max(vc.max(), vr.max()), np.abs(diff).max()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axi, f, t in [(ax[0], vc, a), (ax[1], vr, f"{b} (interp. to {a} nodes)")]:
        sc = axi.tripcolor(tri, f, vmin=0, vmax=vmax, shading="gouraud")
        axi.set_title(t); axi.set_aspect("equal")
    fig.colorbar(sc, ax=list(ax[:2]), label="|v|")
    sc = ax[2].tripcolor(tri, diff, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud")
    ax[2].set_title(f"{a} − {b}"); ax[2].set_aspect("equal")
    fig.colorbar(sc, ax=ax[2], label="Δ|v|")
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()
    return fig


def mesh_conc_errors(coarse_path: str, ref_path: str, coarse_design, time_step_size,
                     labels: Tuple[str, str] = ("coarse", "ref"), floor: float = 0.01,
                     verbose: bool = True) -> Dict:
    """Concentration history of a coarse mesh vs a finer reference, at the coarse nodes.

    Args:
      coarse_path: `save_nodal_conc` file of the coarse mesh.
      ref_path: `save_nodal_conc` file of the reference (finer) mesh.
      coarse_design: Element design of the coarse mesh, (num_coarse_elems,).
      time_step_size: Array of the physical time step sizes (for the time axis).
      labels: Names used in the printout and plots.
      floor: Per-step relative errors use max(||c_ref||, floor * max_t ||c_ref||) as the
        denominator so near-empty early steps do not blow up.

    Returns: Dict with the metrics, the per-step error and arrays for plotting.
    """
    cc = dict(np.load(coarse_path))
    qc = interp_nodal_flow(ref_path, cc["node_coords"], keys=("concentration",))
    Cc, Cr = cc["concentration"], qc["concentration"]
    if Cc.shape != Cr.shape:
        raise ValueError(f"history shapes differ: {Cc.shape} vs {Cr.shape}")
    ok = qc["inside"]
    fl = ok & _fluid_node_mask(cc["node_coords"], cc["elem_nodes"], coarse_design)

    dC = Cc - Cr
    ref_t = np.linalg.norm(Cr[fl], axis=0)
    rel_t = np.linalg.norm(dC[fl], axis=0) / np.maximum(ref_t, floor * ref_t.max())
    out = {
        "labels": labels,
        "rel_L2_all": float(np.linalg.norm(dC[ok]) / np.linalg.norm(Cr[ok])),
        "rel_L2_fluid": float(np.linalg.norm(dC[fl]) / np.linalg.norm(Cr[fl])),
        "max_abs": float(np.nanmax(np.abs(dC[ok]))),
        "rel_t": rel_t, "times": time_axis(Cc.shape[1], time_step_size),
        "k_peak": int(np.argmax(ref_t)),
        "node_coords": cc["node_coords"], "elem_nodes": cc["elem_nodes"], "inside": ok,
        "C_coarse": Cc, "C_ref": Cr,
    }
    if verbose:
        r = rel_t[1:]
        print(f"[concentration {labels[0]} vs {labels[1]}] {Cc.shape[1]} states")
        print(f"  rel L2 (all) = {out['rel_L2_all']:.3e}, rel L2 (fluid) = "
              f"{out['rel_L2_fluid']:.3e}, max |Δc| = {out['max_abs']:.3e}")
        print(f"  per-step rel L2 (fluid, excl. t=0): mean = {r.mean():.3e}, "
              f"max = {r.max():.3e} at step {1 + r.argmax()}")
    return out


def plot_mesh_conc(res: Dict, step: int | None = None, savepath: str | None = None):
    """Per-step error curve and fields at one step (default: peak), from `mesh_conc_errors`."""
    import matplotlib.pyplot as plt
    a, b = res["labels"]
    t = res["times"]

    fig0, ax0 = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax0.semilogy(t[1:], res["rel_t"][1:])
    ax0.set_xlabel("t [s]"); ax0.set_ylabel("rel L2 (fluid)")
    ax0.set_title(f"{a} vs {b} concentration error over time")

    k = res["k_peak"] if step is None else step
    tri, tris = _quad_triangulation(res["node_coords"], res["elem_nodes"])
    tri.set_mask(~np.all(res["inside"][tris], axis=1))
    cc, cr = res["C_coarse"][:, k], np.nan_to_num(res["C_ref"][:, k])
    d = np.nan_to_num(res["C_coarse"][:, k] - res["C_ref"][:, k])
    vmax, lim = max(cc.max(), cr.max()), max(np.abs(d).max(), 1e-30)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axi, f, ttl in [(ax[0], cc, a), (ax[1], cr, f"{b} (interp. to {a} nodes)")]:
        sc = axi.tripcolor(tri, f, vmin=0, vmax=vmax, shading="gouraud")
        axi.set_title(f"{ttl}, t={t[k]:.3f}s"); axi.set_aspect("equal")
    fig.colorbar(sc, ax=list(ax[:2]), label="c")
    sc = ax[2].tripcolor(tri, d, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud")
    ax[2].set_title(f"{a} − {b}"); ax[2].set_aspect("equal")
    fig.colorbar(sc, ax=ax[2], label="Δc")
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()
    return fig0, fig


def mesh_sinogram_errors(S_coarse, S_ref, time_step_size=None,
                         labels: Tuple[str, str] = ("coarse", "ref"),
                         theta_stride: int = 2, floor: float = 0.01,
                         verbose: bool = True) -> Dict:
    """Sinogram of a coarse mesh vs a finer reference.

    Args:
      S_coarse, S_ref: Arrays (or .npy paths) of (frames, n_det, n_theta).
      time_step_size: Physical step sizes for the time axis; frame index if None.
      labels: Names used in the printout and plots.
      theta_stride: Angle subset used by the loss (every `theta_stride`-th angle).
      floor: Floor (fraction of the peak) for per-frame normalisation and frame weights,
        matching `objective_function`.

    Returns: Dict with the metrics, the per-frame and per-angle errors and the arrays.
    """
    Sc = np.load(S_coarse) if isinstance(S_coarse, str) else np.asarray(S_coarse)
    Sr = np.load(S_ref) if isinstance(S_ref, str) else np.asarray(S_ref)
    if Sc.shape != Sr.shape:
        raise ValueError(f"sinogram shapes differ: {Sc.shape} vs {Sr.shape}")
    dS = Sc - Sr

    ref_f = np.linalg.norm(Sr, axis=(1, 2))
    rel_f = np.linalg.norm(dS, axis=(1, 2)) / np.maximum(ref_f, floor * ref_f.max())

    th = np.arange(0, Sr.shape[2], theta_stride)
    energy = np.mean(Sr[:, :, th]**2, axis=(1, 2))
    W = 1.0 / np.maximum(energy, floor * energy.max())
    out = {
        "labels": labels,
        "norm_ratio": float(np.linalg.norm(Sc) / np.linalg.norm(Sr)),
        "rel_L2": float(np.linalg.norm(dS) / np.linalg.norm(Sr)),
        "max_abs": float(np.abs(dS).max()),
        "max_abs_frac": float(np.abs(dS).max() / np.abs(Sr).max()),
        "J_mesh": float(np.mean(dS[:, :, th]**2 * W[:, None, None])),
        "rel_f": rel_f,
        "rel_theta": np.linalg.norm(dS, axis=(0, 1)) / np.linalg.norm(Sr, axis=(0, 1)),
        "times": (np.arange(Sc.shape[0], dtype=float) if time_step_size is None
                  else time_axis(Sc.shape[0], time_step_size)),
        "S_coarse": Sc, "S_ref": Sr,
    }
    if verbose:
        r = rel_f[1:]
        print(f"[sinogram {labels[0]} vs {labels[1]}] norm ratio = {out['norm_ratio']:.5f}")
        print(f"  rel L2 = {out['rel_L2']:.3e}, max |ΔS| = {out['max_abs']:.3e} "
              f"({out['max_abs_frac']:.2%} of max S)")
        print(f"  per-frame rel L2 (excl. frame 0): mean = {r.mean():.3e}, "
              f"max = {r.max():.3e} at frame {1 + r.argmax()}")
        print(f"  loss of the {labels[0]} sinogram against the {labels[1]} GT: "
              f"J = {out['J_mesh']:.3e}")
    return out


def plot_mesh_sinogram(res: Dict, frame: int | None = None, savepath: str | None = None):
    """Per-frame / per-angle errors and one frame (default: worst), from `mesh_sinogram_errors`."""
    import matplotlib.pyplot as plt
    a, b = res["labels"]
    Sc, Sr, t = res["S_coarse"], res["S_ref"], res["times"]

    fig0, ax0 = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
    ax0[0].plot(t[1:], res["rel_f"][1:]); ax0[0].set_ylabel("rel L2")
    ax0[0].set_xlabel("t [s]" if t[-1] != len(t) - 1 else "frame"); ax0[0].set_title("per frame")
    ax0[1].plot(np.linspace(0, 180, Sr.shape[2], endpoint=False), res["rel_theta"])
    ax0[1].set_xlabel("θ [deg]"); ax0[1].set_title("per angle")

    k = 1 + int(np.argmax(res["rel_f"][1:])) if frame is None else frame
    d = Sc[k] - Sr[k]
    lim = max(np.abs(d).max(), 1e-30)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axi, S, ttl in [(ax[0], Sc, a), (ax[1], Sr, b)]:
        sc = axi.imshow(S[k], aspect="auto", vmin=0, vmax=Sr[k].max())
        axi.set_title(f"{ttl}, frame {k}")
    fig.colorbar(sc, ax=list(ax[:2]))
    sc = ax[2].imshow(d, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[2].set_title(f"{a} − {b}")
    fig.colorbar(sc, ax=ax[2])
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()
    return fig0, fig


# ---------------------------------------------------------------------------
# boundary-condition diagnostics
# ---------------------------------------------------------------------------
def inlet_report(mesh, faces: Sequence[Tuple[int, int]], edge, label: str = "",
                 verbose: bool = False) -> Dict:
    """Report the discretised span of a set of boundary faces against its brep edge.

    The selected inlet length is quantised to the mesh, so it can differ from the
    brep edge length and from one resolution to another. That difference changes
    the injected flux and is absorbed by the inlet scale unless corrected, so it
    is worth printing on every run.

    Args:
      mesh: The mesh object.
      faces: List of (elem, face) returned by the face-selection routine.
      edge: Array of (2, num_dim) with the brep edge start and end points.
      label: Name used in the printout.
      verbose: If True, print every node coordinate.

    Returns: Dict with nf, L_sel, L_edge, rel_err, centre and the node coords.
    """
    p0, p1 = np.asarray(edge[0], float), np.asarray(edge[1], float)
    L_edge = float(np.linalg.norm(p1 - p0))
    t_hat = (p1 - p0) / L_edge

    xy_all = np.vstack([face_node_coords(mesh, e, f) for e, f in faces])
    xy = np.unique(np.round(xy_all, 12), axis=0)
    t = (xy - p0) @ t_hat
    order = np.argsort(t)
    xy, t = xy[order], t[order]

    L_sel = float(t.max() - t.min())
    gaps = np.diff(t)
    ok_contig = len(xy) == len(faces) + 1
    ok_even = bool(gaps.size and np.ptp(gaps) < 1e-9)

    face_t = np.array([(face_node_coords(mesh, e, f) - p0) @ t_hat for e, f in faces]).mean(axis=1)
    ok_sorted = bool(np.all(np.diff(face_t) > 0))

    print(
        f"{label:10s} nf={len(faces):3d}  L_sel={L_sel:.6f}  L_edge={L_edge:.6f}  "
        f"err={L_sel / L_edge - 1:+.4%}  centre={0.5 * (t.min() + t.max()):.6f}"
    )
    if not ok_contig:
        print(f"  !! NOT CONTIGUOUS: {len(faces)} faces -> expected {len(faces)+1} nodes, got {len(xy)}")
    if gaps.size and not ok_even:
        print(f"  !! UNEVEN SPACING: gaps {gaps.min():.6e} .. {gaps.max():.6e}")
    if not ok_sorted:
        print("  !! FACES NOT SORTED ALONG THE EDGE -- the inlet profile will be scrambled")
    if verbose:
        for k, (xyk, tk) in enumerate(zip(xy, t)):
            print(f"   {k:>3} x={xyk[0]:>10.6f} y={xyk[1]:>10.6f}  (t-t0)/L={(tk - t.min()) / L_sel:>8.5f}")

    return dict(label=label, nf=len(faces), L_sel=L_sel, L_edge=L_edge,
                rel_err=L_sel / L_edge - 1.0, centre=0.5 * (t.min() + t.max()),
                contiguous=ok_contig, even=ok_even, sorted=ok_sorted,
                node_coords=xy)

# ---------------------------------------------------------------------------
# reference design
# ---------------------------------------------------------------------------
def resample_design(gamma_src, n_target: int) -> np.ndarray:
    """Bilinearly resample a square design-variable field to another resolution.

    Always resample from the single canonical source array. Chaining resamples
    (80 -> 160 -> 60) compounds interpolation error and biases the coarse meshes.

    Args:
      gamma_src: Array of (n*n,) or (n, n) design variables.
      n_target: Target number of elements per side.

    Returns: Array of (n_target*n_target, 1).
    """
    g = np.asarray(gamma_src).reshape(-1)
    n = int(round(np.sqrt(g.size)))
    if n * n != g.size:
        raise ValueError(f"source design is not square: size {g.size}")
    g = g.reshape(n, n)
    if n == n_target:
        return g.reshape(-1, 1)
    if jax is None:
        raise ImportError("jax is required for resample_design")
    out = jax.image.resize(jnp.asarray(g), (n_target, n_target), method="linear")
    return np.asarray(out).reshape(-1, 1)


# ---------------------------------------------------------------------------
# geometry error metrics
# ---------------------------------------------------------------------------
def geometry_errors(gamma_rec, gamma_true, dens_filter, threshold_filter_fn: Callable,
                    beta: float = 4.0, thr: float = 0.5, label: str = "") -> Dict:
    """Dice, nRMSE and volume-fraction error between a reconstruction and the truth.

    Both fields are pushed through the same filter and projection so the
    comparison is between geometries rather than raw design variables. Use the
    same `beta` that defined the ground-truth geometry -- scoring at a different
    beta measures a different shape.

    Args:
      gamma_rec: Array of (num_elems,) or (num_elems, 1) reconstructed design vars.
      gamma_true: Same shape, the reference design resampled to this mesh.
      dens_filter: The sparse density filter for this mesh.
      threshold_filter_fn: Callable (field, beta) -> projected field.
      beta: Projection sharpness; must match the value used for data generation.
      thr: Fluid/solid cut-off applied to the projected field.
      label: Name used in the printout.

    Returns: Dict of the metrics.
    """
    gr = jnp.asarray(np.asarray(gamma_rec).reshape(-1))
    gt = jnp.asarray(np.asarray(gamma_true).reshape(-1))

    mf_out = np.asarray(threshold_filter_fn(dens_filter @ gr, beta=beta)).reshape(-1)
    mf_true = np.asarray(threshold_filter_fn(dens_filter @ gt, beta=beta)).reshape(-1)

    num = np.sqrt(np.sum((mf_out - mf_true) ** 2))
    den = np.sqrt(np.sum(mf_true ** 2)) + 1e-12
    nrmse = float(num / den)

    fluid_out, fluid_true = mf_out <= thr, mf_true <= thr
    tp = int(np.sum(fluid_out & fluid_true))
    fp = int(np.sum(fluid_out & ~fluid_true))
    fn = int(np.sum(~fluid_out & fluid_true))
    dice = float(2.0 * tp / (2.0 * tp + fp + fn + 1e-12))

    vf_out, vf_true = float(mf_out.mean()), float(mf_true.mean())

    print(f"--- geometry {label} ---")
    print(f"  Dice                   : {dice:.6f}")
    print(f"  nRMSE (relative L2)    : {nrmse:.6f}  ({nrmse:.2%})")
    print(f"  volume fraction rec/true: {vf_out:.6f} / {vf_true:.6f}  (delta {vf_out - vf_true:+.6f})")

    return dict(dice=dice, nrmse=nrmse, volfrac_rec=vf_out, volfrac_true=vf_true,
                d_volfrac=vf_out - vf_true, beta=beta,
                mat_frac_rec=mf_out, mat_frac_true=mf_true)


# ---------------------------------------------------------------------------
# flow error metrics
# ---------------------------------------------------------------------------
def flow_errors(press_vel_rec, press_vel_true, mesh, fluid_mask,
                outlet_faces: Sequence[Tuple[int, int]] | None = None,
                a_rec=None, a_true=None, label: str = "") -> Dict:
    """Velocity, pressure and flux errors, restricted to the true lumen.

    The reference field is recomputed on the same mesh as the reconstruction, so
    these are reconstruction errors and not discretisation errors. State that in
    the paper -- the reference peak speed changes slightly between resolutions.

    Note the outlet flux error and the inlet-scale error are not independent:
    mass conservation ties them together. Report the flux because it is the
    clinically meaningful quantity, but do not present it as an independent check.

    Args:
      press_vel_rec: Reconstructed (p, u, v) dof vector.
      press_vel_true: Reference (p, u, v) dof vector on the same mesh.
      mesh: The fluid mesh.
      fluid_mask: Boolean array of (num_elems,), True inside the true lumen.
      outlet_faces: Optional list of (elem, face) for the outlet flux integral.
      a_rec: Optional recovered inlet scales.
      a_true: Optional true inlet scales.
      label: Name used in the printout.

    Returns: Dict of the metrics.
    """
    ur, vr = elem_velocity(press_vel_rec, mesh)
    ut, vt = elem_velocity(press_vel_true, mesh)
    m = np.asarray(fluid_mask).reshape(-1).astype(bool)

    du, dv = ur[m] - ut[m], vr[m] - vt[m]
    l2 = float(np.sqrt((du ** 2 + dv ** 2).sum()) /
               (np.sqrt((ut[m] ** 2 + vt[m] ** 2).sum()) + 1e-30))

    sp_r = np.sqrt(ur[m] ** 2 + vr[m] ** 2)
    sp_t = np.sqrt(ut[m] ** 2 + vt[m] ** 2)
    peak_err = float(sp_r.max() / sp_t.max() - 1.0)
    mean_err = float(sp_r.mean() / sp_t.mean() - 1.0)

    pr, pt = elem_pressure(press_vel_rec, mesh), elem_pressure(press_vel_true, mesh)
    dp_r = float(pr[m].max() - pr[m].min())
    dp_t = float(pt[m].max() - pt[m].min())
    dp_err = dp_r / dp_t - 1.0 if dp_t != 0 else float("nan")

    out = dict(vel_l2=l2, peak_err=peak_err, mean_err=mean_err,
               peak_true=float(sp_t.max()), peak_rec=float(sp_r.max()),
               dp_true=dp_t, dp_rec=dp_r, dp_err=dp_err)

    print(f"--- flow {label} ---")
    print(f"  velocity relative L2 (lumen) : {l2:.4%}")
    print(f"  peak speed true/rec          : {sp_t.max():.6f} / {sp_r.max():.6f}  ({peak_err:+.3%})")
    print(f"  mean speed true/rec          : {sp_t.mean():.6f} / {sp_r.mean():.6f}  ({mean_err:+.3%})")
    print(f"  pressure drop true/rec       : {dp_t:.6f} / {dp_r:.6f}  ({dp_err:+.3%})")

    if outlet_faces is not None:
        def _flux(pv):
            pvn = np.asarray(pv)
            Q = 0.0
            for e, f in outlet_faces:
                fc = np.asarray(mesh.elem_template.face_connectivity)
                gids = np.asarray(mesh.elem_nodes)[e][fc[f]]
                xy = np.asarray(mesh.nodes.coords)[gids]
                uu = pvn[3 * gids + 1]
                Q += 0.5 * (uu[0] + uu[1]) * abs(xy[1, 1] - xy[0, 1])
            return float(Q)

        Qr, Qt = _flux(press_vel_rec), _flux(press_vel_true)
        out.update(flux_true=Qt, flux_rec=Qr, flux_err=Qr / Qt - 1.0 if Qt != 0 else float("nan"))
        print(f"  outlet flux true/rec         : {Qt:.6f} / {Qr:.6f}  ({out['flux_err']:+.3%})")

    if a_rec is not None and a_true is not None:
        ar, at = np.asarray(a_rec, float).ravel(), np.asarray(a_true, float).ravel()
        out.update(a_rec=ar.tolist(), a_true=at.tolist(),
                   a_err=(ar / at - 1.0).tolist(),
                   a_err_mean=float(np.mean(ar / at - 1.0)))
        print(f"  inlet scale true/rec         : {at} / {ar}  ({out['a_err_mean']:+.3%} mean)")

    return out


# ---------------------------------------------------------------------------
# objective floor / self-consistency
# ---------------------------------------------------------------------------
def floor_test(objective_fn: Callable, gamma_true, a_true, flow_solver, press_vel_init,
               ramp_penalty: float = 50.0, diffusion_ramp_penalty: float = 10.0,
               thresh_beta: float = 4.0, data_scale: float | None = None,
               label: str = "") -> Dict:
    """Evaluate the objective at the known answer -- the best score achievable.

    Read it three ways:
      * floor ~ 0 and the optimizer well above it  -> search failure, fix the optimizer
      * floor ~ optimizer value                    -> the optimizer is done; the rest is modelling
      * optimizer *below* the floor                -> the objective's minimum is not at the truth

    On the generating mesh this must return exactly 0. Anywhere else the floor is
    an upper bound, because a resampled reference is not the best design that
    mesh can represent.

    Args:
      objective_fn: The notebook's `objective_function`.
      gamma_true: Reference design resampled to this mesh, shape (num_elems, 1).
      a_true: True inlet scales in NORMALISED design-variable units.
      flow_solver: The fluid solver instance.
      press_vel_init: Initial pressure-velocity guess.
      ramp_penalty, diffusion_ramp_penalty, thresh_beta: Forward-model settings;
        must match those used for data generation.
      data_scale: Optional mean(S**2) for reporting a relative residual.
      label: Name used in the printout.

    Returns: Dict with the objective and, if available, the relative residual.
    """
    x = np.vstack([np.asarray(gamma_true).reshape(-1, 1),
                   np.asarray(a_true, float).reshape(-1, 1)])
    out = objective_fn(jnp.asarray(x), flow_solver, ramp_penalty,
                       diffusion_ramp_penalty, thresh_beta, press_vel_init)
    obj = float(out[0])
    res = dict(floor_J=obj)
    msg = f"floor {label}: J = {obj:.6e}"
    if data_scale:
        res["floor_rel"] = obj / data_scale
        msg += f"   rel = {obj / data_scale:.6e}"
    print(msg)
    return res


# ---------------------------------------------------------------------------
# cost instrumentation
# ---------------------------------------------------------------------------
def memory_mb() -> Tuple[float, float]:
    """Current RSS and peak RSS in MB."""
    import resource, sys
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    peak = ru / 1e6 if sys.platform == "darwin" else ru / 1e3
    if jax is not None:
        try:
            st = jax.local_devices()[0].memory_stats()   # None on CPU
            return st["bytes_in_use"] / 1e6, max(st["peak_bytes_in_use"] / 1e6, peak)
        except Exception:
            pass
    cur = _PROC.memory_info().rss / 1e6 if _PROC is not None else float("nan")
    return cur, peak

class EpochTimer:
    """Per-epoch wall-clock timing for the optimization loop.

    JAX dispatches asynchronously, so `tick` must be called after the result has
    been forced, otherwise you time the dispatch and not the compute. Pass the
    objective array to `tick` and it will block for you.

    Usage inside `optimize_design`::

        timer = _post.EpochTimer()          # before the loop
        ...
        timer.start()                        # top of each iteration
        objective, grad, ... = objective_function(...)
        dt = timer.tick(objective)           # blocks, records, returns seconds
        ...
        timer.report(mesh=fe.mesh, num_time_steps=num_time_steps)
    """

    def __init__(self) -> None:
        self.dt: List[float] = []
        self._t0 = time.perf_counter()
        self._tick = None

    def start(self) -> None:
        self._tick = time.perf_counter()

    def tick(self, blocking_on=None) -> float:
        if blocking_on is not None and hasattr(blocking_on, "block_until_ready"):
            blocking_on.block_until_ready()
        now = time.perf_counter()
        dt = now - (self._tick if self._tick is not None else now)
        self.dt.append(dt)
        self._tick = now
        return dt

    def report(self, mesh=None, num_time_steps: int | None = None,
               nr_flow: int = 4, nr_transport: int = 3, label: str = "") -> Dict:
        """Print and return the cost summary for the run."""
        dt = np.asarray(self.dt, float)
        cur, peak = memory_mb()
        total = time.perf_counter() - self._t0
        steady = dt[1:] if dt.size > 1 else dt

        res = dict(epochs=int(dt.size), t_total_s=float(total),
                   t_first_epoch_s=float(dt[0]) if dt.size else float("nan"),
                   t_epoch_mean_s=float(steady.mean()) if steady.size else float("nan"),
                   t_epoch_median_s=float(np.median(steady)) if steady.size else float("nan"),
                   mem_current_mb=cur, mem_peak_mb=peak)

        print(f"--- cost {label} ---")
        if mesh is not None:
            res.update(num_elems=int(mesh.num_elems), num_dofs=int(mesh.num_dofs))
            print(f"  elements / flow dofs   : {mesh.num_elems} / {mesh.num_dofs}")
        print(f"  epochs                 : {res['epochs']}")
        print(f"  first epoch (incl jit) : {res['t_first_epoch_s']:.2f} s")
        print(f"  per epoch mean/median  : {res['t_epoch_mean_s']:.3f} / {res['t_epoch_median_s']:.3f} s")
        print(f"  total wall clock       : {total:.1f} s  ({total / 60:.1f} min)")
        print(f"  memory current/peak    : {cur:.0f} / {peak:.0f} MB")

        if num_time_steps is not None:
            fwd = nr_flow + num_time_steps * nr_transport
            res.update(linear_solves_fwd=int(fwd), linear_solves_epoch=int(2 * fwd))
            print(f"  linear solves / epoch  : ~{fwd} forward "
                  f"(1 NS x ~{nr_flow} NR + {num_time_steps} transport x ~{nr_transport} NR), "
                  f"~{2 * fwd} incl. reverse pass")
        return res


def cost_report(*args, **kwargs) -> Dict:
    """Backwards-compatible alias for `EpochTimer.report`."""
    return EpochTimer().report(*args, **kwargs)


# ---------------------------------------------------------------------------
# run bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    """Collects everything about one run so results can be tabulated later.

    Numpy arrays are dropped on save; only scalars and small lists are kept.
    """

    case: str = ""
    nelx: int = 0
    gt_file: str = ""
    notes: str = ""
    data: Dict = field(default_factory=dict)

    def add(self, d: Dict | None = None, **kw) -> "RunRecord":
        if d:
            self.data.update({k: v for k, v in d.items() if not isinstance(v, np.ndarray)})
        self.data.update(kw)
        return self

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = dict(case=self.case, nelx=self.nelx, gt_file=self.gt_file,
                       notes=self.notes,
                       timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), **self.data)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=float)
        print(f"saved -> {path}")
        return path


def load_table(paths: Sequence[str], columns: Sequence[str] | None = None):
    """Load several `RunRecord` json files into a table for the paper.

    Args:
      paths: Paths to the saved json files.
      columns: Optional subset of keys, in the order you want them.

    Returns: A pandas DataFrame if pandas is available, else a list of dicts.
    """
    rows = []
    for p in paths:
        with open(p) as fh:
            rows.append(json.load(fh))
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        return df[list(columns)] if columns else df
    except ImportError:  # pragma: no cover
        return rows

def inlet_scale_report(a_init, a_rec, a_true=None, label: str = "") -> Dict:
    """Print the initial, recovered and true inlet scales in physical units.

    Args:
      a_init: Initial guess, physical units (pass `a0`, never a hardcoded value).
      a_rec: Recovered scales, physical units (a_min + x * a_ref).
      a_true: Optional true scales.
      label: Name used in the printout.

    Returns: Dict of the scales and, if `a_true` is given, the relative errors.
    """
    ai, ar = np.asarray(a_init, float).ravel(), np.asarray(a_rec, float).ravel()
    print(f"--- inlet scale {label} ---")
    print(f"  init      : {ai}")
    print(f"  optimized : {ar}")
    out = dict(a_init=ai.tolist(), a_rec=ar.tolist())
    if a_true is not None:
        at = np.asarray(a_true, float).ravel()
        err = ar / at - 1.0
        out.update(a_true=at.tolist(), a_err=err.tolist(), a_err_mean=float(err.mean()))
        print(f"  true      : {at}   error {err}  (mean {err.mean():+.3%})")
    return out


def plot_design(viz, mesh, field, ax=None, title: str = "", cmap=None,
                contour: bool = False, contour_level: float = 0.5,
                colorbar: bool = False, savepath: str | None = None):
    """Plot a projected material-fraction field on the mesh.

    Pass the PROJECTED field (`geometry_errors()["mat_frac_rec"]`), not the raw
    design variable, so the figure shows the same geometry the metrics score.

    Args:
      viz: The notebook's `src.viz` module (passed in to keep this file decoupled).
      mesh: The mesh to plot on.
      field: Array of (num_elems,) of the projected material fraction.
      ax: Optional existing axis.
      title: Axis title.
      cmap: Colormap; defaults to `viz.fluid_cmap`.
      contour: If True, draw a filled contour at `contour_level` instead of cells.
      colorbar: Whether to draw a colorbar.
      savepath: If given, save the figure there.

    Returns: The axis.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    f = np.asarray(field).reshape(-1)
    cmap = cmap if cmap is not None else getattr(viz, "fluid_cmap", None)

    if contour:
        ax = viz.plot_grid_contour(mesh=mesh, field=f, ax=ax, cmap=cmap,
                                   contour_level=contour_level, binary_fill=True,
                                   colorbar=colorbar)
    else:
        ax = viz.plot_grid_mesh(mesh=mesh, field=f, ax=ax, colorbar=colorbar,
                                cmap=cmap, val_range=(0.0, 1.0))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        ax.figure.savefig(savepath, bbox_inches="tight", pad_inches=0)
        print(f"saved -> {savepath}")
    return ax


def plot_comparison(viz, mesh, geo: Dict, label: str = "", savepath: str | None = None):
    """Side-by-side reconstruction / truth / difference, from a `geometry_errors` dict."""
    import matplotlib.pyplot as plt

    rec, true = geo["mat_frac_rec"], geo["mat_frac_true"]
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
    plot_design(viz, mesh, rec, ax=axs[0], title=f"recovered {label}")
    plot_design(viz, mesh, true, ax=axs[1], title="reference")
    d = np.abs(np.asarray(rec) - np.asarray(true))
    axs[2] = viz.plot_grid_mesh(mesh=mesh, field=d, ax=axs[2], colorbar=True)
    axs[2].set_xticks([]); axs[2].set_yticks([])
    axs[2].set_title(f"|diff|  Dice {geo['dice']:.4f}")
    fig.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"saved -> {savepath}")
    return fig, axs

def save_state(path: str, mma_state, history: Dict | None = None, **meta) -> str:
    """Save the design vector, convergence history and run metadata in one file.

    The optimization takes hours; post-processing takes seconds. Persisting the
    history means metrics, plots and cost tables can be regenerated after a
    kernel restart without re-optimizing.

    Args:
      path: Output path; `.npz` is appended if missing.
      mma_state: The final MMA state (uses `.x`).
      history: The history dict from `optimize_design`.
      **meta: Anything worth recording -- case, nelx, gt_file, a0, a_max,
        move_limit, beta schedule, eps_c, mu, D, and so on.

    Returns: The path written.
    """
    if not path.endswith(".npz"):
        path += ".npz"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    arrays = {"x": np.asarray(mma_state.x)}
    scalars = {}
    for k, v in (history or {}).items():
        if isinstance(v, dict):
            scalars[k] = v
        else:
            arr = np.asarray(v)
            if arr.dtype.kind in "fiu":
                arrays[f"hist_{k}"] = arr
            else:
                scalars[k] = v

    meta = dict(meta)
    meta.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    meta.update(scalars)
    arrays["_meta_json"] = np.asarray(json.dumps(meta, default=float))

    np.savez_compressed(path, **arrays)
    print(f"saved -> {path}  ({os.path.getsize(path)/1e3:.0f} kB, "
          f"{len(arrays)-2} history arrays)")
    return path


def load_state(path: str) -> Dict:
    """Load a file written by `save_state`.

    Returns: Dict with `x`, `history` (the hist_* arrays, keys stripped) and
      `meta` (everything else, including any cost dict).
    """
    if not path.endswith(".npz"):
        path += ".npz"
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["_meta_json"]))
    history = {k[5:]: z[k] for k in z.files if k.startswith("hist_")}
    print(f"loaded {path}: x{z['x'].shape}, history keys {list(history)}, "
          f"case={meta.get('case')} n={meta.get('nelx')}")
    return dict(x=z["x"], history=history, meta=meta)
