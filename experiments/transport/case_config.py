"""Per-geometry configuration and the generalised boundary-condition builders for the inversion template and the
GT generator (bifurcation, stenosis, carotid).

All inlets lie on x = 0. An inlet shift moves every inlet by `shift_elems` elements in y; the wall segments on
x = 0 are always the complement of the (shifted) inlets on [0, 1], so every boundary face carries exactly one
condition. Other walls and the outlets come from the brep edge indices. For the bifurcation, the builder performs the
same identify_faces calls with the same edges in the same order as the template's earlier make_fluid_bc, so its
output is identical.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

# pulses and timing are the bifurcation protocol for every case (T = 3 s; raised-cosine ramps of 0.05 s from src)
_PROTOCOL = dict(pulse_start=[0.025, 1.525], pulse_end=[0.525, 2.025], t_end_s=3.0)

CASES = {
    "bifurcation": dict(
        brep="./brep/artery_no_ext.json",
        inlets={"top": 3, "btm": 1},               # name -> brep edge on x = 0 (order = inlet-scale order)
        char_length_inlet="top",                   # L = width of this inlet (0.15)
        outlets=[7], other_walls=[5, 6, 8, 9],     # walls not on x = 0 (x = 0 walls are derived from the inlets)
        design="bifurcation_gt_design_sq{n}.npy",
        gt="bifurcation_gt_sngrm_160_dt1280_chord4.npy",
        **_PROTOCOL),
    "stenosis": dict(
        brep="./brep/artery_no_ext_stenosis.json",
        inlets={"in": 1}, char_length_inlet="in",  # L = 0.325
        outlets=[5], other_walls=[3, 4, 6, 7],
        design="stenosis_gt_design_sq{n}.npy",
        gt="stenosis_gt_sngrm_160_dt1280_chord4.npy",
        **_PROTOCOL),
    "carotid": dict(
        brep="./brep/artery_no_ext_carotid.json",
        inlets={"in": 1}, char_length_inlet="in",  # L = 0.175
        outlets=[5, 7], other_walls=[3, 4, 6, 8, 9],
        design="carotid_gt_design_sq{n}.npy",   # iso-contour snapped to the brep at the inlet top and upper-outlet bottom
        gt="carotid_gt_sngrm_160_dt1280_chord4.npy",
        **_PROTOCOL),
}


def char_length(geom, case):
    e = np.asarray(geom.edges[CASES[case]["inlets"][CASES[case]["char_length_inlet"]]], float)
    return float(np.linalg.norm(e[1] - e[0]))


def shifted_inlet_edges(geom, case, shift_elems, h_elem):
    _shift = np.array([0.0, shift_elems * h_elem])
    out = {}
    for name, i in CASES[case]["inlets"].items():
        e = np.asarray(geom.edges[i], dtype=float)
        assert np.allclose(e[:, 0], 0.0), f"{case}: inlet edge {i} is not on x = 0"
        out[name] = e + _shift
    return out


def build_fluid_bc(mesh, geom, case, char_velocity, shift_elems, face_tol, h_elem, _bc, _FluidField, verbose=True, outlet_shift_elems=0):
    """Fluid Dirichlet BCs: parabolic inlets with the flux correction, no-slip walls, outlets v = 0 and p = 0.
    Checks: one condition per boundary face, no gaps, every boundary node has a velocity condition (outlet nodes:
    v and p fixed, u free). Raises AssertionError if any check fails.
    outlet_shift_elems (default 0 = unchanged code path): every outlet moved by that many elements in y; the walls on x = 1
    become the complement of the shifted outlets on [0, 1] (other walls on x = 1 are dropped), as for the inlet shift."""
    cfg = CASES[case]
    inlet_edges = shifted_inlet_edges(geom, case, shift_elems, h_elem)
    if shift_elems and verbose:
        print(f"INLET SHIFTED by {shift_elems:+d} elements ({shift_elems * h_elem:+.4f}); the x = 0 walls move with it; "
              f"the measurements still use the true location.")
    inlet_faces, inlet_vals = {}, {}
    for key, edge in inlet_edges.items():
        faces = _bc.identify_faces(mesh, edges=[edge], tol=face_tol)
        nf = len(faces)
        fc = mesh.elem_template.face_connectivity
        y_sel = np.array([mesh.elem_node_coords[e][fc[f]][:, 1] for e, f in faces])
        L_sel = y_sel.max() - y_sel.min()
        L_true = float(np.linalg.norm(np.asarray(edge[1]) - np.asarray(edge[0])))
        flux_correction = (L_true / L_sel) / (1.0 - 1.0 / nf**2)
        x_nodes = jnp.linspace(-1.0, 1.0, nf + 1)
        u_profile = char_velocity * (1.0 - x_nodes**2) * flux_correction
        face_node_vals = jnp.stack([u_profile[1:], u_profile[:-1]], axis=1)
        inlet_faces[key] = faces
        inlet_vals[key] = [(_FluidField.U_VEL, face_node_vals), (_FluidField.V_VEL, jnp.zeros_like(face_node_vals))]
        if verbose:
            h_sel = L_sel / nf
            flux = float(jnp.sum(0.5 * (u_profile[1:] + u_profile[:-1])) * h_sel)
            print(f"inlet_{key}: nf={nf:3d}  L_sel={L_sel:.6f}  L_true={L_true:.6f}  corr={flux_correction:.6f}  "
                  f"flux={flux:.9f}  target={(2/3)*char_velocity*L_true:.9f}")

    # walls on x = 0: complement of the (shifted) inlets on [0, 1]
    spans = sorted((float(e[:, 1].min()), float(e[:, 1].max())) for e in inlet_edges.values())
    seg = lambda y0, y1: np.array([[0.0, y0], [0.0, y1]])
    left_walls, y_prev = [], 0.0
    for lo, hi in spans:
        if lo > y_prev + 1e-12: left_walls.append(seg(y_prev, lo))
        y_prev = hi
    if y_prev < 1.0 - 1e-12: left_walls.append(seg(y_prev, 1.0))
    if outlet_shift_elems == 0:
        wall_edges = left_walls + [geom.edges[i] for i in cfg["other_walls"]]
        outlet_edges = {i: geom.edges[i] for i in cfg["outlets"]}
    else:
        _osh = np.array([0.0, outlet_shift_elems * h_elem])
        outlet_edges = {i: np.asarray(geom.edges[i], dtype=float) + _osh for i in cfg["outlets"]}
        for e in outlet_edges.values():
            assert np.allclose(e[:, 0], 1.0) and e[:, 1].min() >= -1e-12 and e[:, 1].max() <= 1.0 + 1e-12, "shifted outlet leaves x = 1, [0, 1]"
        segr = lambda y0, y1: np.array([[1.0, y0], [1.0, y1]])
        right_walls, y_prev = [], 0.0
        for lo, hi in sorted((float(e[:, 1].min()), float(e[:, 1].max())) for e in outlet_edges.values()):
            if lo > y_prev + 1e-12: right_walls.append(segr(y_prev, lo))
            y_prev = hi
        if y_prev < 1.0 - 1e-12: right_walls.append(segr(y_prev, 1.0))
        on_x1 = lambda e: np.allclose(np.asarray(e, float)[:, 0], 1.0)
        wall_edges = left_walls + [geom.edges[i] for i in cfg["other_walls"] if not on_x1(geom.edges[i])] + right_walls
        if verbose:
            print(f"OUTLET SHIFTED by {outlet_shift_elems:+d} elements ({outlet_shift_elems * h_elem:+.4f}); the x = 1 walls move with it; "
                  f"the measurements still use the true location.")
    wall_faces = _bc.identify_faces(mesh, edges=wall_edges, tol=face_tol)
    n = len(wall_faces)
    wall_face_val = [(_FluidField.U_VEL, jnp.zeros(n)), (_FluidField.V_VEL, jnp.zeros(n))]

    outlet_faces = _bc.identify_faces(mesh, edges=list(outlet_edges.values()), tol=face_tol)
    outlet_faces_by_edge = {i: _bc.identify_faces(mesh, edges=[e], tol=face_tol) for i, e in outlet_edges.items()}
    n = len(outlet_faces)
    outlet_face_val = [(_FluidField.V_VEL, jnp.zeros(n)), (_FluidField.PRESSURE, jnp.zeros(n))]

    inlet_bcs = {k: _bc.DirichletBC(elem_faces=inlet_faces[k], values=inlet_vals[k], name=f"inlet_{k}") for k in inlet_edges}
    outlet_bc = _bc.DirichletBC(elem_faces=outlet_faces, values=outlet_face_val, name="outlet")
    wall_bc = _bc.DirichletBC(elem_faces=wall_faces, values=wall_face_val, name="wall")
    bcs_list = [wall_bc] + [inlet_bcs[k] for k in inlet_edges] + [outlet_bc]
    fbc = _bc.process_boundary_conditions(bcs_list, mesh)
    check = check_fluid_bc(mesh, fbc, inlet_faces, wall_faces, outlet_faces)
    if verbose or not check["ok"]:
        print(f"BC check ({case}, n={int(round(1 / h_elem))}, shift {shift_elems:+d}" + (f", outlet shift {outlet_shift_elems:+d}" if outlet_shift_elems else "") + f"): {check['summary']}")
    assert check["ok"], "fluid boundary conditions are inconsistent"
    return dict(fluid_bc=fbc, bcs_list=bcs_list, inlet_edges=inlet_edges, inlet_faces=inlet_faces, inlet_bcs=inlet_bcs,
                wall_faces=wall_faces, outlet_faces=outlet_faces, outlet_faces_by_edge=outlet_faces_by_edge, outlet_edges=outlet_edges, check=check)


def check_fluid_bc(mesh, fbc, inlet_faces, wall_faces, outlet_faces):
    as_set = lambda F: {(int(e), int(f)) for e, f in F}
    S = {f"inlet_{k}": as_set(v) for k, v in inlet_faces.items()}
    S.update(wall=as_set(wall_faces), outlet=as_set(outlet_faces))
    names = list(S)
    overlaps = {f"{a}&{b}": len(S[a] & S[b]) for i, a in enumerate(names) for b in names[i + 1:] if S[a] & S[b]}
    fc = np.asarray(mesh.elem_template.face_connectivity); enc = np.asarray(mesh.elem_node_coords)
    fxy = enc[:, fc, :]
    on = lambda c, v: np.all(np.isclose(fxy[..., c], v), axis=-1)
    bnd_faces = {(int(e), int(f)) for e, f in np.argwhere(on(0, 0.0) | on(0, 1.0) | on(1, 0.0) | on(1, 1.0))}
    covered = set().union(*S.values())
    gaps, extra = bnd_faces - covered, covered - bnd_faces
    xy = np.asarray(mesh.nodes.coords); fixed = set(np.asarray(fbc["fixed_dofs"]).tolist())
    bnd_nodes = np.where(np.isclose(xy[:, 0], 0) | np.isclose(xy[:, 0], 1) | np.isclose(xy[:, 1], 0) | np.isclose(xy[:, 1], 1))[0]
    en = np.asarray(mesh.elem_nodes)
    nodes_of = lambda F: {int(en[e][fc[f]][k]) for e, f in F for k in (0, 1)}
    non_outlet = set().union(*[v for k, v in S.items() if k != "outlet"])
    outlet_only = nodes_of(S["outlet"]) - nodes_of(non_outlet)
    no_v = [int(nd) for nd in bnd_nodes if 3 * nd + 2 not in fixed]
    no_u = [int(nd) for nd in bnd_nodes if 3 * nd + 1 not in fixed and int(nd) not in outlet_only]
    ok = not overlaps and not gaps and not extra and not no_v and not no_u
    summary = (f"boundary faces {len(bnd_faces)}, covered {len(covered)} ({', '.join(f'{k} {len(v)}' for k, v in S.items())}), "
               f"overlaps {overlaps or 'none'}, gaps {len(gaps)}, non-boundary {len(extra)} | boundary nodes {len(bnd_nodes)}: "
               f"without v {len(no_v)}, without u outside the outlet {len(no_u)}, outlet-only nodes (u free) {len(outlet_only)}")
    return dict(ok=ok, summary=summary, overlaps=overlaps, gaps=len(gaps), extra=len(extra), no_v=len(no_v), no_u=len(no_u),
                n_faces={k: len(v) for k, v in S.items()})


def design_bc_consistency(mesh, gamma, inlet_faces, wall_faces, outlet_faces):
    """Faces whose adjacent element disagrees with the condition: inlet/outlet faces on solid (gamma > 0.5),
    wall faces on x = 0 or x = 1 next to fluid (gamma < 0.5)."""
    g = np.asarray(gamma).reshape(-1)
    xy_faces = np.asarray(mesh.elem_node_coords)[:, np.asarray(mesh.elem_template.face_connectivity), :]
    side = lambda e, f: np.all(np.isclose(xy_faces[e, f, :, 0], 0.0)) or np.all(np.isclose(xy_faces[e, f, :, 0], 1.0))
    open_on_solid = sum(int(g[e] > 0.5) for F in list(inlet_faces.values()) + [outlet_faces] for e, f in F)
    wall_on_fluid = sum(int(g[e] < 0.5) for e, f in wall_faces if side(e, f))
    return dict(open_faces_on_solid=open_on_solid, side_wall_faces_on_fluid=wall_on_fluid)


def build_conc_bc(mesh, geom, case, shift_elems, h_elem, _bc, _ConcentrationField, c_in=1.0):
    """Concentration inlets (c = c_in) on the same, possibly shifted, inlet edges as the flow; same order."""
    bcs = []
    for key, edge in shifted_inlet_edges(geom, case, shift_elems, h_elem).items():
        faces = _bc.identify_faces(mesh, edges=[edge])
        bcs.append(_bc.DirichletBC(elem_faces=faces, values=[(_ConcentrationField.CONCENTRATION, jnp.ones(len(faces)) * c_in)],
                                   name=f"inlet_{key}"))
    cbc = _bc.process_boundary_conditions(bcs, mesh)
    init = jnp.zeros((mesh.num_dofs,)).at[cbc["fixed_dofs"]].set(cbc["dirichlet_values"])
    return bcs, cbc, init
