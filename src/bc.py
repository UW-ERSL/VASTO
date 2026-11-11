"""Classes of  boundary conditions."""

import abc
from typing import List, TypedDict, Tuple, Callable, Union, Optional
import enum
import numpy as np
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs

import src.utils as _utils
import src.mesher as _mesh

_Direc = _utils.Direction


def _is_face_on_edges(
  face_coords: np.ndarray, edges: List[np.ndarray], tol: float
) -> bool:
  """Check if the face coordinates are on the edges.

  Args:
    face_coords: Array of shape (n, num_dims) representing the coordinates of points
      on the face.
    edges: A list of arrays of shape (2, num_dims) representing the edge defined by two
      points (start and end).
    tol: Tolerance for checking if the point is on the segment. Usually set it to half
      the mesh size.
  Returns: True if the face coordinates are on any of the edges, False otherwise.
  """
  for edge in edges:
    if np.all(
      [_utils.is_point_on_segment(edge[0], edge[1], c, tol) for c in face_coords]
    ):
      return True
  return False


def identify_faces(
  mesh: _mesh.Mesh,
  cond_fn: Optional[Callable[[np.ndarray], np.ndarray | bool]] = None,
  edges: List[np.ndarray] | None = None,
  tol: float = 1e-6,
) -> List[Tuple[int, int]]:
  """Identify the boundary faces of the mesh that satisfy a condition.

  Args:
    mesh: Mesh object.
    cond_fn: (Optional)Function that takes the coordinates of the nodes of a face and
      returns a boolean. If None, then edge must be provided.
    edges: (Optional) A list of arrays of shape (2, n) representing the edge to check
      against. If provided, cond_fn is ignored.

  Returns: List of tuples (elem, face) where elem is the element index and face
    is the face index.
  """
  face_nodes = mesh.elem_template.face_connectivity
  face_coords = mesh.elem_node_coords[:, face_nodes]

  elems, faces = np.nonzero(mesh.boundary_faces)
  coords = face_coords[elems, faces]

  if edges is not None:
    cond_fn = lambda c: _is_face_on_edges(c, edges, tol)  # noqa: E731

  keep = [cond_fn(c) for c in coords]
  elems = elems[keep]
  faces = faces[keep]
  return list(zip(elems.tolist(), faces.tolist()))


class BCType(enum.Enum):
  """Boundary condition types."""

  DIRICHLET = enum.auto()
  NEUMANN = enum.auto()


class BoundaryCondition(abc.ABC):
  """Base class for boundary conditions."""

  @abc.abstractmethod
  def __init__(self, type: BCType, name: str = ""):
    self.type = type
    self.name = name

  @abc.abstractmethod
  def process(self, mesh: _mesh.Mesh):
    """Process the boundary condition to the mesh."""
    pass


class DirichletBC(BoundaryCondition):
  """Dirichlet boundary condition.

  The Dirichlet boundary condition is a type of boundary condition where the value of
  the solution is known at the boundary.

                             u = u_0 on Γ_D
  Where, u is the field variable, u_0 is the known value and Γ_D is the Dirichlet
  boundary.

  Attributes:
    elem_faces: A list of tuples of (element, face) for the boundary condition. The
      element is the element index in the mesh and the face is the face index of a face
      in the element. The face index is the local face index of the element.
    value: A list of tuples of (Enum, value) for the boundary condition. The
      enum consists of the dof number. For instance, in structural, the enum would be
      keys of (U,V,W) with values (0,1,2) respectively and so on. The value also
      consists of an array of (num_faces,) of the values for the boundary condition.
  """

  def __init__(
    self,
    elem_faces: List[tuple[int, int]],
    values: List[Tuple[enum.Enum, np.ndarray]],
    name: str = "",
  ):
    """Initialize Dirichlet boundary condition."""
    super().__init__(BCType.DIRICHLET, name)
    self.elem_faces = elem_faces
    self.values = values

  def process(self, mesh: _mesh.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Process the Dirichlet boundary conditions.

    Args:
      mesh: The mesh object.

    Returns: Tuple of Arrays of (fixed_dofs, dirichlet_values) for the boundary
      condition.
    """
    face_conn = mesh.elem_template.face_connectivity
    nodes_per_face = face_conn.shape[1]
    dofs, values = [], []
    for i, (elem, face) in enumerate(self.elem_faces):
      face_nodes = mesh.elem_nodes[elem, face_conn[face]]
      face_dofs = _mesh.get_dofs_of_nodes(mesh, face_nodes)
      for dir, val in self.values:
        new_dofs = face_dofs[:, dir.value]
        new_vals = val[i] * np.ones((nodes_per_face,))
        for nd, nv in zip(new_dofs, new_vals):
          if nd not in dofs:
            dofs.append(nd)
            values.append(nv)

    values = np.array(values).reshape(-1)
    dofs = np.array(dofs).reshape(-1)
    return dofs, values


class NeumannBC(BoundaryCondition):
  """Neumann boundary condition. The Neumann boundary condition is a type of boundary
  condition where the value of the derivative of the solution is known at the
  boundary.

                n . ∇u = g on Γ_N
  Where, n is the outward normal to the boundary, ∇u is the gradient of the field
  variable and g is the known value of the derivative. Γ_N is the Neumann boundary.

  Attributes:
  elem_faces: A list of tuples of (element, face) for the boundary condition. The
    element is the element index in the mesh and the face is the face index of a face
    in the element. The face index is the local face index of the element.
  value: A list of tuples of (Enum, value) for the boundary condition. The
      enum consists of the dof number. For instance, in structural, the enum would be
      keys of (U,V,W) with values (0,1,2) respectively and so on. The value also
      consists of an array of (num_faces,) of the values for the boundary condition.
  """

  def __init__(
    self,
    elem_faces: List[tuple[int, int]],
    values: List[Tuple[_Direc, np.ndarray]],
    name: str = "",
  ):
    """Initialize Neumann boundary condition."""
    super().__init__(BCType.NEUMANN, name)
    self.elem_faces = elem_faces
    self.values = values

  def process(self, mesh: _mesh.Mesh) -> List[tuple[int, np.ndarray]]:
    """Process the Neumann boundary conditions.

    Args:
      mesh: The mesh object.

    Returns: A list of elem_forces tuple where each tuple is (elem, force_at_elem_dofs).
       The elem is the element index and force_at_elem_dofs is an array of
       (num_dofs_per_elem,) of the forces at the element dofs.
       The size of the list is equal to the number of elements in `elem_faces`.
    """
    face_conn = mesh.elem_template.face_connectivity
    nodes_per_face = face_conn.shape[1]
    forces = []
    for i, (elem, face) in enumerate(self.elem_faces):
      face_dofs_local = _mesh.get_dofs_of_nodes(mesh, face_conn[face])
      face_force = jnp.zeros((mesh.num_dofs_per_elem,))
      for dofno, val in self.values:
        dofs = face_dofs_local[:, dofno.value]
        applied_force = val[i] * np.ones((nodes_per_face,)) / nodes_per_face
        face_force = face_force.at[dofs].set(applied_force)
      forces.append((elem, face_force))
    return forces


class BCDict(TypedDict):
  elem_forces: np.ndarray
  fixed_dofs: np.ndarray
  free_dofs: np.ndarray
  dirichlet_values: np.ndarray


def process_boundary_conditions(
  bcs: List[BoundaryCondition],
  mesh: _mesh.Mesh,
) -> BCDict:
  """Process the boundary conditions.

  Args:
    bcs: List of boundary conditions.
    mesh: The mesh object.

  Returns: Tuple of (force, dirichlet_dofs, dirichlet_values)
  """
  dirichlet_dofs = []
  dirichlet_values = []
  elem_forces = jnp.zeros((mesh.num_elems, mesh.num_dofs_per_elem))

  for bc in bcs:
    if isinstance(bc, NeumannBC):
      forces = bc.process(mesh)
      for elem, elem_force in forces:
        elem_forces = elem_forces.at[elem].add(elem_force)

    elif isinstance(bc, DirichletBC):
      fixed_dofs, fixed_values = bc.process(mesh)
      dirichlet_dofs.extend(fixed_dofs)
      dirichlet_values.extend(fixed_values)

    else:
      raise ValueError("Unsupported boundary condition type")

  fixed_dofs = np.array(dirichlet_dofs).astype(int).reshape(-1)
  free_dofs = np.setdiff1d(np.arange(mesh.num_dofs), fixed_dofs)

  return BCDict(
    elem_forces=elem_forces,
    fixed_dofs=fixed_dofs,
    free_dofs=free_dofs,
    dirichlet_values=np.array(dirichlet_values).reshape(-1),
  )


def apply_dirichlet_bc(
  jacobian: jax_sprs.BCOO, fixed_dofs: np.ndarray, zero_columns: bool = True
) -> jax_sprs.BCOO:
  """Applies Dirichlet boundary conditions to a BCOO Jacobian matrix.

  The function modifies the Jacobian matrix by zeroing out the rows and columns
    corresponding to the fixed degrees of freedom (Dirichlet conditions) and adding
    back the diagonal entries with a value of 1.0.

  Args:
    jacobian: The Jacobian matrix as a JAX BCOO sparse matrix.
    dirichlet_nodes: A list of node indices with Dirichlet conditions.

  Returns: The modified Jacobian matrix as a JAX BCOO sparse matrix.
  """
  data, indices = jacobian.data, jacobian.indices
  num_dofs = jacobian.shape[0]
  is_fixed = jnp.zeros((num_dofs,), dtype=bool).at[fixed_dofs].set(True)
  # 1. Zeroing rows:
  row_indices = indices[:, 0]
  mask = is_fixed[row_indices] 
  data = jnp.where(mask, 0.0, data)

  # 2. Zeroing columns -- skipped when the prescribed values are differentiated.
  #    The column entries couple the interior dofs to the constrained ones. In the
  #    forward solve the increment at constrained dofs is always zero, so dropping
  #    them changes nothing. In the tangent solve the increment at constrained dofs
  #    is NOT zero (it is the change in the prescribed value), and these entries are
  #    what carries that change into the interior.
  if zero_columns:
    col_indices = indices[:, 1]
    mask = is_fixed[col_indices]
    data = jnp.where(mask, 0.0, data)

  # Add back diagonal entries with 1.0
  diag_indices = jnp.stack([fixed_dofs, fixed_dofs], axis=1)
  indices = jnp.concatenate([indices, diag_indices])
  diag_data = jnp.ones_like(fixed_dofs, dtype=data.dtype)
  data = jnp.concatenate([data, diag_data])

  return jax_sprs.BCOO((data, indices), shape=jacobian.shape)


# ---------------------------------------------------------------------------------------------------------------------
# Generic vessel boundary conditions on the unit square: inlets on x = 0, outlets on x = 1.
# The geometry enters only through the edges passed in (brep edges), so every case (bifurcation, stenosis, carotid)
# uses the same builder. For the same edges these give the same faces, dofs and values as
# experiments/transport/case_config.build_fluid_bc / build_conc_bc.
# ---------------------------------------------------------------------------------------------------------------------


def _shift_edges(edges: dict, shift_elems: int, h_elem: float | None, x: float, side: str) -> dict:
  """Edges moved by shift_elems elements in y (copies as float arrays); checks that each edge lies on x."""
  dy = 0.0 if shift_elems == 0 else shift_elems * h_elem
  out = {}
  for key, e in edges.items():
    e = np.asarray(e, dtype=float)
    assert np.allclose(e[:, 0], x), f"{side} edge {key!r} is not on x = {x:g}"
    out[key] = e + np.array([0.0, dy])
  return out


def _complement_walls(edges, x: float) -> List[np.ndarray]:
  """Wall segments on the line x covering [0, 1] minus the y-spans of `edges` (inlets or outlets on that line)."""
  spans = sorted((float(e[:, 1].min()), float(e[:, 1].max())) for e in edges)
  seg = lambda y0, y1: np.array([[x, y0], [x, y1]])  # noqa: E731
  walls, y_prev = [], 0.0
  for lo, hi in spans:
    if lo > y_prev + 1e-12:
      walls.append(seg(y_prev, lo))
    y_prev = hi
  if y_prev < 1.0 - 1e-12:
    walls.append(seg(y_prev, 1.0))
  return walls


def build_fluid_bc(
  mesh: _mesh.Mesh,
  inlets: dict,
  outlets: dict,
  walls: List[np.ndarray],
  char_velocity: float,
  face_tol: float,
  h_elem: float,
  field,
  inlet_shift_elems: int = 0,
  outlet_shift_elems: int = 0,
  verbose: bool = True,
  label: str = "",
) -> dict:
  """Fluid Dirichlet BCs: parabolic inlets with the flux correction, no-slip walls, outlets v = 0 and p = 0.

  Args:
    mesh: fluid mesh.
    inlets: {name: edge} of the inlets, all on x = 0; the order is the order of the inlet scalings.
    outlets: {key: edge} of the outlets, all on x = 1 (key: e.g. the brep edge number).
    walls: no-slip edges that are NOT on x = 0. The walls on x = 0 are the complement of the inlets on [0, 1].
    char_velocity: peak inlet velocity of the parabola.
    face_tol: tolerance of identify_faces (0.49 element sizes in the template).
    h_elem: element size (used for the shifts and the printout).
    field: the fluid field enum with U_VEL, V_VEL and PRESSURE (fe_fluid.FluidField).
    inlet_shift_elems: every inlet moved by that many elements in y (0 = the brep); the x = 0 walls follow.
    outlet_shift_elems: every outlet moved by that many elements in y; the walls on x = 1 become the complement of the
      shifted outlets (walls passed on x = 1 are dropped).
    verbose: print the inlet fluxes and the check summary.
    label: name used in the printout.

  Checks: one condition per boundary face, no gaps, every boundary node has a velocity condition (outlet nodes: v and
  p fixed, u free). Raises AssertionError if any check fails.

  Returns: dict with fluid_bc (processed), bcs_list, inlet_edges, inlet_faces, inlet_bcs, wall_faces, outlet_faces,
    outlet_faces_by_edge, outlet_edges, check.
  """
  inlet_edges = _shift_edges(inlets, inlet_shift_elems, h_elem, 0.0, "inlet")
  if inlet_shift_elems and verbose:
    print(f"INLET SHIFTED by {inlet_shift_elems:+d} elements ({inlet_shift_elems * h_elem:+.4f}); the x = 0 walls move with it; "
          f"the measurements still use the true location.")
  inlet_faces, inlet_vals = {}, {}
  fc = mesh.elem_template.face_connectivity
  for key, edge in inlet_edges.items():
    faces = identify_faces(mesh, edges=[edge], tol=face_tol)
    nf = len(faces)
    y_sel = np.array([mesh.elem_node_coords[e][fc[f]][:, 1] for e, f in faces])
    L_sel = y_sel.max() - y_sel.min()
    L_true = float(np.linalg.norm(np.asarray(edge[1]) - np.asarray(edge[0])))
    # flux of the nodal parabola on the selected faces = (2/3) char_velocity L_true
    flux_correction = (L_true / L_sel) / (1.0 - 1.0 / nf**2)
    x_nodes = jnp.linspace(-1.0, 1.0, nf + 1)
    u_profile = char_velocity * (1.0 - x_nodes**2) * flux_correction
    face_node_vals = jnp.stack([u_profile[1:], u_profile[:-1]], axis=1)
    inlet_faces[key] = faces
    inlet_vals[key] = [(field.U_VEL, face_node_vals), (field.V_VEL, jnp.zeros_like(face_node_vals))]
    if verbose:
      h_sel = L_sel / nf
      flux = float(jnp.sum(0.5 * (u_profile[1:] + u_profile[:-1])) * h_sel)
      print(f"inlet_{key}: nf={nf:3d}  L_sel={L_sel:.6f}  L_true={L_true:.6f}  corr={flux_correction:.6f}  "
            f"flux={flux:.9f}  target={(2/3)*char_velocity*L_true:.9f}")

  left_walls = _complement_walls(inlet_edges.values(), 0.0)
  if outlet_shift_elems == 0:
    wall_edges = left_walls + list(walls)
    outlet_edges = dict(outlets)
  else:
    outlet_edges = _shift_edges(outlets, outlet_shift_elems, h_elem, 1.0, "outlet")
    for e in outlet_edges.values():
      assert e[:, 1].min() >= -1e-12 and e[:, 1].max() <= 1.0 + 1e-12, "shifted outlet leaves x = 1, [0, 1]"
    on_x1 = lambda e: np.allclose(np.asarray(e, float)[:, 0], 1.0)  # noqa: E731
    wall_edges = left_walls + [w for w in walls if not on_x1(w)] + _complement_walls(outlet_edges.values(), 1.0)
    if verbose:
      print(f"OUTLET SHIFTED by {outlet_shift_elems:+d} elements ({outlet_shift_elems * h_elem:+.4f}); the x = 1 walls move with it; "
            f"the measurements still use the true location.")
  wall_faces = identify_faces(mesh, edges=wall_edges, tol=face_tol)
  n = len(wall_faces)
  wall_face_val = [(field.U_VEL, jnp.zeros(n)), (field.V_VEL, jnp.zeros(n))]

  outlet_faces = identify_faces(mesh, edges=list(outlet_edges.values()), tol=face_tol)
  outlet_faces_by_edge = {k: identify_faces(mesh, edges=[e], tol=face_tol) for k, e in outlet_edges.items()}
  n = len(outlet_faces)
  outlet_face_val = [(field.V_VEL, jnp.zeros(n)), (field.PRESSURE, jnp.zeros(n))]

  inlet_bcs = {k: DirichletBC(elem_faces=inlet_faces[k], values=inlet_vals[k], name=f"inlet_{k}") for k in inlet_edges}
  outlet_bc = DirichletBC(elem_faces=outlet_faces, values=outlet_face_val, name="outlet")
  wall_bc = DirichletBC(elem_faces=wall_faces, values=wall_face_val, name="wall")
  bcs_list = [wall_bc] + [inlet_bcs[k] for k in inlet_edges] + [outlet_bc]
  fbc = process_boundary_conditions(bcs_list, mesh)
  check = check_fluid_bc(mesh, fbc, inlet_faces, wall_faces, outlet_faces)
  if verbose or not check["ok"]:
    print(f"BC check ({label}, n={int(round(1 / h_elem))}, shift {inlet_shift_elems:+d}"
          + (f", outlet shift {outlet_shift_elems:+d}" if outlet_shift_elems else "") + f"): {check['summary']}")
  assert check["ok"], "fluid boundary conditions are inconsistent"
  return dict(fluid_bc=fbc, bcs_list=bcs_list, inlet_edges=inlet_edges, inlet_faces=inlet_faces, inlet_bcs=inlet_bcs,
              wall_faces=wall_faces, outlet_faces=outlet_faces, outlet_faces_by_edge=outlet_faces_by_edge,
              outlet_edges=outlet_edges, check=check)


def check_fluid_bc(mesh: _mesh.Mesh, fbc: BCDict, inlet_faces: dict, wall_faces, outlet_faces) -> dict:
  """Consistency of the fluid BCs on the unit square: no face with two conditions, no boundary face without one, no
  condition off the boundary, every boundary node has v fixed and u fixed (except nodes only on outlet faces)."""
  as_set = lambda F: {(int(e), int(f)) for e, f in F}  # noqa: E731
  S = {f"inlet_{k}": as_set(v) for k, v in inlet_faces.items()}
  S.update(wall=as_set(wall_faces), outlet=as_set(outlet_faces))
  names = list(S)
  overlaps = {f"{a}&{b}": len(S[a] & S[b]) for i, a in enumerate(names) for b in names[i + 1:] if S[a] & S[b]}
  fc = np.asarray(mesh.elem_template.face_connectivity); enc = np.asarray(mesh.elem_node_coords)
  fxy = enc[:, fc, :]
  on = lambda c, v: np.all(np.isclose(fxy[..., c], v), axis=-1)  # noqa: E731
  bnd_faces = {(int(e), int(f)) for e, f in np.argwhere(on(0, 0.0) | on(0, 1.0) | on(1, 0.0) | on(1, 1.0))}
  covered = set().union(*S.values())
  gaps, extra = bnd_faces - covered, covered - bnd_faces
  xy = np.asarray(mesh.nodes.coords); fixed = set(np.asarray(fbc["fixed_dofs"]).tolist())
  bnd_nodes = np.where(np.isclose(xy[:, 0], 0) | np.isclose(xy[:, 0], 1) | np.isclose(xy[:, 1], 0) | np.isclose(xy[:, 1], 1))[0]
  en = np.asarray(mesh.elem_nodes)
  nodes_of = lambda F: {int(en[e][fc[f]][k]) for e, f in F for k in (0, 1)}  # noqa: E731
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


def build_conc_bc(
  mesh: _mesh.Mesh,
  inlets: dict,
  field,
  c_in: float = 1.0,
  inlet_shift_elems: int = 0,
  h_elem: float | None = None,
):
  """Concentration inlets (c = c_in) on the same, possibly shifted, inlet edges as the flow, in the same order.

  Args:
    mesh: concentration mesh.
    inlets: {name: edge} of the inlets on x = 0 (as for build_fluid_bc).
    field: the concentration field enum (fe_transient_transport.ConcentrationField).
    c_in: inlet concentration.
    inlet_shift_elems, h_elem: inlet shift, as for build_fluid_bc.

  Returns: (bcs, processed bcs, initial concentration with the inlet values set).
  """
  bcs = []
  for key, edge in _shift_edges(inlets, inlet_shift_elems, h_elem, 0.0, "inlet").items():
    faces = identify_faces(mesh, edges=[edge])
    bcs.append(DirichletBC(elem_faces=faces, values=[(field.CONCENTRATION, jnp.ones(len(faces)) * c_in)],
                           name=f"inlet_{key}"))
  cbc = process_boundary_conditions(bcs, mesh)
  init = jnp.zeros((mesh.num_dofs,)).at[cbc["fixed_dofs"]].set(cbc["dirichlet_values"])
  return bcs, cbc, init


bcLike = Union[BoundaryCondition, DirichletBC, NeumannBC]
