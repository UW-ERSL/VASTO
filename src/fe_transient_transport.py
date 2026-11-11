"""Transient transport equation solver (non-dimensional form).

We solve the transient concentration transport on Ω

    ∂c/∂t + u · ∇c − (1/Pe) ∇² c = 0,

with Dirichlet/Neumann boundary conditions on Γ.

  c(x, t): concentration field (iodine),
  Pe     : element/global Péclet number (nondimensional), where Pe = L V / D
  u      : prescribed velocity field,
  D(x, t): diffuison coefficient,

Velocity field:
  • u is obtained by solving the (incompressible) Navier–Stokes equations on the
    same domain. 
    The flow is non-dimensionalized using a characteristic length 
    and velocity, and the dimensionless velocity is passed to this
    module. This module assumes nondimensional inputs; if using dimensional inputs, replace (1/Pe)∇²c with ∇·(D∇c). 
    For non dimensionalization, see equation D.3 (Appendix D) in reference below.

The formulation is implemented with stabilized finite elements. We use the
Streamline-Upwind/Petrov–Galerkin (SUPG) stabilization to suppress
spurious oscillations in advection-dominated (high-Péclet) regimes while
retaining accuracy.

We follow the approach of :
  Alexandersen, Joe. "Topology optimisation for coupled convection problems." (2013)


The time integration is performed using the backward Euler method:
            (C_n - C_{n-1})/dt + u . grad(C_n) - 1/Pe (grad . (grad(C_n))) = 0
"""

import enum
import numpy as np
import jax.numpy as jnp
import jax
import jax.experimental.sparse as jax_sprs
from jax.typing import ArrayLike

import src.solver as _nlsolv
import src.mesher as _mesh
import src.material as _mat
import src.bc as _bc


class ConcentrationField(enum.Enum):
  """The concentration field."""

  CONCENTRATION = 0


class FEA(_nlsolv.NonlinearProblem):
  """Scalar concentration transport (advection-diffuision) finite element analysis in incompressible flow."""

  def __init__(
    self,
    mesh: _mesh.Mesh,
    material: _mat.FluidMaterial,
    bc: _bc.BCDict,
    solver_settings: dict,
  ):
    """Initializes the Transient scalar advection–diffusion (concentration) FEA problem."""
    super().__init__(solver_settings=solver_settings)
    self.mesh, self.material, self.bc = (
      mesh,
      material,
      bc,
    )
    self.node_id_jac = np.stack((self.mesh.iK, self.mesh.jK)).astype(np.int32).T
    self.shp_fn = jax.vmap(self.mesh.elem_template.shape_functions)(mesh.gauss_pts)

  def _compute_elem_stabilization(
    self,
    velocity: jnp.ndarray,
    peclet_number: jnp.ndarray,
    elem_char_length: jnp.ndarray,
    delta_time: float = 1e-8,
  ) -> jnp.ndarray:
    """Returns the stabilization parameter for SUPG term.

      This function computes the stabilization parameter (τ) used in
        advection-diffusion problems for each element.The stabilization
        parameter (τ) is computed using an approximate minimum function considering
        three limiting cases:

      - τ₁: Convective limit
      - τ2: Transient limit
      - τ₃: Diffusive limit

      The stabilization parameter is assumed constant within each element, and
      τ₁ is computed based on the velocity components evaluated at the element centroid.

      Stabilization parameter (τ) is computed as:

      τ = ( τ₁⁻² + τ2⁻² + τ₃⁻² )^(-1/2)

      Where:
      τ₁ = h / ( 2√(uᵢ uᵢ) )   # Convective limit
      τ₃ = h²Pe / 4            # Diffusive limit
      τ_2 = Δt / 2             # Transient limit

      Variables:
      - h: Element characteristic length
      - uᵢ: Velocity components
      - Pe: Peclet number

     For details see Appendix A, equation A.52 of:
      Alexandersen, J., 2013. Topology optimisation for coupled convection problems.
      and
      Remark 24, equation 56 of:
      TE Tezduyar, 2001. Adaptive determination of the finite element stabilization parameters

    Args:
      velocity: An array of shape (num_velocity_dofs_per_elem,) of the velocity field at
        the element dofs. The velocity is assumed to be in the order of
        [u₁, v₁, w₁, u₂, v₂, w₂, ...] for 2D and 3D problems.
      peclet_number: A scalar array containing the Péclet number of the element.
      elem_char_length: A scalar array containing the characteristic length of the
        element.
      delta_time: Time step size.

    Returns:
      A scalar array containing the stabilization parameter for the element.
    """

    gp_center = jnp.zeros((self.mesh.num_dim,))
    shp_fn = self.mesh.elem_template.shape_functions(gp_center)

    # (d)(i)m, (g)auss, (n)odes_per_elem

    # centroid velocity: u_c = Σ_n N_n(0) u_n   -> shape (d,)
    u0 = jnp.einsum("n, nd -> d", shp_fn, velocity.reshape(-1, self.mesh.num_dim))
    # |u_c|^2  -> scalar
    ue = jnp.einsum("d, d -> ", u0, u0)


    inv_sq_tau1 = (4 * ue) / elem_char_length**2
    tau_2 = 0.5 * delta_time
    tau_3 = (peclet_number * elem_char_length**2) / 4

    return (inv_sq_tau1 + tau_2 ** (-2) + tau_3 ** (-2)) ** (-1 / 2)

  def _compute_elem_residual(
    self,
    concentration: jnp.ndarray,
    prev_concentration_elem: jnp.ndarray,
    velocity: jnp.ndarray,
    peclet_number: jnp.ndarray,
    node_coords: jnp.ndarray,
    delta_time: float,
    elem_char_length: float,
  ) -> jnp.ndarray:
    """Computes the elemental residual of the concentration stiffness matrix.

    The weak form of the transient mass transport equation (dimensional form) with SUPG
    stabilisation can be written as:
        ∫_Ω_e  w * ∂C/∂t dΩ_e                         (transient)
      + ∫_Ω_e  w * u_j * ∂C/∂x_j dΩ_e                   (convection)
      + ∫_Ω_e  (∂w/∂x_j) * (1/Pe)* ∂C/∂x_j  dΩ_e        (diffusion)
      + ∫_Ω_e  τ_C * u_j * ∂w/∂x_j * R_C(u, C) dΩ_e     (SUPG)
      = 0

    where:
          Ω_e      : element analysis domain
          u_j      : velocity component in direction x_j
          C        : concentration field
          w        : weight / test function
          Pe       : Péclet number. It is defined as Pe = L V / α
          L        : characteristic length of the domain,
          V        : characteristic velocity,
          α        : mass diffusivity of the fluid.
          τ_C      : SUPG stabilisation parameter
          R_C(u,C) : strong-form residual of the mass transport equation

    Where:
          R_C = u_j * (∂C/∂x_j)


    For details see 3.1c and A.44 (Appendix A 6) of:
      Alexandersen, J., 2013. Topology optimisation for coupled convection problems.

    NOTE: This implementation assumes there are no externally applied surface mass flux
      or volumetric mass sources. This simplification is valid only for the optimization
      problems considered herein. For problems such as mass sinks, with mass generation
      these terms need to be added to the residual.

    Args:
      concentration: Array of (num_dofs_per_elem,) containing the concentration of the nodes
        of an element.
      prev_concentration_elem: Array of (num_dofs_per_elem,) containing the concentration
        of the nodes of an element at the previous time step.
      velocity: Array of (num_nodes_per_elem * num_dim,) containing the velocity at the
        nodes of an element. The velocity  are assumed to be ordered as
        (u1, v1, w2 u2, v2, w2...) etc. The velocity is part of the convective mass
        transfer.
      peclet_number: Scalar value of the Péclet number of the element.
      node_coords: Array of (num_nodes_per_elem, num_dims) containing the coordinates of
        the nodes of an element.
      delta_time: Time step size.
      elem_char_length: Scalar value of the diagonal length of the element.

    Returns: Array of (num_dofs_per_elem,) containing the residual of the element. The
      residual's ordered is assumed as (t1, t2, t3,...) of the concentration at the nodes.
    """
      # (d)(i)m, (g)auss, (n)odes_per_elem = (c)oncentration_dofs_per_elem, (v)el_dofs_per_elem
    shp_fn = jax.vmap(self.mesh.elem_template.shape_functions, in_axes=(0,))(
      self.mesh.gauss_pts
    )  # {gn}

    grad_shp_fn = jax.vmap(
      self.mesh.elem_template.get_gradient_shape_function_physical, in_axes=(0, None)
    )(self.mesh.gauss_pts, node_coords)  # (g,n,d)

    _, det_jac = jax.vmap(
      self.mesh.elem_template.compute_jacobian_and_determinant, in_axes=(0, None)
    )(self.mesh.gauss_pts, node_coords)

    stab_param = self._compute_elem_stabilization(
      velocity, peclet_number, elem_char_length, delta_time
    )
    vel_gauss = jnp.einsum(
      "gn, nd -> gd", shp_fn, velocity.reshape(-1, self.mesh.num_dim)
    )
    dconcentration_xy = jnp.einsum("gnd, n -> gd", grad_shp_fn, concentration)

    # Mass term
    mass_elem = jnp.einsum(
      "gn, go, g, g -> no",
      shp_fn,
      shp_fn,
      det_jac,
      self.mesh.gauss_weights,
    )
    res_mass = jnp.einsum("ij, j->i", mass_elem, concentration) - jnp.einsum("ij, j->i", mass_elem, prev_concentration_elem)

    # Advection term
    res_conv = jnp.einsum(
      "gn, gd, gd, g, g -> n", self.shp_fn, vel_gauss, dconcentration_xy, self.mesh.gauss_weights, det_jac)

    # Diffusion term
    res_diff = (1.0 / peclet_number) * jnp.einsum(
      "gnd, gd, g, g -> n", grad_shp_fn, dconcentration_xy, self.mesh.gauss_weights, det_jac
    )

    concentration_at_gauss = jnp.einsum("gn, n -> g", shp_fn, concentration)
    prev_concentration_elem_at_gauss = jnp.einsum("gn, n -> g", shp_fn, prev_concentration_elem)
    # per-GP strong residual (omit Laplacian term for P1)
    res_strong_form = (concentration_at_gauss - prev_concentration_elem_at_gauss) / delta_time + jnp.einsum("gd, gd -> g", vel_gauss, dconcentration_xy)
    # SUPG contribution
    u_dot_grad_w_g = jnp.einsum("gd, gnd -> gn", vel_gauss, grad_shp_fn)
    res_conv_supg  = stab_param * jnp.einsum("gn, g, g -> n", u_dot_grad_w_g * res_strong_form[:,None], det_jac, self.mesh.gauss_weights)
    
    k_res = res_diff + res_conv + res_conv_supg
    return res_mass + delta_time * k_res

  def get_residual_and_tangent_stiffness(
    self,
    concentration: ArrayLike,
    prev_concentration: ArrayLike,
    delta_time: float,
    elem_velocity: ArrayLike,
    peclet_number: ArrayLike,
  ) -> tuple[ArrayLike, jax_sprs.BCOO]:
    """Compute the residual of the system of equations.

      The residual takes into  account the convection and diffusion of concentration. We solve
      the transient mass transport equation with SUPG stabilization. The residual is given by:
                  res = res_mass + res_diff + res_conv + res_conv_supg
      where:
        res_mass: Transient term of the residual.
        res_diff: Diffusion term of the residual.
        res_conv: Convection term of the residual.
        res_conv_supg: SUPG stabilization term of the residual.

      Then the tangent stiffness matrix is computed as the Jacobian of the residual
      with respect to the concentration field. We compute the Jacobian using
      automatic differentiation.

    Args:
      concentration: Array of size (num_dofs,) which is the concentration of the nodes of the mesh.
      prev_concentration: Array of size (num_dofs,) which is the concentration of the nodes of the
        mesh at the previous time step.
      elem_velocity: Array of size (num_elems, num_nodes_per_elem*num_dim) that contain
        the velocity at the nodes of the elements. The velocity is assumed to be ordered
        as (u1, v1, w1, u2, v2, w2, ...).
      peclet_number: Array of size (num_elems,) that contain the peclet number of the
        elements.

    Returns:
      residual: Array of size (num_dofs,) which is the residual of the system.
      assm_jac: Sparse matrix of size (num_dofs, num_dofs) which is the tangent stiffness
        matrix of the system.
    """
    # (e)lement, (d)ofs_per_elem
    concentration_elem = concentration[self.mesh.elem_dof_mat]  # {ed}
    prev_concentration_elem = prev_concentration[self.mesh.elem_dof_mat]  # {ed}

    res_args = (
      concentration_elem,
      prev_concentration_elem,
      elem_velocity,
      peclet_number,
      self.mesh.elem_node_coords,
      delta_time,
      self.mesh.elem_diag_length,
    )

    vmap_axes = (0, 0, 0, 0, 0, None, 0)
    # residual
    elem_residual = jax.vmap(self._compute_elem_residual, in_axes=vmap_axes)(*res_args)  # {ed}
    residual = jnp.zeros((self.mesh.num_dofs,))
    residual = residual.at[self.mesh.elem_dof_mat].add(elem_residual)
    residual = residual.at[self.bc["fixed_dofs"]].set(0.0)

    # tangent stiffness
    elem_jac = jax.vmap(jax.jacfwd(self._compute_elem_residual, argnums=0), in_axes=vmap_axes)(*res_args)
    assm_jac = jax_sprs.BCOO(
      (elem_jac.flatten(), self.node_id_jac),
      shape=(self.mesh.num_dofs, self.mesh.num_dofs),
    ).T
    assm_jac = _bc.apply_dirichlet_bc(assm_jac, self.bc["fixed_dofs"])
    return residual, assm_jac

def inlet_scale_from_intervals(t, intervals_start, intervals_end, c_in=1.0, ramp=0.05):
    """
    For a given time t, returns c_in if t is within any of the specified intervals, else 0.0.
    This function can be used to model inlet concentration profiles that are active
    during specific time intervals.
    Args:
      t: time (scalar) in physical time (seconds). 
      c_in: Inlet concentration value to return if t is within any interval.
      intervals_start: Array of start times for intervals (seconds) in physical time.
      intervals_end: Array of end times for intervals (seconds) in physical time.
      ramp: length (s) of a raised-cosine ramp 0.5(1 - cos θ) centred on each start/end time; 
      dose per pulse stays c_in*(end - start). ramp=0 gives the original step.

    """
    if ramp <= 0.0:
        in_any = jnp.any((t >= intervals_start) & (t < intervals_end))
        return jnp.where(in_any, c_in, 0.0)
    th_up = jnp.pi * jnp.clip((t - intervals_start) / ramp + 0.5, 0.0, 1.0)
    th_dn = jnp.pi * jnp.clip((intervals_end - t) / ramp + 0.5, 0.0, 1.0)
    s = jnp.minimum(0.5 * (1.0 - jnp.cos(th_up)), 0.5 * (1.0 - jnp.cos(th_dn)))
    return c_in * jnp.max(s)

def solve_transient_fea(
  fea: FEA,
  init_concentration: ArrayLike,
  eff_diffusivity: ArrayLike,
  time_step_sizes_star: ArrayLike,
  characteristic_length: ArrayLike,
  elem_velocity: ArrayLike,
  char_velocity: float,
  dt_phys: ArrayLike,              # physical dt in seconds, shape (num_time_steps,)
  intervals_start: ArrayLike,  # physical time in seconds
  intervals_end: ArrayLike,    # physical time in seconds
) -> ArrayLike:
  """Solve the transient mass transport FEA problem.

  Args:
    fea: The `FEA` object of the transient transport solver.
    init_concentration: Array of size (num_dofs,) which is the initial concentration at the nodes
      of the mesh.
    eff_diffusivity: Array of size (num_elems,) that contain the effective diffusivity of each element.
    time_step_sizes_star: Array of size (num_time_steps,) which is the time step sizes
      for each time step (non-dimensional).
    characteristic_length: Array of size (num_elems,) that contain the
      characteristic length of each element.
    elem_velocity: Array of size (num_elems, num_nodes_per_elem*num_dim) that contain
      the velocity at the nodes of the elements. The velocity is assumed to be ordered
      as (u1, v1, w1, u2, v2, w2, ...).
    char_velocity: A scalar float which is the characteristic velocity of the flow.
    dt_phys: Array of size (num_time_steps,) which is the time step sizes
      for each time step (physical time in seconds).
    intervals_start: Array of start times for inlet concentration intervals (seconds) in physical time.
    intervals_end: Array of end times for inlet concentration intervals (seconds) in physical time.

  Returns: Array of size (num_time_steps, num_dofs) which is the concentration at the
    nodes of the mesh at each time step.
  """
  # Initalize the concentration history array
  num_time_steps = time_step_sizes_star.shape[0]
  c_hist = jnp.zeros((num_time_steps + 1, fea.mesh.num_dofs))

  fixed = fea.bc["fixed_dofs"]
  dirich_on  = fea.bc["dirichlet_values"]          # assumes these are all ones*c_in
  dirich_off = jnp.zeros_like(dirich_on)              # baseline 0 (change if needed)

  def inlet_scale(t):
        return inlet_scale_from_intervals(t, intervals_start, intervals_end)

  # Apply BC at t=0
  t0_phys = 0.0 # initial physical time
  s0 = inlet_scale(t0_phys)
  dir0 = s0 * dirich_on + (1.0 - s0) * dirich_off

  init_concentration = init_concentration.at[fixed].set(dir0)
  c_prev = init_concentration

  c0 = init_concentration
  c_hist = c_hist.at[0, :].set(c0)

  peclet_number = characteristic_length * char_velocity / eff_diffusivity

  def loop_body(c_step, val):
    c_prev, c0, c_hist, t_phys = val

    # non-dimensional dt used by your residual
    del_time_star = time_step_sizes_star[c_step]
    # physical dt used for scheduling
    del_time_phys = dt_phys[c_step]

    t_next = t_phys + del_time_phys
    s = inlet_scale(t_next)
    dirich_step = s * dirich_on + (1.0 - s) * dirich_off

    # enforce BC on initial guess for NR
    c0 = c0.at[fixed].set(dirich_step)
    c_prev = c_prev.at[fixed].set(dirich_step)

    c = _nlsolv.modified_newton_raphson_solve(
      fea,
      c0,
      c_prev,
      del_time_star,
      elem_velocity,
      peclet_number,
    )

    # enforce BC after solve
    c = c.at[fixed].set(dirich_step)

    #c = jnp.clip(c, a_min=0.0)  # Concentration cannot be negative
    c0_new = jax.lax.stop_gradient(c)

    c_hist = c_hist.at[c_step + 1, :].set(c)
    return c, c0_new, c_hist, t_next
  
  loop_body = jax.checkpoint(loop_body)
  jitted_loop_body = jax.jit(loop_body)

  init_val = (c_prev, c0, c_hist, t0_phys)
  _, _, c_hist, _ = jax.lax.fori_loop(0, num_time_steps, jitted_loop_body, init_val)

  return c_hist

def solve_transient_linear(
  fea: FEA,
  init_concentration: ArrayLike,
  eff_diffusivity: ArrayLike,
  time_step_sizes_star: ArrayLike,
  characteristic_length: ArrayLike,
  elem_velocity: ArrayLike,
  char_velocity: float,
  dt_phys: ArrayLike,
  intervals_start: ArrayLike,
  intervals_end: ArrayLike,
) -> ArrayLike:
  """Linear time stepping of the transient transport problem (same discretisation as
  `solve_transient_fea`, same arguments and return value).

  For a fixed design the element residual is linear in (C_n, C_{n-1}):
      r(C_n, C_{n-1}) = A C_n - B C_{n-1},   A = dr/dC_n,  B = -dr/dC_{n-1},
  with A and B depending only on the velocity, the Péclet number and dt (the SUPG
  parameter does not depend on C). A and B are therefore assembled once, and each time
  step is a single linear solve with the existing `solve` (one A-solve forward, one
  A^T-solve backward), instead of the modified Newton loop.

  Dirichlet convention (identical to `solve_transient_fea`): at step n both the guess
  and C_{n-1} take the new boundary value g_n at the fixed dofs, the residual rows of the
  fixed dofs are zeroed, and A gets the same row/column zeroing with a unit diagonal
  (`apply_dirichlet_bc`). One Newton step from that guess is exact for this linear
  residual:
      C_g = C_{n-1} with C_g[fixed] = g_n
      C_n = C_g - A_bc^{-1} r_free(C_g, C_g),   r(C_g, C_g) = (A - B) C_g.

  NOTE: the time step must be uniform (A and B are assembled with
    time_step_sizes_star[0]; the SUPG parameter depends on dt, so A is not affine in dt).

  Args: see `solve_transient_fea`.

  Returns: Array of size (num_time_steps + 1, num_dofs) with the concentration at the
    nodes at each time step (row 0 is the initial condition).
  """
  try:  # check in numpy only: under jit, even indexing a concrete array gives a tracer
    dts = np.asarray(time_step_sizes_star)
    if not np.allclose(dts, dts[0], rtol=1e-12, atol=0.0):
      raise ValueError("solve_transient_linear requires uniform time steps.")
  except jax.errors.TracerArrayConversionError:
    pass  # traced time steps: uniformity cannot be checked here
  dt_star = time_step_sizes_star[0]

  fixed = fea.bc["fixed_dofs"]
  dirich_on = fea.bc["dirichlet_values"]
  dirich_off = jnp.zeros_like(dirich_on)

  def inlet_scale(t):
    return inlet_scale_from_intervals(t, intervals_start, intervals_end)

  peclet_number = characteristic_length * char_velocity / eff_diffusivity

  # ---- assemble A = dr/dC_n and A - B = d r(C, C)/dC once per design
  # (r is linear, so the Jacobians do not depend on the evaluation point)
  zeros_elem = jnp.zeros(fea.mesh.elem_dof_mat.shape)
  res_args = (
    zeros_elem,
    zeros_elem,
    elem_velocity,
    peclet_number,
    fea.mesh.elem_node_coords,
    dt_star,
    fea.mesh.elem_diag_length,
  )
  vmap_axes = (0, 0, 0, 0, 0, None, 0)
  elem_a, elem_b = jax.vmap(
    jax.jacfwd(fea._compute_elem_residual, argnums=(0, 1)), in_axes=vmap_axes
  )(*res_args)  # {eij}: dr_i/dC_n_j and dr_i/dC_{n-1}_j  (= -B)

  def assemble(elem_mat):  # same construction as get_residual_and_tangent_stiffness
    return jax_sprs.BCOO(
      (elem_mat.flatten(), fea.node_id_jac),
      shape=(fea.mesh.num_dofs, fea.mesh.num_dofs),
    ).T

  a_bc = _bc.apply_dirichlet_bc(assemble(elem_a), fixed)
  a_minus_b = assemble(elem_a + elem_b)

  # ---- initial condition (as in solve_transient_fea)
  t0_phys = 0.0
  s0 = inlet_scale(t0_phys)
  c_init = init_concentration.at[fixed].set(s0 * dirich_on + (1.0 - s0) * dirich_off)

  def step(carry, k):
    c_prev, t_phys = carry
    t_next = t_phys + dt_phys[k]
    s = inlet_scale(t_next)
    dirich_step = s * dirich_on + (1.0 - s) * dirich_off

    c_g = c_prev.at[fixed].set(dirich_step)
    residual = (a_minus_b @ c_g).at[fixed].set(0.0)
    delta_c = _nlsolv.solve(a_bc, residual, params=fea.solver_settings["linear"])
    c = (c_g - delta_c).at[fixed].set(dirich_step)
    return (c, t_next), c

  num_time_steps = time_step_sizes_star.shape[0]
  _, c_steps = jax.lax.scan(
    step, (c_init, jnp.asarray(t0_phys)), jnp.arange(num_time_steps)
  )
  return jnp.concatenate([c_init[None, :], c_steps], axis=0)
