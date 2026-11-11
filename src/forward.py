"""Forward chain and objective of the inversion: design -> Brinkman flow -> transient SUPG transport -> chord-limited
sinograms -> mean-square misfit to the measured sinograms, with its gradient.

The notebook builds the meshes, solvers, boundary conditions, time grid, data and loss weights; make_objective takes
them as arguments and returns the jitted objective that optimize.optimize_design and the post-processing call:

  objective(x, flow_solver, q_alpha, q_D, beta, press_vel, filt=filt) -> (J, dJ/dx, press_vel, projected design, sinograms)
"""

import functools

import jax
import jax.numpy as jnp

import src.ct_radon_transform as _radon
import src.fe_fluid as _fea
import src.fe_transient_transport as _fea_transient_transport
import src.material as _mat
import src.solver as _solv
import src.utils as _utils


def make_objective(
  fluid_mesh,
  a_ref,
  default_filter,
  inv_permeability_ext: _utils.Extent,
  dirich_vals: jnp.ndarray,
  inlet_pos: list,
  fixed: jnp.ndarray,
  char_velocity: float,
  eff_diffusivity_ext: _utils.Extent,
  concentration_solver,
  init_concentration: jnp.ndarray,
  time_step_sizes_star: jnp.ndarray,
  characteristic_length: jnp.ndarray,
  time_step_size: jnp.ndarray,
  start_times: jnp.ndarray,
  end_times: jnp.ndarray,
  conc_ext: _utils.Extent,
  data: jnp.ndarray,
  grid: tuple[int, int],
  loss_angle_idx,
  frame_stride: int,
):
  """Jitted objective of the inversion (the notebook's objective_function).

  Args:
    fluid_mesh: fluid mesh (the dof map of the pressure-velocity field).
    a_ref: reference inlet scalings; the design variable of inlet k is a_k / a_ref[k].
    default_filter: density filter used when filt is None (the small filter: post-processing, floor_test).
    inv_permeability_ext: Brinkman range (alpha_min, alpha_max) of the convex RAMP.
    dirich_vals: Dirichlet values of the fluid BCs, in the order of fixed.
    inlet_pos: per inlet, the positions in fixed of its Dirichlet values (inlet-scale order).
    fixed: fixed dofs of the fluid BCs.
    char_velocity: velocity scale of the non-dimensional transport.
    eff_diffusivity_ext: diffusivity range (D_min, D) of the convex RAMP on 1 - gamma.
    concentration_solver: transient transport solver (fe_transient_transport.FEA).
    init_concentration: initial concentration with the inlet values set.
    time_step_sizes_star, time_step_size: non-dimensional and physical time steps.
    characteristic_length: per-element length of the Peclet number.
    start_times, end_times: injection intervals (s).
    conc_ext: range used to dimensionalise the concentration.
    data: measured sinograms (frames, detectors, all angles).
    grid: (nely, nelx) of the design; the projector uses the (nely + 1) x (nelx + 1) nodes.
    loss_angle_idx: angles (indices into data's last axis) the loss uses.
    frame_stride: time steps per frame.

  Returns:
    objective(mat_frac, flow_solver, ramp_penalty, diffusion_ramp_penalty, thresh_beta, init_press_vel=None, filt=None)
      -> (J, dJ/d(design) of shape (n, 1), press_vel, projected design, predicted sinograms at the loss angles).
      mat_frac is the design vector (num_elems + num_inlets, 1): element densities (0 fluid, 1 solid), then the
      inlet scalings divided by a_ref. ramp_penalty, diffusion_ramp_penalty and thresh_beta are q_alpha, q_D and beta
      of the continuation. init_press_vel is the Newton initial guess.
  """
  nely, nelx = grid

  @functools.partial(jax.jit, static_argnames=("flow_solver",))
  def objective_function(
    mat_frac: jnp.ndarray,
    flow_solver: _fea.FluidSolver,
    ramp_penalty: float,
    diffusion_ramp_penalty: float,
    thresh_beta: float,
    init_press_vel=None,
    filt=None,
  ):
    def objective_wrapper(design_var):
      num_elems = flow_solver.mesh.num_elems
      mat_frac = design_var[:num_elems, ]     # (num_elems,)
      scale_frac = design_var[num_elems:, 0]   # (number of inlets,)
      scale_var = scale_frac * jnp.asarray(a_ref)

      m = (default_filter if filt is None else filt) @ mat_frac
      mat_frac_thresh = _utils.threshold_filter(m, beta=thresh_beta)
      brinkman_penalty = _mat.compute_ramp_interpolation(
        prop=mat_frac_thresh,
        ramp_penalty=ramp_penalty,
        prop_ext=inv_permeability_ext,
        mode="convex",
      )

      vals_scaled = dirich_vals
      for _k, _pos in enumerate(inlet_pos):                             # one scale per inlet
        vals_scaled = vals_scaled.at[_pos].set(scale_var[_k] * dirich_vals[_pos])

      init_press_vel_s = init_press_vel.at[fixed].set(vals_scaled)

      press_vel = _solv.modified_newton_raphson_solve(
          flow_solver, init_press_vel_s, brinkman_penalty, vals_scaled
      )
      press_vel = press_vel.at[fixed].set(vals_scaled)

      press_vel_elem = press_vel[fluid_mesh.elem_dof_mat]
      elem_velocity = jnp.stack([press_vel_elem[:, 1::3], press_vel_elem[:, 2::3]], axis=2)  # (num_elems, num_nodes, 2)
      elem_velocity = elem_velocity.reshape(elem_velocity.shape[0], -1)  # (num_elems, 2*num_nodes)
      elem_velocity_star = elem_velocity / char_velocity

      eff_diffusivity_ramp = _mat.compute_ramp_interpolation(
        prop=(1 - mat_frac_thresh.reshape(-1,)),
        ramp_penalty=diffusion_ramp_penalty,
        prop_ext=eff_diffusivity_ext,
        mode="convex"
      )
      conc_t_hist = _fea_transient_transport.solve_transient_linear(
          concentration_solver,
          init_concentration,
          eff_diffusivity_ramp,
          time_step_sizes_star,
          characteristic_length,
          elem_velocity_star,
          char_velocity=char_velocity,
          dt_phys=time_step_size,
          intervals_start=start_times,
          intervals_end=end_times,
      )
      # dimensionalise the concentration
      conc_t_hist_d = conc_t_hist * conc_ext.range + conc_ext.min  # (num_time_steps + 1, n_dofs)

      def loss_fn(conc_t: jnp.ndarray):
        """conc_t: dimensional concentration at the frame times, shape (n_frames, n_dofs)."""
        n_det = data.shape[1]
        n_theta = data.shape[2]
        # only the loss angles are projected
        sinograms_pred = _radon.sinogram_from_conc(conc_t, nely + 1, nelx + 1, n_theta=n_theta, n_det=n_det, chord_limited=True,
                                                   angle_idx=loss_angle_idx)
        diff = sinograms_pred - data[:, :, loss_angle_idx]
        obj = jnp.mean(diff**2)
        return obj, sinograms_pred

      obj, sinograms_pred = loss_fn(conc_t_hist_d[::frame_stride])
      return obj, (press_vel, mat_frac_thresh, sinograms_pred)

    (obj, (press_vel, mat_frac_thresh, sinograms_pred)), d_obj = jax.value_and_grad(
      objective_wrapper, has_aux=True
    )(mat_frac)
    return obj, d_obj.reshape((-1, 1)), press_vel, mat_frac_thresh, sinograms_pred

  return objective_function
