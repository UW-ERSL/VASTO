"""MMA design loop for the inversion: continuation of (q_alpha, beta, q_D), J rescaling and the run outputs
(iter_log.txt, history.npz, designs.npz, design snapshots).

The notebook defines the problem (mesh, physics, boundary conditions, data) and the optimization settings
(Continuation, InletScales); this module only runs the loop. The objective is any callable with the
signature of the objective_function built by src/forward.make_objective:

  objective(x, fe, q_alpha, q_D, beta, press_vel, filt=filt) -> (J, dJ/dx, press_vel, projected design, sinograms)

The design vector x is (num_elems + num_inlets, 1): element densities in [0, 1], then the inlet scalings divided by
a_ref.
"""

import dataclasses
import pathlib
import resource as _resource
import time as _time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import src.fe_fluid as _fea
import src.mma as _mma
import src.postprocessing as _post


@dataclasses.dataclass(frozen=True)
class Continuation:
  """Linear continuation of the RAMP and projection parameters at the 0-based epoch e (paper Sec. 4.1).

    q_alpha(e) = max(q_alpha_floor, q_alpha_start - q_alpha_rate * e)
    beta(e)    = min(beta_cap,      beta_start    + beta_rate    * e)
    q_D(e)     = min(q_D_cap,       q_D_start     + q_D_rate     * e)
  """

  q_alpha_start: float
  q_alpha_rate: float
  q_alpha_floor: float
  beta_start: float
  beta_rate: float
  beta_cap: float
  q_D_start: float
  q_D_rate: float
  q_D_cap: float

  def __call__(self, e: int) -> tuple[float, float, float]:
    """(q_alpha, beta, q_D) at the 0-based epoch e."""
    return (max(self.q_alpha_floor, self.q_alpha_start - self.q_alpha_rate * e),
            min(self.beta_cap, self.beta_start + self.beta_rate * e),
            min(self.q_D_cap, self.q_D_start + self.q_D_rate * e))

  def end_iter(self) -> int:
    """Last epoch at which any schedule is still changing (100 for the paper's schedule). A schedule with rate 0, or
    already at its end value, contributes no continuation iterations."""
    span = lambda dist, rate: max(dist, 0.0) / rate if rate > 0 else 0.0  # noqa: E731
    return int(np.ceil(max(span(self.beta_cap - self.beta_start, self.beta_rate),
                           span(self.q_alpha_start - self.q_alpha_floor, self.q_alpha_rate),
                           span(self.q_D_cap - self.q_D_start, self.q_D_rate)) - 1e-9))


@dataclasses.dataclass(frozen=True)
class InletScales:
  """Inlet-velocity scalings optimised with the design, one per inlet, in the inlet order of the boundary conditions.
  The design variable is a / a_ref, bounded by [a_min / a_ref, a_max / a_ref]."""

  a0: np.ndarray
  a_ref: np.ndarray
  a_min: np.ndarray
  a_max: np.ndarray

  @property
  def num(self) -> int:
    return len(self.a_ref)


def optimize_design(
  objective,
  fe: _fea.FluidSolver,
  max_iter: int,
  continuation: Continuation,
  inlets: InletScales,
  filt,
  run_dir: pathlib.Path,
  grid: tuple[int, int],
  j_rescale_factor: float = 5.0,
  move_limit: float = 1e-2,
  kkt_tol: float = 1e-3,
  step_tol: float = 1e-3,
  plot_interval: int = 5,
  design_cmap=None,
  out_tag: str = "",
  num_time_steps: int | None = None,
):
  """MMA loop with the continuation, refreshed J scaling, a per-iteration log (run_dir/iter_log.txt) and the run-output
  files (history.npz, designs.npz). The loop runs until MMA's convergence flag is set (src/mma.update_mma): at
  max_iter, or when the KKT residual falls below kkt_tol or the design step below step_tol.

  Args:
    objective: the objective callable (signature in the module docstring).
    fe: fluid solver; its bc gives the Dirichlet values the pressure-velocity guess is reset to.
    max_iter: maximum number of MMA iterations.
    continuation: (q_alpha, beta, q_D) schedule.
    inlets: start values, reference and bounds of the inlet scalings.
    filt: density filter (BCOO) applied to the design at every iteration.
    run_dir: output folder (must contain outputs/ if plot_interval > 0).
    grid: (nely, nelx) of the design, for the snapshots and designs.npz.
    j_rescale_factor: J_scale is reset to the current J whenever J has dropped by this factor; 0 turns it off.
    move_limit, kkt_tol, step_tol: MMA settings.
    plot_interval: design snapshot every plot_interval iterations (and at it 1); 0 turns them off.
    design_cmap: colour map of the snapshots.
    out_tag: written to the log header only.
    num_time_steps: passed to the cost report.

  Returns:
    (mma_state, history).
  """
  run_dir = pathlib.Path(run_dir)
  nely, nelx = grid
  cont_end_iter = continuation.end_iter()
  num_inlets = inlets.num
  a0, a_ref, a_min, a_max = inlets.a0, inlets.a_ref, inlets.a_min, inlets.a_max
  LB, UB = a_min / a_ref, a_max / a_ref

  gamma0 = 0.5 * np.ones((fe.mesh.num_elems, 1))
  x_a0 = (a0 / a_ref).reshape(num_inlets, 1)
  design_var = np.vstack([gamma0, x_a0])
  num_design_var = design_var.shape[0]
  lower_bound = np.zeros((num_design_var, 1)); upper_bound = np.ones((num_design_var, 1))
  lower_bound[-num_inlets:, 0] = LB; upper_bound[-num_inlets:, 0] = UB
  mma_params = _mma.MMAParams(max_iter=max_iter, kkt_tol=kkt_tol, step_tol=step_tol, move_limit=move_limit,
                              num_design_var=num_design_var, num_cons=1,
                              lower_bound=lower_bound, upper_bound=upper_bound)
  mma_state = _mma.init_mma(design_var, mma_params)
  press_vel = jnp.zeros((fe.mesh.num_dofs,)).at[fe.bc["fixed_dofs"]].set(fe.bc["dirichlet_values"])

  history = {k: [] for k in ("epoch", "obj", "obj_scaled", "J_scale_iter", "kkt", "step_l2", "step_max", "rel_dJ", "beta", "q_alpha",
                             "q_D", "t_iter", "t_wall", "rss_peak_GB", "a", "vol_cons")}
  log = open(run_dir / "iter_log.txt", "a")
  timer = _post.EpochTimer()
  DES_RAW, DES_FILT, DES_PROJ = [], [], []
  t_start = _time.time(); J_scale = None; J_prev = None
  while not mma_state.is_converged:
    e = int(mma_state.epoch)                        # 0-based epoch of this evaluation
    q_alpha_e, beta_e, q_D_e = continuation(e)
    timer.start()
    J_val, grad_obj, press_vel, mat_frac, sinogram_pred = objective(
      mma_state.x, fe, q_alpha_e, q_D_e, beta_e, press_vel, filt=filt)
    t_it = timer.tick(grad_obj)
    J = float(J_val)
    if J_scale is None:                             # first objective
      J_scale = J if j_rescale_factor > 0 else (1.0 if 1.0 <= J <= 10.0 else J)
      history["J0"] = J; history["J_scale"] = J_scale; history["J_scale_refresh"] = [[e + 1, J]]
      log.write(f"[{out_tag}] J0 = {J:.6e}  J_scale = {J_scale:.6e}  j_rescale_factor = {j_rescale_factor}  a0 = {a0}  "
                f"bounds [{a_min}, {a_max}]  cont_end_iter = {cont_end_iter}\n")
    elif j_rescale_factor > 0 and J <= J_scale / j_rescale_factor:
      # J has dropped j_rescale_factor x since the last refresh: rescale objective and gradient from here on.
      # MMA's asymptotes depend only on the x history, so they are unaffected; kkt is reported in scaled units.
      J_scale = J; history["J_scale_refresh"].append([e + 1, J])
      log.write(f"[rescale] it {e + 1}: J_scale = {J_scale:.6e}\n")

    x_old = np.array(mma_state.x, copy=True)
    _g = x_old[:fe.mesh.num_elems, 0]                # design evaluated at this iteration: raw, filtered, projected
    DES_RAW.append(_g.copy())
    DES_FILT.append(np.asarray(filt @ jnp.asarray(_g), dtype=np.float32))
    DES_PROJ.append(np.asarray(mat_frac, dtype=np.float32).reshape(-1))
    press_vel = jax.lax.stop_gradient(press_vel).at[fe.bc["fixed_dofs"]].set(fe.bc["dirichlet_values"])
    mma_state = _mma.update_mma(mma_state, mma_params, np.asarray(J / J_scale), np.asarray(grad_obj) / J_scale,
                                np.array([[-1.0]]), np.zeros((1, num_design_var)))
    it = int(mma_state.epoch)                       # 1-based iteration just completed
    dx_max = float(np.max(np.abs(np.asarray(mma_state.x) - x_old)))
    rel = abs(J - J_prev) / abs(J_prev) if J_prev else np.inf
    J_prev = J
    kkt = float(getattr(mma_state, "kktnorm", np.nan))
    a_now = np.asarray(mma_state.x[-num_inlets:, 0]) * a_ref
    rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024**3     # macOS reports bytes
    for k, v in (("epoch", it), ("obj", J), ("obj_scaled", J / J_scale), ("J_scale_iter", J_scale), ("kkt", kkt),
                 ("step_l2", float(mma_state.change_design_var)), ("step_max", dx_max), ("rel_dJ", rel),
                 ("beta", beta_e), ("q_alpha", q_alpha_e), ("q_D", q_D_e),
                 ("t_iter", t_it), ("t_wall", _time.time() - t_start), ("rss_peak_GB", rss),
                 ("a", a_now.tolist()), ("vol_cons", -1.0)):
      history[k].append(v)
    line = (f"it {it:3d} | J {J:.6e} | dJ/J {rel:.2e} | max|dx| {dx_max:.3e} | kkt {kkt:.2e} | "
            f"beta {beta_e:.2f} q_a {q_alpha_e:.2f} q_D {q_D_e:.2f} | "
            f"a {np.round(a_now, 4)} | {t_it:.1f} s/it | peak {rss:.2f} GB")
    print(line, flush=True); log.write(line + "\n"); log.flush()

    if plot_interval and (it % plot_interval == 0 or it == 1):
      fig, ax = plt.subplots(figsize=(3, 3))
      ax.imshow(np.round(np.asarray(mat_frac)).reshape(nely, nelx), origin="lower", cmap=design_cmap,
                vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
      ax.set_title(f"it {it}")
      fig.savefig(run_dir / "outputs" / f"design_it{it:04d}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

  stop_reason = f"max_iter {max_iter}" if it >= max_iter else f"converged (kkt_tol {kkt_tol:g} / step_tol {step_tol:g})"
  history["stop_reason"] = stop_reason; history["iterations"] = it; history["cont_end_iter"] = cont_end_iter
  # ---- run-output standard: designs.npz and history.npz (row k = iteration k + 1) ----
  np.savez_compressed(run_dir / "designs.npz", raw=np.array(DES_RAW), filtered=np.array(DES_FILT), projected=np.array(DES_PROJ),
                      inlet_scale=np.array(history["a"]), nelx=nelx, nely=nely,
                      note="row k = design evaluated at iteration k+1; filtered = filter @ raw; projected = threshold(filtered, beta of that iteration)")
  np.savez_compressed(run_dir / "history.npz", iteration=np.array(history["epoch"]), J=np.array(history["obj"]),
                      J_scale=np.array(history["J_scale_iter"]), J_scaled=np.array(history["obj_scaled"]), kkt=np.array(history["kkt"]),
                      beta=np.array(history["beta"]), q_alpha=np.array(history["q_alpha"]), q_D=np.array(history["q_D"]),
                      inlet_scale=np.array(history["a"]),
                      step_max=np.array(history["step_max"]), step_l2=np.array(history["step_l2"]),
                      t_iter=np.array(history["t_iter"]), t_wall=np.array(history["t_wall"]), rss_peak_GB=np.array(history["rss_peak_GB"]),
                      J_scale_refresh=np.array(history["J_scale_refresh"]), J0=history["J0"], cont_end_iter=cont_end_iter)
  history["total_time_s"] = _time.time() - t_start
  log.write(f"[stop] {stop_reason} after {it} iterations, {history['total_time_s']:.0f} s\n"); log.close()
  print(f"[stop] {stop_reason} after {it} iterations, {history['total_time_s']:.0f} s")
  history["cost"] = timer.report(mesh=fe.mesh, num_time_steps=num_time_steps)
  return mma_state, history
