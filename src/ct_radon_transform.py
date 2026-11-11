import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path
from matplotlib.colors import Normalize

# ---------------------------
# Bilinear sampler (JAX)
# ---------------------------
def bilinear_sample(img, x, y):
    """
    Args:
        img: (H, W)
        x, y: same shape, in pixel coordinates [0, W-1], [0, H-1]
    Returns: 
        sampled values, same shape
    """
    H, W = img.shape

    # integer corners
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # clamp to image bounds
    x0 = jnp.clip(x0, 0, W - 1)
    x1 = jnp.clip(x1, 0, W - 1)
    y0 = jnp.clip(y0, 0, H - 1)
    y1 = jnp.clip(y1, 0, H - 1)

    # gather pixel values
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    # fractional parts
    fx = x - x0.astype(x.dtype)
    fy = y - y0.astype(y.dtype)

    # bilinear weights
    wa = (1.0 - fx) * (1.0 - fy)
    wb = fx * (1.0 - fy)
    wc = (1.0 - fx) * fy
    wd = fx * fy

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def bilinear_sample_zero(
    img: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """
    Differentiable bilinear sampler for a 2D image, with **zero padding** outside the image.

    This function samples `img` at (x, y) locations using bilinear interpolation.
    If a sample point lies outside the valid image domain, its value is set to 0.0
    (i.e., "air/background"), which matches typical CT/Radon conventions and avoids
    boundary-clamping artifacts.

    Parameters
    ----------
    img : jnp.ndarray
        2D image array of shape (H, W).
    x : jnp.ndarray
        Sample x-coordinates in **pixel index units**, where valid range is [0, W-1].
        Can be any shape; must match `y` shape (e.g., (n_det, n_samples)).
    y : jnp.ndarray
        Sample y-coordinates in **pixel index units**, where valid range is [0, H-1].
        Same shape as `x`.

    Returns
    -------
    jnp.ndarray
        Interpolated values at (x, y), same shape as `x` and `y`.

    Notes
    -----
    - The function is fully differentiable w.r.t. `img` and (x, y) in JAX.
    - We clip indices only for safe array indexing, but we apply a boolean mask
      so that out-of-bounds samples contribute exactly 0.0 (rather than sampling
      the nearest edge pixel).
    """

    # Image height and width.
    H, W = img.shape

    # Compute the integer pixel corner indices around each (x, y) sample location.
    # (x0, y0) is the "lower-left" corner; (x1, y1) is the "upper-right" corner.
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Identify which sample points lie inside the valid image extent.
    # Points outside will be forced to 0.0 after interpolation.
    in_bounds = (x >= 0.0) & (x <= (W - 1)) & (y >= 0.0) & (y <= (H - 1))

    # Clip indices for safe gather (prevents indexing errors at borders).
    # IMPORTANT: clipping alone would create boundary artifacts (edge replication),
    # so we still apply `in_bounds` to zero-out samples that were actually out-of-range.
    x0c = jnp.clip(x0, 0, W - 1)
    x1c = jnp.clip(x1, 0, W - 1)
    y0c = jnp.clip(y0, 0, H - 1)
    y1c = jnp.clip(y1, 0, H - 1)

    # Gather image values at the four surrounding corners:
    # Ia: (x0, y0), Ib: (x1, y0), Ic: (x0, y1), Id: (x1, y1)
    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]

    # Fractional offsets within the cell (in [0, 1] typically).
    # These determine interpolation weights.
    fx = x - x0.astype(x.dtype)
    fy = y - y0.astype(y.dtype)

    # Bilinear interpolation weights for each corner.
    wa = (1.0 - fx) * (1.0 - fy)
    wb = fx * (1.0 - fy)
    wc = (1.0 - fx) * fy
    wd = fx * fy

    # Weighted sum of the four corner values.
    val = wa * Ia + wb * Ib + wc * Ic + wd * Id

    # Enforce zero outside the image domain.
    # This makes rays that miss the object contribute zero attenuation (as in CT).
    return jnp.where(in_bounds, val, 0.0)


def bilinear_sample_zero_tent(img: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Same as `bilinear_sample_zero` in exact arithmetic, but each corner weight is the
    tent max(0, 1 - |x - i|) of the distance to the index i actually gathered. A
    floor() that rounds to the neighbouring index at an exact grid line then gets
    weight 0 instead of reading the wrong pixel. Used by the chord-limited projector.
    """
    H, W = img.shape
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1, y1 = x0 + 1, y0 + 1
    in_bounds = (x >= 0.0) & (x <= (W - 1)) & (y >= 0.0) & (y <= (H - 1))

    tent = lambda t, i: jnp.maximum(0.0, 1.0 - jnp.abs(t - i.astype(t.dtype)))
    wx0, wx1 = tent(x, x0), tent(x, x1)
    wy0, wy1 = tent(y, y0), tent(y, y1)

    x0c, x1c = jnp.clip(x0, 0, W - 1), jnp.clip(x1, 0, W - 1)
    y0c, y1c = jnp.clip(y0, 0, H - 1), jnp.clip(y1, 0, H - 1)
    val = (wx0 * wy0 * img[y0c, x0c] + wx1 * wy0 * img[y0c, x1c]
           + wx0 * wy1 * img[y1c, x0c] + wx1 * wy1 * img[y1c, x1c])
    return jnp.where(in_bounds, val, 0.0)


def ray_chords(angles_deg, s_vals, H: int, W: int):
    """
    Entry/exit line parameters [l_lo, l_hi] of each ray through the image square, in
    the parameterisation of `radon_jax` (x' = s cos - l sin, y' = s sin + l cos, pixel
    units, image-centred). Computed in NumPy.

    Args:
        angles_deg: (N_theta,) angles in degrees (concrete values).
        s_vals: (n_det,) detector positions in pixel units.
        H, W: image size in pixels (nodes).

    Returns: (l_lo, l_hi), each (n_det, N_theta). Rays that miss the square get
        l_lo = l_hi = 0, i.e. zero length.
    """
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    th = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    S, T = np.meshgrid(np.asarray(s_vals, dtype=np.float64), th, indexing="ij")
    c, s = np.cos(T), np.sin(T)

    lo = np.full(S.shape, -np.inf)
    hi = np.full(S.shape, np.inf)
    for p0, d, half in ((S * c, -s, cx), (S * s, c, cy)):   # coordinate = p0 + d * l
        par = np.abs(d) < 1e-14
        d_safe = np.where(par, 1.0, d)
        t1, t2 = (-half - p0) / d_safe, (half - p0) / d_safe
        inside = np.abs(p0) <= half
        l1 = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
        l2 = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
        lo, hi = np.maximum(lo, l1), np.minimum(hi, l2)

    miss = ~(hi > lo)                       # also catches inf - inf / NaN cases
    return np.where(miss, 0.0, lo), np.where(miss, 0.0, hi)


# ---------------------------
# Differentiable Radon
# ---------------------------
def radon_jax(
    image: jnp.ndarray,
    angles: jnp.ndarray,
    n_det: int | None = None,
    n_samples: int | None = None,
    pixel_size: float | None = None,
    chord_limited: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Differentiable parallel-beam Radon transform in JAX.
    Args:
        image:  (H, W) array (jnp.ndarray, float32/64)
        angles: (N_theta,) in *degrees*  (e.g. jnp.linspace(0, 180, N_theta, endpoint=False))
        n_det:  number of detector bins (default: max(H, W))
        n_samples: number of sample points along each ray (default: ceil(2R) + 1)
        pixel_size: size of each pixel (default: 1 / (max(H, W) - 1))
        chord_limited: If False (default), samples are spread uniformly over [-R, R]
            (the image diagonal) for every ray, as before. The field is generally not
            zero on the image border, so the border jump falls between samples and
            the quadrature is only first order in the sample spacing.
            If True, each ray's entry/exit points on the image square are computed
            analytically and the n_samples midpoints are placed inside that chord, so
            the border falls on a sample-interval boundary (second order). Rays that
            miss the square return exactly 0. The detector layout (s_vals) and
            angles are unchanged. Requires concrete `angles` (NumPy or a non-traced
            array): the sample positions are precomputed in NumPy and passed to XLA as
            fixed arrays, because under jax.jit XLA may otherwise recompute a position
            separately for the gather index and the weight with different rounding,
            reading the wrong pixel when a sample lies exactly on a grid line.
    Returns:
        sinogram: (n_det, N_theta)
        s_vals:   (n_det,) detector coordinates (in pixel units)
    """
    if chord_limited:
        return _radon_jax_chord(image, angles, n_det, n_samples, pixel_size)

    H, W = image.shape
    if H != W:
        raise ValueError(
            "radon_jax assumes square pixels; the ray parameterisation is in "
            f"pixel coordinates and is not valid for H != W (got {H}, {W})."
        )
    if n_det is None:
        n_det = max(H, W)

    # Physical size of one pixel. The domain is the unit square, so spacing is
    # 1/(n-1) between nodes. Without this, the line integral is returned in
    # pixel units and its magnitude scales with mesh resolution.
    if pixel_size is None:
        pixel_size = 1.0 / (max(H, W) - 1)

    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    R = float(np.hypot(cx, cy))

    if n_samples is None:
        n_samples = int(np.ceil(2.0 * R)) + 1     # ~1 sample per pixel

    angles = jnp.deg2rad(angles)          # <-- RESTORE
        
    # detector positions (signed distance from centre)
    s_vals = jnp.linspace(-R, R, n_det)

    # samples along each ray (line parameter l)
    # Midpoint rule: sample at interval centres so the quadrature is exact for a
    # uniform field. Sampling the endpoints (linspace) counts each end at full
    # weight and overcounts the integral by one interval.
    dl = 2.0 * R / n_samples                              # pixels
    l_vals = -R + dl * (jnp.arange(n_samples) + 0.5)
    dl_phys = dl * pixel_size                             # physical length


    # grids: shape (n_det, n_samples)
    s_grid, l_grid = jnp.meshgrid(s_vals, l_vals, indexing="ij")

    def proj_at_angle(theta: jnp.ndarray) -> jnp.ndarray:
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        # line parameterization in *image-centred* coordinates:
        # x' = s*c - l*s
        # y' = s*s + l*c
        x_prime = s_grid * c - l_grid * s
        y_prime = s_grid * s + l_grid * c

        # convert to pixel indices
        x_pix = x_prime + cx
        y_pix = y_prime + cy
        
        #samples = bilinear_sample(image, x_pix, y_pix)
        samples = bilinear_sample_zero(image, x_pix, y_pix)
        
        # approximate line integral: sum over l * dl
        return jnp.sum(samples, axis=-1) * dl_phys   # (n_det,)

    # vectorize over angles
    sinogram = jax.vmap(proj_at_angle)(angles)   # (N_theta, n_det)
    sinogram = jnp.swapaxes(sinogram, 0, 1)      # (n_det, N_theta)

    return sinogram, s_vals


def _radon_jax_chord(image, angles, n_det, n_samples, pixel_size):
    """Chord-limited branch of `radon_jax` (see its docstring)."""
    H, W = image.shape
    if H != W:
        raise ValueError(
            "radon_jax assumes square pixels; the ray parameterisation is in "
            f"pixel coordinates and is not valid for H != W (got {H}, {W})."
        )
    if n_det is None:
        n_det = max(H, W)
    if pixel_size is None:
        pixel_size = 1.0 / (max(H, W) - 1)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    R = float(np.hypot(cx, cy))
    if n_samples is None:
        n_samples = int(np.ceil(2.0 * R)) + 1

    try:
        angles_np = np.asarray(angles, dtype=np.float64)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError) as e:
        raise TypeError(
            "radon_jax(chord_limited=True) needs concrete angles (e.g. np.linspace), "
            "not values traced inside jax.jit."
        ) from e

    # Same detector layout as the default branch (jnp.linspace(-R, R, n_det)), computed
    # in NumPy so it is concrete under jax.jit. Same formula as jnp.linspace:
    # start * (1 - t) + stop * t for i < n_det - 1, and exactly `stop` for the last bin.
    # It matches jnp.linspace to round-off (measured <= 3e-14 pixel units for n = 80,
    # 120, 160 with n_det = 115), not bit for bit: XLA evaluates jnp.linspace with
    # fused multiply-adds, which NumPy does not reproduce.
    t = np.arange(n_det - 1, dtype=np.float64) / (n_det - 1)
    s_np = np.concatenate([(-R) * (1.0 - t) + R * t, [R]])
    s_vals = jnp.asarray(s_np)
    lo, hi = ray_chords(angles_np, s_np, H, W)                  # (n_det, N_theta)

    th = np.deg2rad(angles_np)
    frac = (np.arange(n_samples) + 0.5) / n_samples             # midpoints in [0, 1]
    lo_t, len_t = lo.T, (hi - lo).T                              # (N_theta, n_det)
    c, s = np.cos(th)[:, None, None], np.sin(th)[:, None, None]
    l = lo_t[:, :, None] + len_t[:, :, None] * frac[None, None, :]
    x_pix = s_np[None, :, None] * c - l * s + cx                 # (N_theta, n_det, n_samples)
    y_pix = s_np[None, :, None] * s + l * c + cy
    dl_phys = len_t / n_samples * pixel_size                      # (N_theta, n_det); 0 for misses

    samples = bilinear_sample_zero_tent(image, jnp.asarray(x_pix), jnp.asarray(y_pix))
    sinogram = jnp.sum(samples, axis=-1) * jnp.asarray(dl_phys)   # (N_theta, n_det)
    return jnp.swapaxes(sinogram, 0, 1), s_vals

def save_sinogram_gif(sinograms: np.ndarray, out_path_sino: Path, times: np.ndarray, theta: np.ndarray, fps: int = 5):
    """Saves a GIF animation of sinograms over time.
    Args:
        sinograms: np.ndarray of shape (n_frames, n_s, n_theta)
        out_path_sino: Path to save the GIF
        times: total number of time steps
        theta: angle of the orientaion of the emitted rays. (N_theta,) in *degrees*  (e.g. jnp.linspace(0, 180, N_theta, endpoint=False))
        fps: frames per second
    Returns:
        None
    """
    frames_sino = []

    # Fix color scale over all times to avoid flicker
    vmin = float(sinograms.min())
    vmax = float(sinograms.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig_s, ax_s = plt.subplots(figsize=(4.8, 3.6), dpi=140)
    cb_s = None

    for k, t in enumerate(times):
        S = sinograms[k]  # shape (n_s, n_theta)

        ax_s.clear()
        im_s = ax_s.imshow(
            S,
            origin="lower",
            aspect="auto",
            extent=[theta[0], theta[-1], 0, S.shape[0]],
            cmap="viridis",
            norm=norm,
            interpolation="bilinear",
        )
        ax_s.set_xlabel("theta (degrees)")
        ax_s.set_ylabel("detector coordinate s (index)")
        ax_s.set_title(f"Sinogram at t = {t:0.3f}")

        if cb_s is None:
            cb_s = fig_s.colorbar(im_s, ax=ax_s, label="line integral of c")

        fig_s.tight_layout()

        # capture frame (same pattern as your concentration GIF)
        fig_s.canvas.draw()
        w, h = fig_s.canvas.get_width_height()
        buf = np.frombuffer(fig_s.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape(h, w, 4)[..., :3]
        frames_sino.append(frame.copy())

    plt.close(fig_s)

    imageio.mimsave(out_path_sino, frames_sino, fps=fps)
    print(f"Saved sinogram GIF -> {out_path_sino.resolve()}")

def sinogram_from_conc(t_hist_np, ny, nx, n_theta=180, n_det=200,
                       alpha=1., pixel_size=None, n_samples=None, chord_limited=False,
                       angle_idx=None):
    """
        Computes sinograms from the concentration field at all the time steps.
        Args:
            t_hist_np: Concentration data at each time step. shape (n_frames, s_len, n_theta)
            ny: number of nodes in y direction:
            nx: number of nodes in x direction
            n_theta: n_theta angles sampled between from 0 and 180 degrees. (the source will be rotated n_theta times and projections are captured.)
            alpha: is the mass attenuation coefficient of iodine, converting concentration into linear X-ray attenuation.
            chord_limited: Passed to `radon_jax`. Default False keeps the original projector.
            angle_idx: Optional concrete integer array of column indices into the full
              (n_theta-angle) sinogram. If given, only those projections are computed and the
              result equals the full sinogram's [:, :, angle_idx]. Default None: all angles.
        Returns:
            sinograms: Sinograms at all the time steps. shape (n_frames, s_len, n_theta),
              or (n_frames, s_len, len(angle_idx)) if angle_idx is given.
    """
    if pixel_size is None:
        pixel_size = 1.0 / (max(ny, nx) - 1)
    if chord_limited:
        # concrete angles (the chord geometry is precomputed in NumPy)
        theta = np.linspace(0.0, 180.0, n_theta, endpoint=False)
    else:
        theta = jnp.linspace(0.0, 180.0, n_theta, endpoint=False)
    imgs = alpha * t_hist_np.reshape(t_hist_np.shape[0], ny, nx)
    if angle_idx is None:
        one = lambda img: radon_jax(img, angles=theta, n_det=n_det, n_samples=n_samples,
                                    pixel_size=pixel_size,
                                    chord_limited=chord_limited)[0][::-1, ::-1]
    else:
        # The full output reverses the angle axis ([..., ::-1]), so its column j holds
        # theta[n_theta - 1 - j]. Project only those angles; reverse the detector axis only.
        idx = np.asarray(angle_idx)
        theta_sel = theta[n_theta - 1 - idx]
        one = lambda img: radon_jax(img, angles=theta_sel, n_det=n_det, n_samples=n_samples,
                                    pixel_size=pixel_size,
                                    chord_limited=chord_limited)[0][::-1, :]
    return jax.vmap(one)(imgs)
    