from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

@dataclass
class Settings:
    # FFT preprocessing
    remove_mean: bool = True
    use_hann_window: bool = True

    # Radial spectrum
    n_radial_bins: int = 120
    min_q_nm_inv: float = 0.0
    max_q_nm_inv: Optional[float] = None

    # FFT display (dB)
    fft_db_vmin: Optional[float] = None
    fft_db_vmax: Optional[float] = None
    fft_db_floor_db: float = -80.0
    fft_cmap: str = "viridis"

    # Real-space display
    real_space_cmap: str = "afmhot"
    real_vmin: Optional[float] = None
    real_vmax: Optional[float] = None
    real_percentile_low: float = 1.0
    real_percentile_high: float = 99.0

    # Residual display
    residual_vlim: Optional[float] = None
    residual_percentile: float = 99.0
    residual_cmap: str = "RdBu_r"

    # Radial spectrum display
    plot_log_y: bool = True
    plot_normalized_spectrum: bool = True
    spectrum_q_min: float = 0.0
    spectrum_q_max: Optional[float] = None

    # High-frequency metric threshold
    high_q_threshold_nm_inv: float = 0.25

    # Figure / output
    save_figures: bool = True
    show_figures: bool = True
    dpi: int = 300
    figure_format: str = "pdf"
    save_csv: bool = True
    save_image_matrices: bool = True

def load_xyz_file(filename):
    """
    Load a whitespace-delimited XYZ file and reshape it into a 2D grid.

    Parameters
    ----------
    filename : str or Path
        Path to the XYZ file.

    Returns
    -------
    x_unique : ndarray
        Sorted unique X coordinates.
    y_unique : ndarray
        Sorted unique Y coordinates.
    image : ndarray
        2D array of Z values on the (y, x) grid.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File not found:\n{filename}")

    try:
        data = np.loadtxt(filename)
    except ValueError:
        data = np.loadtxt(filename, skiprows=1)

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"{filename.name}: expected at least 3 columns (X, Y, Z), "
            f"but found shape {data.shape}."
        )

    x, y, z = (np.asarray(data[:, i], dtype=float) for i in range(3))

    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        raise ValueError(f"{filename.name}: X, Y, or Z contains NaN/Inf values.")

    x_unique, y_unique = np.unique(x), np.unique(y)
    nx, ny = len(x_unique), len(y_unique)

    if nx * ny != len(z):
        raise ValueError(
            f"{filename.name}: incomplete grid. Unique X={nx}, Y={ny}, "
            f"but X*Y={nx * ny} and rows={len(z)}."
        )

    image = np.full((ny, nx), np.nan, dtype=float)
    ix = np.searchsorted(x_unique, x)
    iy = np.searchsorted(y_unique, y)
    image[iy, ix] = z

    if np.isnan(image).any():
        raise ValueError(f"{filename.name}: missing grid points detected.")

    return x_unique, y_unique, image

def median_spacing(coords):
    """
    Compute the median spacing between sorted, strictly increasing coordinates.

    Parameters
    ----------
    coords : array_like
        1D array of coordinate values.
    
    Returns
    -------
    float
        Median spacing between consecutive coordinates.
    """
    if len(coords) < 2:
        raise ValueError("Coordinate array must contain at least two points.")
    diffs = np.diff(coords)
    if np.any(diffs <= 0):
        raise ValueError("Coordinates must be strictly increasing after sorting.")
    return float(np.median(diffs))

def validate_matching_grids(
    x_raw, y_raw, raw, x_den, y_den, den, verbose=True):
    """Check that RAW and DENOISED images describe the same spatial grid.

    Parameters
    ----------
    x_raw, y_raw : array_like
        X and Y coordinates of the RAW grid.
    raw : ndarray
        RAW 2D image.
    x_den, y_den : array_like
        X and Y coordinates of the DENOISED grid.
    den : ndarray
        DENOISED 2D image.
    verbose : bool, optional
        If True, print grid diagnostics, by default True.
    
    Returns
    -------
    x : ndarray
        Common X coordinates.
    y : ndarray
        Common Y coordinates.
    """
    if verbose:
        print("\nRAW grid")
        print(f"  size   : {len(x_raw)} x {len(y_raw)}")
        print(f"  dx, dy : {median_spacing(x_raw):.9f}, {median_spacing(y_raw):.9f} nm")
        print("\nDENOISED grid")
        print(f"  size   : {len(x_den)} x {len(y_den)}")
        print(f"  dx, dy : {median_spacing(x_den):.9f}, {median_spacing(y_den):.9f} nm")

    if raw.shape != den.shape:
        raise ValueError(
            f"RAW and DENOISED image dimensions do not match: "
            f"RAW={raw.shape}, DENOISED={den.shape}."
        )

    dx_raw, dy_raw = median_spacing(x_raw), median_spacing(y_raw)
    dx_den, dy_den = median_spacing(x_den), median_spacing(y_den)

    checks = [
        (np.isclose(dx_raw, dx_den, atol=1e-4), "X spacing differ too much"),
        (np.isclose(dy_raw, dy_den, atol=1e-4), "Y spacing differ too much"),
        (np.isclose(x_raw[0], x_den[0], atol=1e-2), "X minimum values differ too much"),
        (np.isclose(y_raw[0], y_den[0], atol=1e-2), "Y minimum values differ too much"),
        (np.isclose(x_raw[-1], x_den[-1], atol=1e-2), "X maximum values differ too much"),
        (np.isclose(y_raw[-1], y_den[-1], atol=1e-2), "Y maximum values differ too much"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(f"RAW and DENOISED {msg}.")

    xr = (x_raw - x_raw[0])/(x_raw[-1] - x_raw[0])
    xd = (x_den - x_den[0])/(x_den[-1] - x_den[0])
    yr = (y_raw - y_raw[0])/(y_raw[-1] - y_raw[0])
    yd = (y_den - y_den[0])/(y_den[-1] - y_den[0])

    if not np.allclose(xr, xd, atol=1e-6):
        raise ValueError("RAW and DENOISED X grid structures differ.")
    if not np.allclose(yr, yd, atol=1e-6):
        raise ValueError("RAW and DENOISED Y grid structures differ.")

    if verbose:
        print("\nGrid structure is equivalent. No interpolation is used.")

    return x_raw, y_raw 

def hann_window_2d(shape):
    """
    Construct a 2D Hann window from the outer product of two 1D windows.

    Parameters
    ----------
    shape : tuple of int
        Shape (ny, nx) of the desired window.
    
    Returns
    -------
    ndarray
        2D Hann window of the given shape.
    """
    ny, nx = shape
    return np.outer(np.hanning(ny), np.hanning(nx))

def compute_fft_power(image, dx_nm, dy_nm, remove_mean=True, use_hann=True):
    """
    Compute the 2D FFT power spectrum of an image, with optional preprocessing.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    dx_nm : float
        Pixel spacing along X, in nanometers.
    dy_nm : float
        Pixel spacing along Y, in nanometers.
    remove_mean : bool, optional
        If True, subtract the mean before transforming, by default True.
    use_hann : bool, optional
        If True, apply a 2D Hann window before transforming, by default True.

    Returns
    -------
    fft : ndarray
        Shifted complex FFT of the preprocessed image.
    power : ndarray
        FFT power spectrum (squared magnitude).
    qx : ndarray
        Spatial frequency coordinates along X.
    qy : ndarray
        Spatial frequency coordinates along Y.
    """
    work = np.asarray(image, dtype=float).copy()

    if remove_mean:
        work -= np.mean(work)
    if use_hann:
        work *= hann_window_2d(work.shape)

    fft = np.fft.fftshift(np.fft.fft2(work))
    power = np.abs(fft) ** 2

    ny, nx = image.shape
    qx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_nm))
    qy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy_nm))

    return fft, power, qx, qy

def power_to_db(power, floor_db=-80.0):
    """
    Convert a power spectrum to a floored decibel scale, normalized to its peak.

    Parameters
    ----------
    power : ndarray
        Input power spectrum.
    floor_db : float, optional
        Minimum dB value to clip to, by default -80.0.
    
    Returns
    -------
    ndarray
        Power spectrum in dB, floored at floor_db.
    """
    p = np.asarray(power, dtype=float)
    pmax = np.nanmax(p)
    if pmax <= 0:
        raise ValueError("FFT power contains no positive values.")
    db = 10.0 * np.log10(np.maximum(p, pmax * 10.0 ** (floor_db / 10.0)) / pmax)
    return np.maximum(db, floor_db)

def radial_average(power, qx, qy, n_bins=120, q_min=0.0, q_max=None):
    """
    Compute the radially averaged power spectrum over spatial frequency bins.

    Parameters
    ----------
    power : ndarray
        2D power spectrum.
    qx : ndarray
        Spatial frequency coordinates along X.
    qy : ndarray
        Spatial frequency coordinates along Y.
    n_bins : int, optional
        Number of radial bins, by default 120.
    q_min : float, optional
        Minimum spatial frequency to include, by default 0.0.
    q_max : float, optional
        Maximum spatial frequency to include. Defaults to the maximum available.

    Returns
    -------
    q : ndarray
        Mean spatial frequency per bin.
    p_mean : ndarray
        Mean power per bin.
    p_std : ndarray
        Standard deviation of power per bin.
    counts : ndarray
        Number of pixels averaged per bin.
    """
    qx_grid, qy_grid = np.meshgrid(qx, qy)
    q = np.sqrt(qx_grid ** 2 + qy_grid ** 2)

    q_flat, p_flat = q.ravel(), np.asarray(power, dtype=float).ravel()
    valid = np.isfinite(q_flat) & np.isfinite(p_flat)
    q_flat, p_flat = q_flat[valid], p_flat[valid]

    if q_max is None:
        q_max = float(np.max(q_flat))
    if q_max <= q_min:
        raise ValueError("Invalid q range for radial averaging.")

    mask = (q_flat >= q_min) & (q_flat <= q_max) & (p_flat >= 0)
    q_flat, p_flat = q_flat[mask], p_flat[mask]

    edges = np.linspace(q_min, q_max, n_bins + 1)
    idx = np.digitize(q_flat, edges) - 1

    q_out, p_mean, p_std, counts = [], [], [], []
    for i in range(n_bins):
        m = idx == i
        if not np.any(m):
            continue
        q_out.append(np.mean(q_flat[m]))
        p_mean.append(np.mean(p_flat[m]))
        p_std.append(np.std(p_flat[m]))
        counts.append(np.sum(m))

    return (np.asarray(q_out), np.asarray(p_mean),
            np.asarray(p_std), np.asarray(counts, dtype=int))

def save_xyz(filename, x, y, image):
    """
    Save a 2D image and its coordinate grid to a whitespace-delimited XYZ file.

    Parameters
    ----------
    filename : str or Path
        Output file path.
    x : array_like
        X coordinates.
    y : array_like
        Y coordinates.
    image : ndarray
        2D array of Z values on the (y, x) grid.
    """
    xx, yy = np.meshgrid(x, y)
    out = np.column_stack([xx.ravel(), yy.ravel(), image.ravel()])
    np.savetxt(filename, out, fmt="%.9f %.9f %.12g", header="X Y Z", comments="")

class AFMComparison:
    def __init__(self, raw_file, denoised_file, output_dir="afm_comparison_results", **settings):
        """
        Initialize a RAW vs. DENOISED AFM image comparison.

        Parameters
        ----------
        raw_file : str or Path
            Path to the RAW XYZ file.
        denoised_file : str or Path
            Path to the DENOISED XYZ file.
        output_dir : str or Path, optional
            Directory for saved outputs, by default "afm_comparison_results".
        **settings
            Keyword overrides for Settings fields.
        """
        self.raw_file = Path(raw_file)
        self.denoised_file = Path(denoised_file)
        self.output_dir = Path(output_dir)
        self.settings = Settings(**settings)

        # populated by .load()
        self.x = self.y = None
        self.raw = self.den = self.residual = None
        self.dx = self.dy = None

        # populated by .run_fft()
        self.power_raw = self.power_den = None
        self.qx = self.qy = None
        self.radial_raw = self.radial_den = None

        # populated by .compute_metrics()
        self.metrics: Dict[str, Any] = {}

        # figure handles
        self.figures: Dict[str, plt.Figure] = {}

        # cached display limits
        self._real_limits = None
        self._residual_limit = None
        self._fft_limits = None

    def load(self, verbose=True):
        """
        Load and validate the RAW and DENOISED images, and compute their residual.

        Parameters
        ----------
        verbose : bool, optional
            If True, print grid diagnostics, by default True.
        
        Returns
        -------
        AFMComparison
            Self, with x, y, raw, den, residual, dx, dy populated.
        """
        x_raw, y_raw, raw = load_xyz_file(self.raw_file)
        x_den, y_den, den = load_xyz_file(self.denoised_file)

        self.x, self.y = validate_matching_grids(
            x_raw, y_raw, raw, x_den, y_den, den, verbose=verbose
        )
        self.raw, self.den = raw, den
        self.residual = raw - den
        self.dx, self.dy = median_spacing(self.x), median_spacing(self.y)

        if verbose:
            print("\nResidual computed as RAW - DENOISED (point-by-point, no interpolation).")
        return self

    def run_fft(self):
        """
        Compute FFT power spectra and radially averaged spectra for both images.

        Returns
        -------
        AFMComparison
            Self, with power_raw, power_den, qx, qy, radial_raw, radial_den populated.
        """
        self._ensure_loaded()
        s = self.settings

        _, self.power_raw, self.qx, self.qy = compute_fft_power(
            self.raw, self.dx, self.dy, s.remove_mean, s.use_hann_window
        )
        _, self.power_den, qx_den, qy_den = compute_fft_power(
            self.den, self.dx, self.dy, s.remove_mean, s.use_hann_window
        )
        if not (np.allclose(self.qx, qx_den) and np.allclose(self.qy, qy_den)):
            raise ValueError("FFT q grids do not match between RAW and DENOISED.")

        self.radial_raw = radial_average(
            self.power_raw, self.qx, self.qy,
            n_bins=s.n_radial_bins, q_min=s.min_q_nm_inv, q_max=s.max_q_nm_inv,
        )
        self.radial_den = radial_average(
            self.power_den, self.qx, self.qy,
            n_bins=s.n_radial_bins, q_min=s.min_q_nm_inv, q_max=s.max_q_nm_inv,
        )
        return self

    def compute_metrics(self):
        """
        Compute summary statistics and similarity metrics between RAW and DENOISED images.

        Returns
        -------
        dict
            Dictionary of computed metrics, including means, standard deviations,
            residual RMS, pixel correlation, high-frequency power, and grid info.
        """
        self._ensure_loaded()
        if self.radial_raw is None:
            self.run_fft()

        s = self.settings
        raw, den, residual = self.raw, self.den, self.residual

        raw_centered = raw - np.mean(raw)
        den_centered = den - np.mean(den)
        denom = np.sqrt(np.sum(raw_centered ** 2) * np.sum(den_centered ** 2))
        correlation = float(np.sum(raw_centered * den_centered) / denom) if denom > 0 else np.nan

        q_raw, p_raw, _, _ = self.radial_raw
        q_den, p_den, _, _ = self.radial_den
        thr = s.high_q_threshold_nm_inv

        raw_high = (float(np.trapz(p_raw[q_raw >= thr], q_raw[q_raw >= thr]))
                    if np.any(q_raw >= thr) else np.nan)
        den_high = (float(np.trapz(p_den[q_den >= thr], q_den[q_den >= thr]))
                    if np.any(q_den >= thr) else np.nan)

        self.metrics = {
            "raw_mean": float(np.mean(raw)),
            "denoised_mean": float(np.mean(den)),
            "raw_std": float(np.std(raw)),
            "denoised_std": float(np.std(den)),
            "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "residual_std": float(np.std(residual)),
            "raw_denoised_pixel_correlation": correlation,
            f"raw_high_q_integrated_power_above_{thr}_nm^-1": raw_high,
            f"denoised_high_q_integrated_power_above_{thr}_nm^-1": den_high,
            "dx_nm": self.dx,
            "dy_nm": self.dy,
            "nx": raw.shape[1],
            "ny": raw.shape[0],
            "field_of_view_x_nm": self.x[-1] - self.x[0],
            "field_of_view_y_nm": self.y[-1] - self.y[0],
        }
        return self.metrics

    def plot_real_space(self):
        """
        Plot RAW, DENOISED, and residual images side by side in real space.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the three real-space panels.
        """
        self._ensure_loaded()
        s = self.settings
        x, y, raw, den, residual = self.x, self.y, self.raw, self.den, self.residual
        extent = [x[0], x[-1], y[0], y[-1]]

        vmin, vmax = self._get_real_limits()
        rlim = self._get_residual_limit()

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

        im0 = axes[0].imshow(
            raw, origin="lower", extent=extent, cmap=s.real_space_cmap,
            vmin=vmin, vmax=vmax, aspect="equal")
        axes[0].set_title("RAW")
        axes[0].set_xlabel("X (nm)"); axes[0].set_ylabel("Y (nm)")
        fig.colorbar(im0, ax=axes[0], label="Z")

        im1 = axes[1].imshow(
            den, origin="lower", extent=extent, cmap=s.real_space_cmap,
            vmin=vmin, vmax=vmax, aspect="equal")
        axes[1].set_title("DENOISED")
        axes[1].set_xlabel("X (nm)"); axes[1].set_ylabel("Y (nm)")
        fig.colorbar(im1, ax=axes[1], label="Z")

        im2 = axes[2].imshow(
            residual, origin="lower", extent=extent, cmap=s.residual_cmap,
            norm=TwoSlopeNorm(vmin=-rlim, vcenter=0.0, vmax=rlim), aspect="equal")
        axes[2].set_title("RESIDUAL: RAW - DENOISED")
        axes[2].set_xlabel("X (nm)"); axes[2].set_ylabel("Y (nm)")
        fig.colorbar(im2, ax=axes[2], label="RAW - DENOISED")

        self._save_figure(fig, "01_real_space_same_scale")
        self.figures["real_space"] = fig
        return fig

    def plot_fft(self):
        """
        Plot the FFT power spectra (in dB) of the RAW and DENOISED images side by side.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the two FFT panels.
        """
        if self.power_raw is None:
            self.run_fft()
        s = self.settings
        qx, qy = self.qx, self.qy

        db_raw = power_to_db(self.power_raw, s.fft_db_floor_db)
        db_den = power_to_db(self.power_den, s.fft_db_floor_db)

        common_min, common_max = s.fft_db_vmin, s.fft_db_vmax
        if common_min is None:
            common_min = float(np.percentile(np.concatenate([db_raw.ravel(), db_den.ravel()]), 1.0))
            common_min = max(common_min, s.fft_db_floor_db)
        if common_max is None:
            common_max = float(max(np.max(db_raw), np.max(db_den)))
        if common_max <= common_min:
            common_max = common_min + 1.0
        self._fft_limits = (common_min, common_max)

        extent = [qx[0], qx[-1], qy[0], qy[-1]]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

        axes[0].imshow(
            db_raw, origin="lower", extent=extent, cmap=s.fft_cmap,
            vmin=common_min, vmax=common_max, aspect="equal")
        axes[0].set_title("RAW FFT")
        axes[0].set_xlabel(r"$q_x$ (nm$^{-1}$)"); axes[0].set_ylabel(r"$q_y$ (nm$^{-1}$)")

        im1 = axes[1].imshow(
            db_den, origin="lower", extent=extent, cmap=s.fft_cmap,
            vmin=common_min, vmax=common_max, aspect="equal")
        axes[1].set_title("DENOISED FFT")
        axes[1].set_xlabel(r"$q_x$ (nm$^{-1}$)")
        fig.colorbar(im1, ax=axes[1], label="Power (dB)")

        self._save_figure(fig, "03_fft_power_same_scale")
        self.figures["fft"] = fig
        return fig

    def plot_radial_spectrum(self):
        """
        Plot the radially averaged power spectra for RAW and DENOISED images.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure with the (optionally log-scaled) radial power spectrum.
        fig_norm : matplotlib.figure.Figure or None
            Figure with the normalized radial power spectrum, if enabled.
        """
        if self.radial_raw is None:
            self.run_fft()
        s = self.settings
        q_raw, p_raw, _, _ = self.radial_raw
        q_den, p_den, _, _ = self.radial_den

        fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
        if s.plot_log_y:
            ax.set_yscale("log")
        ax.plot(q_raw, p_raw, linewidth=2.0, label="RAW")
        ax.plot(q_den, p_den, linewidth=2.0, label="DENOISED")
        ax.set_xlim(s.spectrum_q_min, s.spectrum_q_max) if s.spectrum_q_max is not None \
            else ax.set_xlim(left=s.spectrum_q_min)
        ax.set_xlabel(r"Spatial frequency $q$ (nm$^{-1}$)")
        ax.set_ylabel(r"Radially averaged power $P(q)$")
        ax.set_title("Radially averaged power spectrum")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        self._save_figure(fig, "02_radially_averaged_power_spectrum_log")
        self.figures["radial_spectrum"] = fig

        fig_norm = None
        if s.plot_normalized_spectrum:
            fig_norm, axn = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
            p_raw_n, p_den_n = p_raw / np.max(p_raw), p_den / np.max(p_den)
            if s.plot_log_y:
                axn.set_yscale("log")
            axn.plot(q_raw, p_raw_n, linewidth=2.0, label="RAW")
            axn.plot(q_den, p_den_n, linewidth=2.0, label="DENOISED")
            axn.set_xlim(s.spectrum_q_min, s.spectrum_q_max) if s.spectrum_q_max is not None \
                else axn.set_xlim(left=s.spectrum_q_min)
            axn.set_xlabel(r"Spatial frequency $q$ (nm$^{-1}$)")
            axn.set_ylabel(r"Normalized radial power $P(q)/P_{max}$")
            axn.set_title("Radially averaged power spectrum (normalized)")
            axn.grid(True, which="both", alpha=0.25)
            axn.legend(frameon=False)
            self._save_figure(fig_norm, "04_radially_averaged_power_spectrum_normalized")
            self.figures["radial_spectrum_normalized"] = fig_norm

        return fig, fig_norm

    def plot_all(self):
        """
        Generate all comparison plots (real space, FFT, radial spectrum).
        
        Returns
        -------
        dict
            Dictionary of generated matplotlib figures, keyed by plot name.
        """
        self.plot_real_space()
        self.plot_fft()
        self.plot_radial_spectrum()
        if self.settings.show_figures:
            plt.show()
        return self.figures

    def save_all(self):
        """Save all computed results (residual, spectra, matrices, metrics, summary) to disk."""
        self._ensure_loaded()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        s = self.settings

        residual_path = self.output_dir / "residual_raw_minus_denoised.xyz"
        save_xyz(residual_path, self.x, self.y, self.residual)

        if s.save_csv:
            if self.radial_raw is None:
                self.run_fft()
            q_raw, p_raw, s_raw, n_raw = self.radial_raw
            q_den, p_den, s_den, n_den = self.radial_den
            n = max(len(q_raw), len(q_den))
            arr = np.full((n, 8), np.nan)
            arr[:len(q_raw), 0:4] = np.column_stack([q_raw, p_raw, s_raw, n_raw])
            arr[:len(q_den), 4:8] = np.column_stack([q_den, p_den, s_den, n_den])
            pd.DataFrame(arr, columns=[
                "q_raw_nm^-1", "power_raw", "power_raw_std", "n_pixels_raw",
                "q_denoised_nm^-1", "power_denoised", "power_denoised_std", "n_pixels_denoised",
            ]).to_csv(self.output_dir / "radial_power_spectrum.csv", index=False)

            if s.save_image_matrices:
                pd.DataFrame(self.raw, index=self.y, columns=self.x).to_csv(
                    self.output_dir / "raw_image_matrix.csv")
                pd.DataFrame(self.den, index=self.y, columns=self.x).to_csv(
                    self.output_dir / "denoised_image_matrix.csv")
                pd.DataFrame(self.residual, index=self.y, columns=self.x).to_csv(
                    self.output_dir / "residual_image_matrix.csv")

        if self.metrics:
            metrics_out = dict(self.metrics)
            if self._real_limits:
                metrics_out["real_space_vmin_common"], metrics_out["real_space_vmax_common"] = self._real_limits
            if self._residual_limit is not None:
                metrics_out["residual_symmetric_vlim"] = self._residual_limit
            if self._fft_limits:
                metrics_out["fft_db_vmin_common"], metrics_out["fft_db_vmax_common"] = self._fft_limits
            pd.DataFrame([metrics_out]).to_csv(self.output_dir / "comparison_metrics.csv", index=False)

        self._write_summary(residual_path)
        print(f"All results saved to: {self.output_dir.resolve()}")

    def _write_summary(self, residual_path):
        """
        Write a human-readable text summary of the comparison to disk.

        Parameters
        ----------
        residual_path : str or Path
            Path to the saved residual XYZ file, referenced in the summary.
        """
        summary_file = self.output_dir / "analysis_summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("RAW vs DENOISED AFM / XYZ comparison\n")
            f.write("=" * 60 + "\n")
            f.write(f"RAW file: {self.raw_file}\n")
            f.write(f"DENOISED file: {self.denoised_file}\n")
            f.write(f"Grid: {self.raw.shape[1]} x {self.raw.shape[0]}\n")
            f.write(f"FOV: {self.x[-1] - self.x[0]:.9f} x {self.y[-1] - self.y[0]:.9f} nm\n")
            f.write(f"dx, dy: {self.dx:.9f}, {self.dy:.9f} nm\n")
            f.write(f"FFT preprocessing: remove_mean={self.settings.remove_mean}, "
                    f"hann={self.settings.use_hann_window}\n")
            if self._real_limits:
                f.write(f"Real-space common color limits: {self._real_limits}\n")
            if self._residual_limit is not None:
                f.write(f"Residual symmetric color limit: +/-{self._residual_limit}\n")
            if self._fft_limits:
                f.write(f"FFT common dB limits: {self._fft_limits}\n")
            f.write(f"Residual file: {residual_path}\n")

    def run_all(self, verbose=True):
        """
        Run the full comparison pipeline: load, FFT, plotting, metrics, and saving.

        Parameters
        ----------
        verbose : bool, optional
            If True, print grid diagnostics and final metrics, by default True.
        
        Returns
        -------
        AFMComparison
            Self, with all attributes populated.
        """
        self.load(verbose=verbose)
        self.run_fft()
        self.plot_real_space()
        self.plot_fft()
        self.plot_radial_spectrum()
        self.compute_metrics()
        self.save_all()
        if self.settings.show_figures:
            plt.show()
        if verbose:
            print("\nMetrics:")
            for k, v in self.metrics.items():
                print(f"  {k}: {v}")
        return self

    def _ensure_loaded(self):
        """Verify that the RAW and DENOISED images have been loaded."""
        if self.raw is None:
            raise RuntimeError("Call .load() first (or .run_all()).")

    def _get_real_limits(self):
        """
        Determine common color-scale limits for the real-space RAW/DENOISED plots.

        Returns
        -------
        tuple of float
            (vmin, vmax) color limits.
        """
        s = self.settings
        if s.real_vmin is not None and s.real_vmax is not None:
            self._real_limits = (float(s.real_vmin), float(s.real_vmax))
            return self._real_limits
        combined = np.concatenate([self.raw.ravel(), self.den.ravel()])
        vmin = float(np.percentile(combined, s.real_percentile_low))
        vmax = float(np.percentile(combined, s.real_percentile_high))
        if vmax <= vmin:
            raise ValueError("Real-space color limits are invalid.")
        self._real_limits = (vmin, vmax)
        return self._real_limits

    def _get_residual_limit(self):
        """
        Determine the symmetric color-scale limit for the residual plot.

        Returns
        -------
        float
            Absolute value defining the symmetric residual color limit.
        """
        s = self.settings
        if s.residual_vlim is not None:
            self._residual_limit = float(abs(s.residual_vlim))
            return self._residual_limit
        limit = float(np.percentile(np.abs(self.residual), s.residual_percentile))
        if limit == 0:
            limit = float(np.max(np.abs(self.residual))) or 1.0
        self._residual_limit = limit
        return limit

    def _save_figure(self, fig, name):
        """
        Save a figure to the output directory if figure saving is enabled.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        name : str
            Base filename (without extension) for the saved figure.
        """
        if self.settings.save_figures:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                self.output_dir / f"{name}.{self.settings.figure_format}",
                dpi=self.settings.dpi, bbox_inches="tight"
            )
