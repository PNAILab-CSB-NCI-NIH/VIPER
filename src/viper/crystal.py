from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, optimize, signal

@dataclass
class Settings:
    # FFT preprocessing
    use_hann_window: bool = True
    remove_mean: bool = True

    # FFT peak detection
    exclude_dc_px: int = 5
    fft_min_separation_px: int = 7
    fft_threshold_sigma: float = 4.0
    max_fft_peaks: int = 40
    fft_fit_radius_px: int = 7

    # Pair +/-q peaks (physical q-distance, cycles/nm)
    fft_pair_tol_nm_inv: float = 0.020

    # Harmonic analysis
    harmonic_tolerance: float = 0.06        # 6% ratio tolerance
    max_harmonic_order: int = 6
    harmonic_angle_tol_deg: float = 12.0

    # Autocorrelation peak detection
    acf_smooth_sigma_px: float = 1.0
    acf_exclude_center_px: int = 4
    acf_min_separation_px: int = 4
    acf_threshold_sigma: float = 2.5
    max_acf_peaks: int = 80
    acf_pair_tol_nm: float = 1.5

    # Candidate pool sizes for the global lattice search
    max_acf_vectors_for_lattice: int = 30
    max_fft_vectors_for_lattice: int = 25

    # Global lattice search
    min_basis_angle_deg: float = 20.0
    max_basis_angle_deg: float = 160.0
    acf_match_tol_nm: float = 1.25
    fft_match_tol_pixels: float = 2.5
    max_lattice_index: int = 8
    min_acf_coverage: float = 0.50
    min_fft_coverage: float = 0.25

    # Plotting
    save_figures: bool = True
    show_figures: bool = True
    dpi: int = 300
    realspace_cmap: str = "afmhot"
    fft_cmap: str = "viridis"
    acf_cmap: str = "viridis"

@dataclass
class FFTPeak:
    rank: int
    px: int
    py: int
    qx: float
    qy: float
    q: float
    period_nm: float
    theta_deg: float
    model: str
    amplitude: float
    background: float
    fwhm_x: float
    fwhm_y: float
    delta_q: float
    xi_nm: float
    r2: float
    aic: float
    success: bool

@dataclass
class ACFPeak:
    rank: int
    px: int
    py: int
    dx_nm: float
    dy_nm: float
    r_nm: float
    theta_deg: float
    value: float

def load_grid(path: Path):
    try:
        data = np.loadtxt(path)
    except ValueError:
        data = np.loadtxt(path, skiprows=1)

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Expected at least 3 columns: X Y Z")

    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    xu, yu = np.unique(x), np.unique(y)
    nx, ny = len(xu), len(yu)

    if nx * ny != len(z):
        raise ValueError(f"Grid is incomplete: {nx} x {ny} != {len(z)} rows")

    order = np.lexsort((x, y))
    xs, ys, zs = x[order], y[order], z[order]
    img = zs.reshape(ny, nx)

    X, Y = np.meshgrid(xu, yu)
    if not np.allclose(xs.reshape(ny, nx), X, atol=1e-6, rtol=1e-6):
        raise ValueError("X values are not a regular row-wise grid")
    if not np.allclose(ys.reshape(ny, nx), Y, atol=1e-6, rtol=1e-6):
        raise ValueError("Y values are not a regular row-wise grid")

    dx = float(np.mean(np.diff(xu)))
    dy = float(np.mean(np.diff(yu)))
    return xu, yu, img, dx, dy, nx * dx, ny * dy

def apply_fov_override(xu, yu, dx, dy, shape, fov_nm):
    if fov_nm is None:
        return xu, yu, dx, dy, shape[1] * dx, shape[0] * dy

    if xu.max() == xu.min() or yu.max() == yu.min():
        raise ValueError("Cannot apply FOV override to zero-width coordinates.")

    sx = fov_nm / (xu.max() - xu.min())
    sy = fov_nm / (yu.max() - yu.min())
    return (xu * sx, yu * sy, dx * sx, dy * sy,
            shape[1] * dx * sx, shape[0] * dy * sy)

def robust_threshold(values, sigma_mult):
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(values)
    return med + sigma_mult * sigma

def detect_fft_peaks(power, exclude_radius_px, min_sep_px, threshold_sigma, max_peaks):
    ny, nx = power.shape
    cx, cy = nx // 2, ny // 2
    yy, xx = np.indices(power.shape)
    mask = np.hypot(xx - cx, yy - cy) >= exclude_radius_px

    threshold = robust_threshold(power[mask], threshold_sigma)
    local_max = power == ndimage.maximum_filter(power, size=5)
    candidates = np.column_stack(np.nonzero(local_max & mask & (power > threshold)))

    if candidates.size == 0:
        return [], threshold

    candidates = sorted(candidates.tolist(), key=lambda p: power[p[0], p[1]], reverse=True)

    selected = []
    for py, px in candidates:
        if all(np.hypot(px - sx, py - sy) >= min_sep_px for sy, sx in selected):
            selected.append((py, px))
            if len(selected) >= max_peaks:
                break
    return selected, threshold

def gaussian2d(coord, amp, qx0, qy0, sx, sy, offset):
    qx, qy = coord
    return offset + amp * np.exp(-0.5 * (((qx - qx0) / sx) ** 2 + ((qy - qy0) / sy) ** 2))

def lorentzian2d(coord, amp, qx0, qy0, gx, gy, offset):
    qx, qy = coord
    denom = 1.0 + ((qx - qx0) / gx) ** 2 + ((qy - qy0) / gy) ** 2
    return offset + amp / denom

def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def aic_score(y, yhat, npar=6):
    n = y.size
    rss = max(float(np.sum((y - yhat) ** 2)), np.finfo(float).tiny)
    return n * np.log(rss / n) + 2 * npar

def fit_fft_peak(power, fx, fy, py, px, rank, patch_radius_px=7):
    ny, nx = power.shape
    r = patch_radius_px
    y0, y1 = max(0, py - r), min(ny, py + r + 1)
    x0, x1 = max(0, px - r), min(nx, px + r + 1)

    qx, qy = np.meshgrid(fx[x0:x1], fy[y0:y1])
    data = power[y0:y1, x0:x1].astype(float)
    xx, yy, zz = qx.ravel(), qy.ravel(), data.ravel()

    edge = np.concatenate([data[0], data[-1], data[:, 0], data[:, -1]])
    offset0 = float(np.median(edge))
    amp0 = max(float(np.max(data) - offset0), np.finfo(float).eps)
    qx0, qy0 = float(fx[px]), float(fy[py])
    step_x, step_y = float(np.mean(np.diff(fx))), float(np.mean(np.diff(fy)))
    sx0, sy0 = max(2*step_x, 2*abs(step_x)), max(2*step_y, 2*abs(step_y))

    candidates = []
    bounds = ([0, fx[x0], fy[y0], step_x/3, step_y/3, -np.inf],
              [np.inf, fx[x1 - 1], fy[y1 - 1], abs(fx[x1 - 1] - fx[x0]),
               abs(fy[y1 - 1] - fy[y0]), np.inf])

    try:
        popt, _ = optimize.curve_fit(
            gaussian2d, (xx, yy), zz,
            p0=[amp0, qx0, qy0, sx0, sy0, offset0],
            bounds=bounds, maxfev=30000
        )
        pred = gaussian2d((xx, yy), *popt)
        amp, qxc, qyc, sx, sy, offset = popt
        fwhmx, fwhmy = 2*np.sqrt(2*np.log(2))*abs(sx), 2*np.sqrt(2*np.log(2))*abs(sy)
        dq = np.sqrt(fwhmx*fwhmy)
        q = np.hypot(qxc, qyc)
        candidates.append({
            "model": "gaussian", "amp": amp, "qx": qxc, "qy": qyc,
            "background": offset, "fwhmx": fwhmx, "fwhmy": fwhmy, "dq": dq,
            "xi": 1 / dq if dq > 0 else np.nan, "q": q,
            "period": 1 / q if q > 0 else np.nan,
            "theta": np.degrees(np.arctan2(qyc, qxc)),
            "r2": r2_score(zz, pred), "aic": aic_score(zz, pred),
        })
    except Exception:
        pass

    try:
        popt, _ = optimize.curve_fit(
            lorentzian2d, (xx, yy), zz,
            p0=[amp0, qx0, qy0, max(sx0, step_x), max(sy0, step_y), offset0],
            bounds=bounds, maxfev=30000
        )
        pred = lorentzian2d((xx, yy), *popt)
        amp, qxc, qyc, gx, gy, offset = popt
        fwhmx, fwhmy = 2*abs(gx), 2*abs(gy)
        dq = np.sqrt(fwhmx*fwhmy)
        q = np.hypot(qxc, qyc)
        candidates.append({
            "model": "lorentzian", "amp": amp, "qx": qxc, "qy": qyc,
            "background": offset, "fwhmx": fwhmx, "fwhmy": fwhmy, "dq": dq,
            "xi": 1 / dq if dq > 0 else np.nan, "q": q,
            "period": 1 / q if q > 0 else np.nan,
            "theta": np.degrees(np.arctan2(qyc, qxc)),
            "r2": r2_score(zz, pred), "aic": aic_score(zz, pred),
        })
    except Exception:
        pass

    if not candidates:
        return FFTPeak(
            rank, px, py, np.nan, np.nan, np.nan, np.nan, np.nan,
            "failed", np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, False)

    best = min(candidates, key=lambda d: d["aic"])
    return FFTPeak(
        rank=rank, px=px, py=py, qx=float(best["qx"]), qy=float(best["qy"]),
        q=float(best["q"]), period_nm=float(best["period"]), theta_deg=float(best["theta"]),
        model=best["model"], amplitude=float(best["amp"]), background=float(best["background"]),
        fwhm_x=float(best["fwhmx"]), fwhm_y=float(best["fwhmy"]), delta_q=float(best["dq"]),
        xi_nm=float(best["xi"]), r2=float(best["r2"]), aic=float(best["aic"]), success=True,
    )

def angle_difference_180(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)

def pair_fft_peaks(peaks, tolerance_q):
    valid = [p for p in peaks if p.success and np.isfinite(p.q)]
    used = set()
    pairs = []
    pair_id = 1

    for i, p in enumerate(valid):
        if i in used:
            continue
        target = np.array([-p.qx, -p.qy])
        best_j, best_dist = None, np.inf
        for j, m in enumerate(valid):
            if j == i or j in used:
                continue
            dist = np.hypot(m.qx - target[0], m.qy - target[1])
            if dist < best_dist:
                best_dist, best_j = dist, j

        if best_j is None or best_dist > tolerance_q:
            continue

        m = valid[best_j]
        used.add(i); used.add(best_j)
        qx, qy = 0.5*(p.qx - m.qx), 0.5*(p.qy - m.qy)
        q = np.hypot(qx, qy)
        pairs.append({
            "pair_id": pair_id, "rank_1": p.rank, "rank_2": m.rank,
            "qx_nm^-1": qx, "qy_nm^-1": qy, "q_nm^-1": q,
            "period_nm": 1 / q if q > 0 else np.nan,
            "theta_deg": np.degrees(np.arctan2(qy, qx)),
            "mean_amplitude": 0.5*(p.amplitude + m.amplitude),
            "mean_delta_q_nm^-1": 0.5*(p.delta_q + m.delta_q),
            "mean_xi_nm": 0.5*(p.xi_nm + m.xi_nm),
            "pair_error_nm^-1": best_dist,
        })
        pair_id += 1

    return sorted(pairs, key=lambda d: d["q_nm^-1"])

def harmonic_matches(pairs, max_order, ratio_tolerance, angle_tol_deg):
    rows = []
    pairs = sorted(pairs, key=lambda d: d["q_nm^-1"])
    for i, base in enumerate(pairs):
        for candidate in pairs[i + 1:]:
            ratio = candidate["q_nm^-1"]/base["q_nm^-1"]
            n = int(round(ratio))
            if n < 2 or n > max_order:
                continue
            rel_dev = abs(ratio - n)/n
            ang = angle_difference_180(base["theta_deg"], candidate["theta_deg"])
            if rel_dev <= ratio_tolerance:
                rows.append({
                    "base_pair_id": base["pair_id"], "candidate_pair_id": candidate["pair_id"],
                    "harmonic_order": n, "q_ratio": ratio, "relative_deviation": rel_dev,
                    "orientation_difference_deg": ang,
                    "orientation_consistent": ang <= angle_tol_deg,
                })
    return rows

def compute_autocorrelation(img0, use_hann):
    work = img0.copy()
    if use_hann:
        wy = signal.windows.hann(work.shape[0], sym=False)
        wx = signal.windows.hann(work.shape[1], sym=False)
        work = work*np.outer(wy, wx)

    F = np.fft.fft2(work)
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(F)**2).real)
    center = ac[work.shape[0] // 2, work.shape[1] // 2]
    if center != 0:
        ac /= center
    return ac

def refine_acf_peak(ac_smooth, py, px):
    ny, nx = ac_smooth.shape
    if px <= 0 or px >= nx - 1 or py <= 0 or py >= ny - 1:
        return float(px), float(py)

    patch = ac_smooth[py - 1:py + 2, px - 1:px + 2]
    if not np.all(np.isfinite(patch)):
        return float(px), float(py)

    yy, xx = np.mgrid[-1:2, -1:2]
    A = np.column_stack(
        [xx.ravel()**2, yy.ravel()**2, (xx*yy).ravel(),
        xx.ravel(), yy.ravel(), np.ones(9)]
    )
    coeff, *_ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
    aa, bb, cc, dd, ee, _ = coeff
    H = np.array([[2*aa, cc], [cc, 2*bb]], dtype=float)
    g = np.array([dd, ee], dtype=float)

    try:
        shift = -np.linalg.solve(H, g)
        if np.all(np.isfinite(shift)) and np.max(np.abs(shift)) <= 1.0:
            return float(px + shift[0]), float(py + shift[1])
    except np.linalg.LinAlgError:
        pass
    return float(px), float(py)

def detect_acf_peaks(
        ac, dx, dy, smooth_sigma_px, exclude_center_px,
        min_separation_px, threshold_sigma, max_peaks):
    smooth = ndimage.gaussian_filter(ac, smooth_sigma_px)
    ny, nx = ac.shape
    cx, cy = nx // 2, ny // 2
    yy, xx = np.indices(ac.shape)
    mask = np.hypot(xx - cx, yy - cy) >= exclude_center_px

    vals = smooth[mask]
    threshold = np.median(vals) + threshold_sigma*np.std(vals)

    local_max = smooth == ndimage.maximum_filter(smooth, size=5)
    candidates = np.column_stack(np.nonzero(local_max & mask & (smooth > threshold)))

    if candidates.size == 0:
        return [], threshold

    candidates = sorted(candidates.tolist(), key=lambda p: smooth[p[0], p[1]], reverse=True)

    selected = []
    for py, px in candidates:
        if all(np.hypot(px - sx, py - sy) >= min_separation_px for sy, sx in selected):
            selected.append((py, px))
            if len(selected) >= max_peaks:
                break

    peaks = []
    for rank, (py, px) in enumerate(selected, 1):
        pxf, pyf = refine_acf_peak(smooth, py, px)
        lagx, lagy = (pxf - cx)*dx, (pyf - cy)*dy
        r = np.hypot(lagx, lagy)
        peaks.append(ACFPeak(
            rank=rank, px=px, py=py, dx_nm=float(lagx), dy_nm=float(lagy),
            r_nm=float(r), theta_deg=float(np.degrees(np.arctan2(lagy, lagx))),
            value=float(ac[py, px])))
    return peaks, threshold


def pair_acf_peaks(peaks, tolerance_nm):
    used = set()
    pairs = []
    pair_id = 1

    for i, p in enumerate(peaks):
        if i in used or p.r_nm <= 0:
            continue
        target = np.array([-p.dx_nm, -p.dy_nm])
        best_j, best_dist = None, np.inf
        for j, m in enumerate(peaks):
            if j == i or j in used or m.r_nm <= 0:
                continue
            dist = np.hypot(m.dx_nm - target[0], m.dy_nm - target[1])
            if dist < best_dist:
                best_dist, best_j = dist, j

        if best_j is None or best_dist > tolerance_nm:
            continue

        m = peaks[best_j]
        used.add(i); used.add(best_j)
        vx, vy = 0.5*(p.dx_nm - m.dx_nm), 0.5*(p.dy_nm - m.dy_nm)
        pairs.append({
            "pair_id": pair_id, "rank_1": p.rank, "rank_2": m.rank,
            "dx_nm": float(vx), "dy_nm": float(vy), "r_nm": float(np.hypot(vx, vy)),
            "theta_deg": float(np.degrees(np.arctan2(vy, vx))),
            "acf_value": 0.5 * (p.value + m.value), "pair_error_nm": float(best_dist),
        })
        pair_id += 1
    return pairs

def vector_angle(v1, v2):
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    den = np.linalg.norm(v1)*np.linalg.norm(v2)
    if den <= 0:
        return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2)/den, -1, 1)))

def canonical_vector(v):
    v = np.asarray(v, dtype=float)
    if v[0] < 0 or (abs(v[0]) < 1e-12 and v[1] < 0):
        v = -v
    return v

def unique_vectors(vectors, tolerance_nm=0.15):
    unique = []
    for v in vectors:
        v = np.asarray(v, dtype=float)
        if np.linalg.norm(v) <= tolerance_nm:
            continue
        if any(np.linalg.norm(v - u) <= tolerance_nm or
               np.linalg.norm(v + u) <= tolerance_nm for u in unique):
            continue
        unique.append(v)
    return unique

def build_candidate_basis_vectors(acf_pairs, max_vectors, n_candidates=60):
    selected = sorted(acf_pairs, key=lambda d: d["acf_value"], reverse=True)[:max_vectors]
    base = [canonical_vector([p["dx_nm"], p["dy_nm"]]) for p in selected]

    candidates = list(base)
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            for s1, s2 in ((1, 1), (1, -1)):
                v = s1 * base[i] + s2 * base[j]
                if np.linalg.norm(v) > 0:
                    candidates.append(canonical_vector(v))

    candidates = unique_vectors(candidates, tolerance_nm=0.10)
    candidates.sort(key=np.linalg.norm)
    return candidates[:n_candidates]

def lattice_predictions(b1, b2, max_index, extent_x, extent_y):
    A = np.column_stack([b1, b2])
    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        return None

    points = []
    scale = max(np.linalg.norm(b1), np.linalg.norm(b2))
    max_extent = max(extent_x, extent_y)
    max_n = min(max(max_index, int(max_extent/max(scale, 1e-9)) + 1), 12)

    for m in range(-max_n, max_n + 1):
        for n in range(-max_n, max_n + 1):
            if m == 0 and n == 0:
                continue
            r = A @ np.array([m, n], dtype=float)
            if abs(r[0]) <= extent_x and abs(r[1]) <= extent_y:
                points.append((m, n, r[0], r[1]))
    return np.asarray(points, dtype=float)


def nearest_lattice_residual(observed, b1, b2, tol_nm, max_lattice_index):
    pred = lattice_predictions(b1, b2, max_lattice_index, 55, 55)
    if pred is None or len(pred) == 0:
        return 0.0, np.inf, []

    pred_xy = pred[:, 2:4]
    residuals, assignments, weights = [], [], []

    for obs in observed:
        r = np.array([obs["dx_nm"], obs["dy_nm"]], dtype=float)
        dist = np.min(np.linalg.norm(pred_xy - r, axis=1))
        residuals.append(dist)
        weights.append(max(float(obs["acf_value"]), 0.0))
        assignments.append(dist <= tol_nm)

    weights, residuals = np.asarray(weights), np.asarray(residuals)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)

    coverage = float(np.sum(weights[assignments]) / np.sum(weights))
    rms = float(np.sqrt(np.sum(weights * residuals ** 2) / np.sum(weights)))
    return coverage, rms, list(zip(residuals, assignments))

def reciprocal_basis_from_direct(b1, b2):
    A = np.column_stack([b1, b2])
    if abs(np.linalg.det(A)) < 1e-10:
        return None
    return np.linalg.inv(A).T

def fft_lattice_coverage(observed_q, G, fft_match_tol_pixels):
    if G is None or len(observed_q) == 0:
        return 0.0, np.inf, []

    Ginv = np.linalg.inv(G)
    assignments, residuals, weights = [], [], []

    for obs in observed_q:
        qvec = np.array([obs["qx_nm^-1"], obs["qy_nm^-1"]], dtype=float)
        hk_round = np.rint(Ginv@qvec)
        q_pred = G @ hk_round
        dist = float(np.linalg.norm(qvec - q_pred))
        residuals.append(dist)
        weights.append(max(float(obs.get("mean_amplitude", 1.0)), 0.0))
        assignments.append(dist <= fft_match_tol_pixels * obs["dq_grid_nm^-1"])

    residuals, weights = np.asarray(residuals), np.asarray(weights)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)

    coverage = float(np.sum(weights[assignments])/np.sum(weights))
    rms = float(np.sqrt(np.sum(weights*residuals**2)/np.sum(weights)))
    return coverage, rms, list(zip(residuals, assignments))

def evaluate_basis(b1, b2, acf_pairs, fft_pairs, dq_grid, s: Settings):
    b1, b2 = np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)
    l1, l2 = np.linalg.norm(b1), np.linalg.norm(b2)
    if l1 <= 0 or l2 <= 0:
        return None

    beta = vector_angle(b1, b2)
    if not np.isfinite(beta) or beta < s.min_basis_angle_deg or beta > s.max_basis_angle_deg:
        return None

    area = abs(np.cross(b1, b2))
    if area <= 1e-8:
        return None

    acf_cov, acf_rms, acf_assign = nearest_lattice_residual(
        acf_pairs, b1, b2, s.acf_match_tol_nm, s.max_lattice_index
    )

    G = reciprocal_basis_from_direct(b1, b2)
    fft_cov, fft_rms, fft_assign = fft_lattice_coverage(
        fft_pairs, G, s.fft_match_tol_pixels
    ) if fft_pairs else (np.nan, np.nan, [])

    score = 5.0*acf_cov - 0.15*acf_rms
    if np.isfinite(fft_cov):
        score += 2.5*fft_cov - 0.10*(fft_rms/max(dq_grid, 1e-9))
    score -= 0.001*area

    return {
        "b1": b1, "b2": b2, "b1_length_nm": float(l1), "b2_length_nm": float(l2),
        "beta_deg": float(beta), "area_nm2": float(area),
        "acf_coverage": float(acf_cov), "acf_rms_nm": float(acf_rms), "acf_assignments": acf_assign,
        "G": G, "fft_coverage": float(fft_cov) if np.isfinite(fft_cov) else np.nan,
        "fft_rms_nm^-1": float(fft_rms) if np.isfinite(fft_rms) else np.nan,
        "fft_assignments": fft_assign, "score": float(score),
    }

def find_global_lattice(acf_pairs, fft_pairs, dq_grid, s: Settings):
    if len(acf_pairs) < 2:
        return None, "Not enough paired autocorrelation vectors."

    candidates = build_candidate_basis_vectors(acf_pairs, s.max_acf_vectors_for_lattice)
    best = None

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            result = evaluate_basis(candidates[i], candidates[j], acf_pairs, fft_pairs, dq_grid, s)
            if result is not None and (best is None or result["score"] > best["score"]):
                best = result

    if best is None:
        return None, "No non-collinear data-supported lattice basis found."

    if best["acf_coverage"] < s.min_acf_coverage:
        return best, f"Best basis found but ACF coverage is low ({best['acf_coverage']:.2f}). Treat result cautiously."

    if np.isfinite(best["fft_coverage"]) and len(fft_pairs) >= 3:
        if best["fft_coverage"] < s.min_fft_coverage:
            return best, f"Best basis found; ACF support is adequate, but FFT coverage is low ({best['fft_coverage']:.2f})."

    return best, "ok"

def reduce_primitive_basis_2d(b1, b2, max_coeff=8, area_tol=0.03):
    b1, b2 = np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)
    primitive_area = abs(np.cross(b1, b2))
    if primitive_area <= 1e-10:
        return b1, b2

    vectors = []
    for m in range(-max_coeff, max_coeff + 1):
        for n in range(-max_coeff, max_coeff + 1):
            if m == 0 and n == 0:
                continue
            v = m*b1 + n*b2
            nv = np.linalg.norm(v)
            if nv > 1e-9:
                vectors.append((nv, v))

    vectors.sort(key=lambda t: t[0])
    v1 = vectors[0][1]

    best_v2, best_len = None, np.inf
    for _, v in vectors:
        area = abs(np.cross(v1, v))
        if abs(area - primitive_area) / primitive_area <= area_tol and np.linalg.norm(v) < best_len:
            best_len, best_v2 = np.linalg.norm(v), v.copy()

    if best_v2 is None:
        for _, v in vectors:
            if abs(np.cross(v1, v)) > 1e-8:
                best_v2 = v.copy()
                break

    if best_v2 is None:
        return b1, b2

    return canonical_vector(v1), canonical_vector(best_v2)

def lattice_rows(lattice):
    b1, b2 = lattice["b1"], lattice["b2"]
    short, long = sorted([b1, b2], key=np.linalg.norm)
    return {
        "v1_dx_nm": b1[0], "v1_dy_nm": b1[1], "v1_length_nm": np.linalg.norm(b1),
        "v1_theta_deg": np.degrees(np.arctan2(b1[1], b1[0])),
        "v2_dx_nm": b2[0], "v2_dy_nm": b2[1], "v2_length_nm": np.linalg.norm(b2),
        "v2_theta_deg": np.degrees(np.arctan2(b2[1], b2[0])),
        "shorter_vector_length_nm": np.linalg.norm(short),
        "shorter_vector_theta_deg": np.degrees(np.arctan2(short[1], short[0])),
        "longer_vector_length_nm": np.linalg.norm(long),
        "longer_vector_theta_deg": np.degrees(np.arctan2(long[1], long[0])),
        "beta_deg": lattice["beta_deg"], "area_nm2": lattice["area_nm2"],
        "acf_coverage": lattice["acf_coverage"], "acf_rms_nm": lattice["acf_rms_nm"],
        "fft_coverage": lattice["fft_coverage"], "fft_rms_nm^-1": lattice["fft_rms_nm^-1"],
        "score": lattice["score"],
    }

def crystal_reference_rows(crystal_phases):
    rows = []
    for phase, vals in crystal_phases.items():
        a_nm, c_nm = vals["a_A"]/10.0, vals["c_A"]/10.0
        rows.append({
            "phase": phase, "a_nm": a_nm, "c_nm": c_nm, "c_over_2_nm": c_nm/2,
            "beta_deg": vals["beta_deg"], "q_a_cycles_nm": 1/a_nm, "q_c_cycles_nm": 1/c_nm,
            "Q_a_rad_nm": 2*np.pi/a_nm, "Q_c_rad_nm": 2*np.pi/c_nm,
        })
    return rows

def compare_lattice_to_crystal(lattice, refs):
    if lattice is None:
        return []
    v1, v2 = lattice["b1"], lattice["b2"]
    rows = []
    for ref in refs:
        assignments = [
            ("v1->a, v2->c", np.linalg.norm(v1), np.linalg.norm(v2)),
            ("v2->a, v1->c", np.linalg.norm(v2), np.linalg.norm(v1)),
        ]
        for assignment, aval, cval in assignments:
            rows.append({
                "phase": ref["phase"], "assignment": assignment,
                "afm_a_candidate_nm": aval, "crystal_a_nm": ref["a_nm"],
                "a_difference_percent": 100 * abs(aval - ref["a_nm"]) / ref["a_nm"],
                "afm_c_candidate_nm": cval, "crystal_c_nm": ref["c_nm"],
                "c_difference_percent": 100 * abs(cval - ref["c_nm"]) / ref["c_nm"],
                "afm_beta_deg": lattice["beta_deg"], "crystal_beta_deg": ref["beta_deg"],
                "beta_abs_difference_deg": abs(lattice["beta_deg"] - ref["beta_deg"]),
            })
    return rows

def radial_spectrum(power, fx, fy, nbins=180):
    FX, FY = np.meshgrid(fx, fy)
    qv = np.hypot(FX, FY).ravel()
    pv = power.ravel()
    bins = np.linspace(0, qv.max(), nbins + 1)
    idx = np.digitize(qv, bins) - 1

    rq, rp = [], []
    for b in range(nbins):
        sel = idx == b
        if np.any(sel):
            rq.append(0.5*(bins[b] + bins[b + 1]))
            rp.append(np.mean(pv[sel]))
    return np.asarray(rq), np.asarray(rp)

def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("\n")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def write_summary(path, info):
    with path.open("w") as f:
        for section, lines in info:
            f.write(f"\n{section}\n")
            f.write("-" * len(section) + "\n")
            for line in lines:
                f.write(str(line) + "\n")

class AFMLatticeAnalysis:
    def __init__(self, file, output_dir="autocorr", fov_nm= None, data_label="AFM", **settings):
        self.file = Path(file)
        self.output_dir = Path(output_dir)
        self.fov_nm = fov_nm
        self.data_label = data_label
        self.settings = Settings(**settings)

        # populated by .load()
        self.xu = self.yu = self.img = None
        self.dx = self.dy = self.Lx = self.Ly = None

        # populated by .run_fft()
        self.power = self.power_log = None
        self.fx = self.fy = None
        self.dq_grid = None
        self.fft_fits: List[FFTPeak] = []
        self.paired_fft: List[dict]  = []
        self.harmonics:  List[dict]  = []
        self.fft_threshold = None

        # populated by .run_autocorrelation()
        self.ac = None
        self.lag_x = self.lag_y = None
        self.raw_acf_peaks: List[ACFPeak] = []
        self.acf_pairs: List[dict] = []
        self.acf_threshold = None

        # populated by .find_lattice()
        self.lattice = None
        self.lattice_status = None

        # populated by .compare_to_crystal_phases()
        self.crystal_phases: Dict[str, Dict[str, float]] = {}
        self.crystal_refs: List[dict] = []
        self.crystal_comparison: List[dict] = []

        self.figures: Dict[str, plt.Figure] = {}

    def load(self, verbose=True):
        if not self.file.exists():
            raise FileNotFoundError(f"Input file not found: {self.file}")

        xu, yu, img, dx, dy, Lx, Ly = load_grid(self.file)
        xu, yu, dx, dy, Lx, Ly = apply_fov_override(xu, yu, dx, dy, img.shape, self.fov_nm)

        if not np.all(np.isfinite(img)):
            raise ValueError("Z contains NaN/Inf values.")

        self.xu, self.yu, self.img = xu, yu, img
        self.dx, self.dy, self.Lx, self.Ly = dx, dy, Lx, Ly

        if verbose:
            ny, nx = img.shape
            print(f"Loaded {self.file.name}: {ny} x {nx} grid, "
                  f"dx={dx:.6f} nm, dy={dy:.6f} nm, FOV={Lx:.3f} x {Ly:.3f} nm")
        return self

    def run_fft(self):
        self._ensure_loaded()
        s = self.settings
        img = self.img
        ny, nx = img.shape
        img0 = img - np.mean(img) if s.remove_mean else img.copy()

        if s.use_hann_window:
            wx, wy = signal.windows.hann(nx, sym=False), signal.windows.hann(ny, sym=False)
            fft_input = img0*np.outer(wy, wx)
        else:
            fft_input = img0

        F = np.fft.fftshift(np.fft.fft2(fft_input))
        self.power = np.abs(F)**2
        self.power_log = np.log10(self.power + np.finfo(float).eps)

        self.fx = np.fft.fftshift(np.fft.fftfreq(nx, d=self.dx))
        self.fy = np.fft.fftshift(np.fft.fftfreq(ny, d=self.dy))
        dqx, dqy = float(np.mean(np.diff(self.fx))), float(np.mean(np.diff(self.fy)))
        self.dq_grid = max(abs(dqx), abs(dqy))

        raw_peaks, self.fft_threshold = detect_fft_peaks(
            self.power, s.exclude_dc_px, s.fft_min_separation_px,
            s.fft_threshold_sigma, s.max_fft_peaks
        )
        self.fft_fits = [
            fit_fft_peak(self.power, self.fx, self.fy, py, px, rank, s.fft_fit_radius_px)
            for rank, (py, px) in enumerate(raw_peaks, 1)
        ]
        self.paired_fft = pair_fft_peaks(self.fft_fits, s.fft_pair_tol_nm_inv)
        for p in self.paired_fft:
            p["dq_grid_nm^-1"] = self.dq_grid

        self.harmonics = harmonic_matches(
            self.paired_fft, s.max_harmonic_order, s.harmonic_tolerance, s.harmonic_angle_tol_deg
        )
        return self

    def run_autocorrelation(self):
        self._ensure_loaded()
        s = self.settings
        img = self.img
        ny, nx = img.shape
        img0 = img - np.mean(img) if s.remove_mean else img.copy()

        self.ac = compute_autocorrelation(img0, s.use_hann_window)
        self.lag_x = (np.arange(nx) - nx // 2) * self.dx
        self.lag_y = (np.arange(ny) - ny // 2) * self.dy

        self.raw_acf_peaks, self.acf_threshold = detect_acf_peaks(
            self.ac, self.dx, self.dy, s.acf_smooth_sigma_px, s.acf_exclude_center_px,
            s.acf_min_separation_px, s.acf_threshold_sigma, s.max_acf_peaks
        )
        self.acf_pairs = pair_acf_peaks(self.raw_acf_peaks, s.acf_pair_tol_nm)
        return self

    def find_lattice(self):
        if not self.acf_pairs and self.ac is None:
            self.run_autocorrelation()
        if not self.paired_fft and self.power is None:
            self.run_fft()

        s = self.settings
        lattice, status = find_global_lattice(
            self.acf_pairs[:s.max_acf_vectors_for_lattice],
            self.paired_fft[:s.max_fft_vectors_for_lattice],
            self.dq_grid, s,
        )
        if lattice is not None:
            rb1, rb2 = reduce_primitive_basis_2d(lattice["b1"], lattice["b2"])
            lattice["b1"], lattice["b2"] = rb1, rb2
            lattice["beta_deg"] = vector_angle(rb1, rb2)
            lattice["area_nm2"] = abs(np.cross(rb1, rb2))
            lattice["G"] = reciprocal_basis_from_direct(rb1, rb2)

        self.lattice, self.lattice_status = lattice, status
        return self

    def compare_to_crystal_phases(self, crystal_phases):
        if self.lattice is None:
            self.find_lattice()

        self.crystal_phases = crystal_phases
        self.crystal_refs = crystal_reference_rows(crystal_phases)
        self.crystal_comparison = compare_lattice_to_crystal(self.lattice, self.crystal_refs)
        return self.crystal_comparison

    def _save_fig(self, fig, name):
        if self.settings.save_figures:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.output_dir / name, dpi=self.settings.dpi, bbox_inches="tight")
        if self.settings.show_figures:
            plt.show(block=False)
        else: plt.close(fig)

    def plot_real_space(self):
        self._ensure_loaded()
        s = self.settings
        fig, ax = plt.subplots(figsize=(6, 5))
        vmin, vmax = np.percentile(self.img, [1, 99])
        im = ax.imshow(
            self.img, origin="lower",
            extent=[self.xu.min(), self.xu.max(), self.yu.min(), self.yu.max()],
            aspect="equal", cmap=s.realspace_cmap, vmin=vmin, vmax=vmax
        )
        ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)")
        ax.set_title(f"{self.data_label}: real space")
        fig.colorbar(im, ax=ax, label="Z")
        self._save_fig(fig, "01_real_space.png")
        self.figures["real_space"] = fig
        return fig

    def plot_fft(self):
        if self.power is None:
            self.run_fft()
        s = self.settings
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            self.power_log, origin="lower",
            extent=[self.fx.min(), self.fx.max(), self.fy.min(), self.fy.max()],
            aspect="equal", cmap=s.fft_cmap
        )
        for p in self.fft_fits:
            if p.success:
                ax.scatter(p.qx, p.qy, s=30, facecolors="none", edgecolors="white")
        for p in self.paired_fft:
            ax.scatter(
                p["qx_nm^-1"], p["qy_nm^-1"], s=75,
                facecolors="none", edgecolors="red", linewidths=1.2
            )
            ax.annotate(
                f"P{p['pair_id']}", (p["qx_nm^-1"], p["qy_nm^-1"]),
                xytext=(4, 4), textcoords="offset points", color="white", fontsize=8
            )
        ax.axhline(0, lw=0.4, alpha=0.3); ax.axvline(0, lw=0.4, alpha=0.3)
        ax.set_xlabel("qx (cycles/nm)"); ax.set_ylabel("qy (cycles/nm)")
        ax.set_title("FFT power spectrum, log10")
        fig.colorbar(im, ax=ax, label="log10(power)")
        self._save_fig(fig, "02_fft_power_paired_peaks.png")
        self.figures["fft"] = fig
        return fig

    def plot_radial_spectrum(self):
        if self.power is None:
            self.run_fft()
        rq, rp = radial_spectrum(self.power, self.fx, self.fy)
        self._radial_q, self._radial_p = rq, rp  # cache for save_all

        fig, ax = plt.subplots(figsize=(7, 4.6))
        mask = (rq > 0) & (rp > 0)
        ax.plot(rq[mask], rp[mask], lw=1.8, label=self.data_label)
        ax.set_yscale("log")
        if np.any(mask):
            ymax = np.nanmax(rp[mask])
            for p in self.paired_fft:
                q = p["q_nm^-1"]
                if q <= rq.max():
                    ax.axvline(q, ls="--", lw=0.8, alpha=0.5)
                    ax.text(
                        q, ymax, f"{p['period_nm']:.2f} nm", rotation=90,
                        va="top", ha="right", fontsize=8
                    )
        ax.set_xlabel("q (cycles/nm)");
        ax.set_ylabel("Radially averaged power")
        ax.set_title("Radially averaged FFT power spectrum")
        ax.grid(alpha=0.2)
        self._save_fig(fig, "03_radial_spectrum_log.png")
        self.figures["radial_spectrum"] = fig
        return fig

    def plot_autocorrelation(self):
        if self.ac is None:
            self.run_autocorrelation()
        if self.lattice is None:
            self.find_lattice()
        s = self.settings
        lattice = self.lattice

        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        im = ax.imshow(
            self.ac, origin="lower",
            extent=[self.lag_x.min(), self.lag_x.max(), self.lag_y.min(), self.lag_y.max()],
            aspect="equal", cmap=s.acf_cmap,
            vmin=np.percentile(self.ac, 1), vmax=1.0
        )

        for p in self.acf_pairs:
            ax.scatter(
                p["dx_nm"], p["dy_nm"], s=45, facecolors="none",
                edgecolors="white", linewidths=0.9
            )
            ax.annotate(
                f"{p['pair_id']}", (p["dx_nm"], p["dy_nm"]),
                xytext=(4, 4), textcoords="offset points", color="white", fontsize=8
            )

        if lattice is not None:
            extent_x = max(abs(self.lag_x.min()), abs(self.lag_x.max()))
            extent_y = max(abs(self.lag_y.min()), abs(self.lag_y.max()))
            pred = lattice_predictions(
                lattice["b1"], lattice["b2"], s.max_lattice_index, extent_x, extent_y)
            if pred is not None and len(pred):
                ax.scatter(
                    pred[:, 2], pred[:, 3], marker="+", s=40,
                    linewidths=0.9, label="predicted lattice"
                )

            b1, b2 = lattice["b1"], lattice["b2"]
            ax.plot([0, b1[0]], [0, b1[1]], lw=2.2, label=f"v1 = {np.linalg.norm(b1):.2f} nm")
            ax.plot([0, b2[0]], [0, b2[1]], lw=2.2, label=f"v2 = {np.linalg.norm(b2):.2f} nm")
            ax.scatter([b1[0], b2[0]], [b1[1], b2[1]], s=40)
            title = (f"ACF: data-driven lattice\n|v1|={np.linalg.norm(b1):.2f} nm, "
                    f"|v2|={np.linalg.norm(b2):.2f} nm, beta={lattice['beta_deg']:.2f}°")
        else: title = "2D autocorrelation"

        ax.axhline(0, lw=0.4, alpha=0.35); ax.axvline(0, lw=0.4, alpha=0.35)
        ax.set_xlabel("Delta X (nm)"); ax.set_ylabel("Delta Y (nm)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Normalized autocorrelation")
        ax.legend(loc="best")
        self._save_fig(fig, "04_autocorrelation_data_driven_lattice.png")
        self.figures["autocorrelation"] = fig
        return fig

    def plot_direct_vs_crystal(self):
        if self.lattice is None or not self.crystal_refs:
            print("No lattice or no crystal references set.")
            return None

        lattice, refs = self.lattice, self.crystal_refs
        v1_len, v2_len = np.linalg.norm(lattice["b1"]), np.linalg.norm(lattice["b2"])

        fig, ax = plt.subplots(figsize=(8.5, 5.))
        ax.scatter([0, 1], [v1_len, v2_len], s=80, marker="o", label="AFM ACF")

        pos, tick_pos, tick_labels = 3, [0, 1], ["AFM v1", "AFM v2"]
        for ref in refs:
            ax.scatter(pos, ref["a_nm"], s=65, marker="s", label="crystal a" if pos == 3 else None)
            ax.scatter(pos, ref["c_nm"], s=65, marker="^", label="crystal c" if pos == 3 else None)
            ax.plot([pos - 0.18, pos + 0.18], [ref["a_nm"]]*2, lw=1.5)
            ax.plot([pos - 0.18, pos + 0.18], [ref["c_nm"]]*2, lw=1.5)
            tick_pos.append(pos); tick_labels.append(ref["phase"])
            pos += 1

        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labels, rotation=25, ha="right")
        ax.set_ylabel("Length (nm)")
        ax.set_title("AFM ACF basis lengths vs crystallographic a and c")
        ax.grid(alpha=0.2, axis="y")
        ax.legend(loc="best")
        self._save_fig(fig, "05_direct_lattice_vs_crystal.png")
        self.figures["direct_vs_crystal"] = fig
        return fig

    def plot_fft_reciprocal_lattice(self):
        if self.lattice is None or self.lattice.get("G") is None:
            print("No lattice found; call find_lattice() first.")
            return None
        if self.power is None:
            self.run_fft()

        s = self.settings
        G = self.lattice["G"]
        fig, ax = plt.subplots(figsize=(7., 6.))

        qx_obs = [p["qx_nm^-1"] for p in self.paired_fft]
        qy_obs = [p["qy_nm^-1"] for p in self.paired_fft]
        ax.scatter(
            qx_obs, qy_obs, s=70, facecolors="none", edgecolors="red",
            linewidths=1.2, label="paired FFT peaks"
        )

        qmax = max(abs(self.fx.min()), abs(self.fx.max()), abs(self.fy.min()), abs(self.fy.max()))
        pts = []
        for h in range(-s.max_lattice_index, s.max_lattice_index + 1):
            for k in range(-s.max_lattice_index, s.max_lattice_index + 1):
                q = G@np.array([h, k], dtype=float)
                if 1e-9 < np.linalg.norm(q) <= qmax:
                    pts.append(q)
        pts = np.asarray(pts)
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], marker="+", s=35, label="predicted reciprocal lattice")

        ax.set_xlim(self.fx.min(), self.fx.max())
        ax.set_ylim(self.fy.min(), self.fy.max())
        ax.set_aspect("equal")
        ax.set_xlabel("qx (cycles/nm)"); ax.set_ylabel("qy (cycles/nm)")
        ax.set_title("FFT peaks vs reciprocal lattice predicted from ACF")
        ax.grid(alpha=0.15)
        ax.legend(loc="best")
        self._save_fig(fig, "06_fft_vs_reciprocal_lattice.png")
        self.figures["fft_reciprocal_lattice"] = fig
        return fig

    def plot_all(self):
        self.plot_real_space()
        self.plot_fft()
        self.plot_radial_spectrum()
        self.plot_autocorrelation()
        if self.crystal_refs:
            self.plot_direct_vs_crystal()
        self.plot_fft_reciprocal_lattice()
        return self.figures

    def save_all(self):
        self._ensure_loaded()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        write_csv(self.output_dir / "afm_fft_peaks_fitted.csv", [
            {"rank": p.rank, "px": p.px, "py": p.py, "qx_nm^-1": p.qx, "qy_nm^-1": p.qy,
             "q_nm^-1": p.q, "period_nm": p.period_nm, "theta_deg": p.theta_deg, "model": p.model,
             "amplitude": p.amplitude, "background": p.background, "FWHM_x_nm^-1": p.fwhm_x,
             "FWHM_y_nm^-1": p.fwhm_y, "delta_q_nm^-1": p.delta_q, "xi_1_over_deltaq_nm": p.xi_nm,
             "R2": p.r2, "AIC": p.aic, "success": p.success}
            for p in self.fft_fits
        ])
        write_csv(self.output_dir / "afm_fft_unique_paired_periodicities.csv", self.paired_fft)
        write_csv(self.output_dir / "afm_fft_harmonic_candidates.csv", self.harmonics)
        write_csv(self.output_dir / "afm_autocorrelation_peaks.csv", [
            {"rank": p.rank, "px": p.px, "py": p.py, "dx_nm": p.dx_nm, "dy_nm": p.dy_nm,
             "r_nm": p.r_nm, "theta_deg": p.theta_deg, "acf_value": p.value}
            for p in self.raw_acf_peaks
        ])
        write_csv(self.output_dir / "afm_autocorrelation_paired_vectors.csv", self.acf_pairs)

        if self.crystal_refs:
            write_csv(self.output_dir / "crystal_reference_periodicities.csv", self.crystal_refs)
        if self.crystal_comparison:
            write_csv(self.output_dir / "afm_vs_crystal_lattice_comparison.csv", self.crystal_comparison)
        if self.lattice is not None:
            write_csv(self.output_dir / "afm_data_driven_lattice.csv", [lattice_rows(self.lattice)])

        if self.power is not None:
            rq, rp = radial_spectrum(self.power, self.fx, self.fy)
            np.savetxt(
                self.output_dir / "radial_power_spectrum.csv",
                np.column_stack([rq, rp]), delimiter=",",
                header="q_cycles_per_nm,mean_power", comments=""
            )

        if self.ac is not None:
            np.savetxt(self.output_dir / "autocorrelation_matrix.csv", self.ac, delimiter=",")
        np.savetxt(self.output_dir / "image_matrix.csv", self.img, delimiter=",")

        self._write_summary()
        print(f"All results saved to: {self.output_dir.resolve()}")

    def _write_summary(self):
        ny, nx = self.img.shape
        dqx = float(np.mean(np.diff(self.fx))) if self.fx is not None else float("nan")
        dqy = float(np.mean(np.diff(self.fy))) if self.fy is not None else float("nan")
        lattice = self.lattice

        sections = [
            ("Input / calibration", [
                f"Input file: {self.file}",
                f"Image shape: {ny} x {nx}",
                f"FOV override: {self.fov_nm}",
                f"dx={self.dx:.10f} nm, dy={self.dy:.10f} nm",
                f"Lx={self.Lx:.10f} nm, Ly={self.Ly:.10f} nm",
                f"dqx={dqx:.10f} cycles/nm, dqy={dqy:.10f} cycles/nm",
            ]),
            ("FFT", [
                f"Detected FFT peaks: {len(self.fft_fits)}",
                f"Paired FFT periodicities: {len(self.paired_fft)}",
                f"FFT threshold: {self.fft_threshold:.10g}" if self.fft_threshold is not None else "n/a",
            ]),
            ("Autocorrelation", [
                f"Detected ACF peaks: {len(self.raw_acf_peaks)}",
                f"Paired ACF vectors: {len(self.acf_pairs)}",
                f"ACF threshold: {self.acf_threshold:.10g}" if self.acf_threshold is not None else "n/a",
            ]),
            ("Data-driven lattice", [
                f"Status: {self.lattice_status}",
                (f"v1=({lattice['b1'][0]:.8f}, {lattice['b1'][1]:.8f}) nm; "
                 f"length={np.linalg.norm(lattice['b1']):.8f} nm" if lattice is not None else "v1: not available"),
                (f"v2=({lattice['b2'][0]:.8f}, {lattice['b2'][1]:.8f}) nm; "
                 f"length={np.linalg.norm(lattice['b2']):.8f} nm" if lattice is not None else "v2: not available"),
                f"beta={lattice['beta_deg']:.6f} deg" if lattice is not None else "beta: n/a",
                f"area={lattice['area_nm2']:.8f} nm^2" if lattice is not None else "area: n/a",
                f"ACF coverage={lattice['acf_coverage']:.6f}" if lattice is not None else "ACF coverage: n/a",
                f"FFT coverage={lattice['fft_coverage']:.6f}" if lattice is not None else "FFT coverage: n/a",
            ]),
            ("Interpretation note", [
                "The lattice basis is extracted from measured periodicity only; no a/c length search windows are used.",
                "v1/v2 are basis vectors. Shorter/longer are reported as length-based labels only.",
                "Crystallographic a/c assignments are tested in both possible permutations.",
                "A close numerical match is a consistency test, not proof of crystallographic identity.",
            ]),
        ]
        write_summary(self.output_dir / "analysis_summary.txt", sections)

    def print_report(self):
        print("\n" + "=" * 82)
        print("AFM FFT + 2D AUTOCORRELATION: DATA-DRIVEN LATTICE ANALYSIS")
        print("=" * 82)
        print(f"Input: {self.file}")
        ny, nx = self.img.shape
        print(f"Image shape: {ny} x {nx}")
        print(f"dx={self.dx:.8f} nm, dy={self.dy:.8f} nm")
        print(f"Detected FFT peaks: {len(self.fft_fits)}, paired: {len(self.paired_fft)}")
        print(f"Detected ACF peaks: {len(self.raw_acf_peaks)}, paired: {len(self.acf_pairs)}")

        print("\n--- UNIQUE FFT PERIODICITIES ---")
        for p in self.paired_fft:
            print(f"P{p['pair_id']:02d}: q={p['q_nm^-1']:.6f} nm^-1, d={p['period_nm']:.4f} nm, "
                  f"theta={p['theta_deg']:.2f} deg")

        print("\n--- HARMONIC CANDIDATES ---")
        if self.harmonics:
            for h in self.harmonics:
                print(f"P{h['candidate_pair_id']} / P{h['base_pair_id']}: n={h['harmonic_order']}, "
                      f"ratio={h['q_ratio']:.4f}, deviation={100*h['relative_deviation']:.2f}%, "
                      f"same orientation={h['orientation_consistent']}")
        else:
            print("No harmonic candidates found within the specified tolerance.")

        print("\n--- GLOBAL LATTICE RESULT ---")
        if self.lattice is None:
            print(f"No lattice: {self.lattice_status}")
        else:
            l = self.lattice
            b1, b2 = l["b1"], l["b2"]
            print(f"Status: {self.lattice_status}")
            print(f"v1 = ({b1[0]:.4f}, {b1[1]:.4f}) nm, |v1| = {np.linalg.norm(b1):.4f} nm")
            print(f"v2 = ({b2[0]:.4f}, {b2[1]:.4f}) nm, |v2| = {np.linalg.norm(b2):.4f} nm")
            print(f"beta = {l['beta_deg']:.3f} deg, area = {l['area_nm2']:.4f} nm^2")
            print(f"ACF coverage = {l['acf_coverage']:.3f}, RMS = {l['acf_rms_nm']:.4f} nm")
            if np.isfinite(l["fft_coverage"]):
                print(f"FFT coverage = {l['fft_coverage']:.3f}, RMS = {l['fft_rms_nm^-1']:.6f} nm^-1")

        if self.crystal_comparison:
            print("\n--- CRYSTAL COMPARISON ---")
            for r in self.crystal_comparison:
                print(f"{r['phase']:<14s} | {r['assignment']:<12s} | "
                      f"a diff={r['a_difference_percent']:6.2f}% | "
                      f"c diff={r['c_difference_percent']:6.2f}% | "
                      f"beta diff={r['beta_abs_difference_deg']:6.2f} deg")

    def run_all(self, crystal_phases=None, verbose=True):
        self.load(verbose=verbose)
        self.run_fft()
        self.run_autocorrelation()
        self.find_lattice()
        if crystal_phases:
            self.compare_to_crystal_phases(crystal_phases)
        self.plot_all()
        self.save_all()
        if verbose:
            self.print_report()
        return self

    def _ensure_loaded(self):
        if self.img is None:
            raise RuntimeError("Call .load() first (or .run_all()).")


_SETTINGS_DOCS = {
    "FFT preprocessing": [
        ("remove_mean", "fixed",
         "Subtracts the mean height before the FFT so the DC (q=0) peak doesn't dominate. Leave True."),
        ("use_hann_window", "fixed",
         "Tapers image edges to zero before the FFT to avoid spectral leakage from sharp edges. Leave True."),
    ],
    "FFT peak detection": [
        ("exclude_dc_px", "caution",
         "Radius (px) around q=0 ignored when hunting for peaks. Too small lets DC artifacts masquerade as peaks."),
        ("fft_min_separation_px", "safe",
         "Minimum pixel distance between two accepted peaks, so one broad blob isn't counted twice."),
        ("fft_threshold_sigma", "safe",
         "Peak must exceed background by this many std-devs to count. Lower = more (noisier) peaks detected."),
        ("max_fft_peaks", "safe",
         "Hard cap on number of FFT peaks kept, mainly a performance/sanity limit."),
        ("fft_fit_radius_px", "safe",
         "Half-width (px) of the patch used to fit a 2D Gaussian/Lorentzian to each peak for sub-pixel precision."),
    ],
    "FFT peak pairing / harmonics": [
        ("fft_pair_tol_nm_inv", "safe",
         "Max distance (cycles/nm) between a peak and its -q mirror to call them one physical periodicity."),
        ("harmonic_tolerance", "safe",
         "Allowed relative deviation from an exact integer ratio when flagging a peak as a harmonic of another."),
        ("max_harmonic_order", "safe",
         "Highest integer multiple considered when checking for harmonics (e.g. 2nd, 3rd... order)."),
        ("harmonic_angle_tol_deg", "safe",
         "Max orientation difference (deg) between base and candidate harmonic peak to call them 'aligned'."),
    ],
    "Autocorrelation (ACF) peak detection": [
        ("acf_smooth_sigma_px", "safe",
         "Gaussian blur (px) applied to the ACF map before peak-hunting, to suppress pixel-noise false peaks."),
        ("acf_exclude_center_px", "caution",
         "Radius (px) around zero-lag excluded from peak search (the trivial self-correlation peak at r=0)."),
        ("acf_min_separation_px", "safe",
         "Minimum pixel distance between two accepted ACF peaks."),
        ("acf_threshold_sigma", "safe",
         "ACF peak must exceed background by this many std-devs to count. Lower = more (noisier) peaks."),
        ("max_acf_peaks", "safe",
         "Hard cap on number of ACF peaks kept."),
        ("acf_pair_tol_nm", "safe",
         "Max distance (nm) between a peak and its -r mirror to call them one real-space lattice vector."),
    ],
    "Global lattice search": [
        ("max_acf_vectors_for_lattice", "safe",
         "How many of the strongest ACF vectors are fed into the combinatorial lattice search (speed/quality trade-off)."),
        ("max_fft_vectors_for_lattice", "safe",
         "How many of the strongest FFT vectors are used as an independent cross-check on candidate lattices."),
        ("min_basis_angle_deg", "caution",
         "Candidate lattice vectors closer to parallel than this are rejected as degenerate."),
        ("max_basis_angle_deg", "caution",
         "Candidate lattice vectors closer to anti-parallel than this are rejected as degenerate."),
        ("acf_match_tol_nm", "caution",
         "How close (nm) a predicted lattice point must land to an observed ACF peak to count as 'explained'. "
         "Tightening this makes the search stricter and may lower reported coverage."),
        ("fft_match_tol_pixels", "caution",
         "Same idea as acf_match_tol_nm, but for matching predicted reciprocal-lattice points to FFT peaks "
         "(in units of the FFT grid spacing)."),
        ("max_lattice_index", "safe",
         "How far out (in integer multiples of v1, v2) predicted lattice points are generated for comparison."),
        ("min_acf_coverage", "caution",
         "Minimum weighted fraction of ACF peak intensity a candidate basis must explain to be trusted. "
         "Lower = more permissive, higher = stricter (may return 'no lattice found' more often)."),
        ("min_fft_coverage", "caution",
         "Same as min_acf_coverage, but for the independent FFT cross-check."),
    ],
    "Plotting": [
        ("save_figures", "safe", "Whether to write figure files to output_dir."),
        ("show_figures", "safe", "Whether to display figures inline (e.g. in Jupyter)."),
        ("dpi", "safe", "Resolution of saved figures."),
        ("realspace_cmap", "safe", "Colormap for the real-space height image."),
        ("fft_cmap", "safe", "Colormap for the FFT power spectrum."),
        ("acf_cmap", "safe", "Colormap for the autocorrelation map."),
    ],
}

def describe_settings(current=None, group=None):
    defaults = Settings()
    ref = current if current is not None else defaults

    flag_symbol = {"safe": "✓ safe", "caution": "△ caution", "fixed": "✗ fixed"}

    for section, entries in _SETTINGS_DOCS.items():
        if group is not None and group.lower() not in section.lower():
            continue
        print(f"\n{section}")
        print("-" * len(section))
        for name, flag, desc in entries:
            value = getattr(ref, name)
            default_value = getattr(defaults, name)
            marker = flag_symbol.get(flag, flag)
            changed = " (changed from default)" if current is not None and value != default_value else ""
            print(f"  [{marker}] {name} = {value!r}{changed}")
            print(f"      {desc}")

    if group is None:
        print(
            "\nLegend: ✓ safe = tune freely | △ caution = understand before changing "
            "| ✗ fixed = leave alone (breaks the underlying math/convention)\n"
        )