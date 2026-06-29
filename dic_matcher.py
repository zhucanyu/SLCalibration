"""DIC matching module: integer-pixel ZNSSD search + IC-GN subpixel refinement.

Implements the algorithm from:
  Zhu et al., "Calibration Method for Structured Light System Based on DIC",
  Applied Optics, Vol. 61, No. 27, 2022.

Search strategies (--search_mode):
  propagate  Neighborhood propagation: first point brute-force, rest
             use nearest-matched-point displacement as prior with
             expanding-ring search. Default. Fast and accurate.
  brute      Full brute-force integer search. Matches MATLAB exactly. Slow.
  sift       SIFT global displacement prior + narrow window search.
  fast       Stride-2 coarse-to-fine scan. Fastest, may drift on some points.
"""

import time

import cv2
import numpy as np
from numpy.linalg import inv


# ---------------------------------------------------------------------------
# Core DIC routines (match original MATLAB implementation exactly)
# ---------------------------------------------------------------------------

def gradient_1d(Y, N):
    """1D gradient via tridiagonal system solve (cf. gradient.m)."""
    DM = np.zeros((N, N))
    D = np.zeros(N)
    for i in range(1, N - 1):
        DM[i, i - 1:i + 2] = [1, 4, 1]
        D[i] = 3 * Y[i + 1] - 3 * Y[i - 1]
    DM[0, 0:2] = [2, 1]
    D[0] = 3 * Y[1] - 3 * Y[0]
    DM[N - 1, N - 2:N] = [1, 2]
    D[N - 1] = 3 * Y[N - 1] - 3 * Y[N - 2]
    return np.linalg.solve(DM, D)


def gradient_2d(subset):
    """2D gradient on a (2r+1)x(2r+1) subset (cf. gradient2.m)."""
    S = subset.shape
    r = (S[0] - 1) // 2
    gra_x = np.zeros((2 * r + 1, 2 * r + 1))
    gra_y = np.zeros((2 * r + 1, 2 * r + 1))
    for m in range(2 * r + 1):
        gra_x[m, :] = gradient_1d(subset[m, :], 2 * r + 1)
    for n in range(2 * r + 1):
        gra_y[:, n] = gradient_1d(subset[:, n], 2 * r + 1)
    return gra_x, gra_y


def calc_correlation_coefficient(refer_subset, deformed_subset):
    """ZNSSD coefficient, lower is better (cf. Calc_correlation_coefficient.m)."""
    m, n = refer_subset.shape
    aver_refer = np.sum(refer_subset) / (m * n)
    aver_deformed = np.sum(deformed_subset) / (m * n)
    f_bar = refer_subset - aver_refer
    g_bar = deformed_subset - aver_deformed
    std_f = np.sqrt(np.sum(f_bar ** 2))
    std_g = np.sqrt(np.sum(g_bar ** 2))
    return np.sum((f_bar / std_f - g_bar / std_g) ** 2)


def _bicubic_spline_interp(around, d_x, d_y):
    """Bicubic spline on a 6x6 neighbourhood (cf. Interpolation_deform.m)."""
    Q = np.array([
        [1 / 11, -6 / 11, 13 / 11, -13 / 11, 6 / 11, -1 / 11],
        [-45 / 209, 270 / 209, -453 / 209, 288 / 209, -72 / 209, 12 / 209],
        [26 / 209, -156 / 209, -3 / 209, 168 / 209, -42 / 209, 7 / 209],
        [0, 0, 1, 0, 0, 0],
    ])
    Vx = np.array([d_x ** 3, d_x ** 2, d_x, 1])
    Vy = np.array([d_y ** 3, d_y ** 2, d_y, 1])
    return Vy @ Q @ around @ Q.T @ Vx


def interpolation_deform(image_deformed, P, x_pos, y_pos, r):
    """Warp deformed subset via first-order shape function + bicubic spline."""
    mat = np.zeros((2 * r + 1, 2 * r + 1))
    for patch_m in range(2 * r + 1):
        for patch_n in range(2 * r + 1):
            x = patch_n - r
            y = patch_m - r
            x_1 = (P[1] + 1) * x + P[2] * y + P[0]
            y_1 = P[4] * x + (P[5] + 1) * y + P[3]
            x_new = x_pos + x_1
            y_new = y_pos + y_1
            m = int(np.floor(y_new))
            n = int(np.floor(x_new))
            around = image_deformed[m - 2:m + 4, n - 2:n + 4].astype(np.float64)
            d_y = y_new - m
            d_x = x_new - n
            mat[patch_m, patch_n] = _bicubic_spline_interp(around, d_x, d_y)
    return mat


def middle_mat(refer_subset, r, order=1):
    """Precompute -(J^T J)^{-1} J^T for IC-GN (cf. Middle_mat.m).

    order=1: 6-parameter first-order shape function.
    order=2: 12-parameter second-order shape function.
    """
    gra_x, gra_y = gradient_2d(refer_subset)
    N_sub = (2 * r + 1) ** 2
    n_param = 6 if order == 1 else 12
    J = np.zeros((N_sub, n_param))
    i = 0
    for patch_m in range(2 * r + 1):
        for patch_n in range(2 * r + 1):
            G = np.array([gra_x[patch_m, patch_n], gra_y[patch_m, patch_n]])
            x = patch_n - r
            y = patch_m - r
            if order == 1:
                dW = np.array([[1, x, y, 0, 0, 0], [0, 0, 0, 1, x, y]])
            else:
                dW = np.array([[1, x, y, x * x, x * y, y * y, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, x, y, x * x, x * y, y * y]])
            J[i, :] = G @ dW
            i += 1
    return -inv(J.T @ J) @ J.T


def ic_gn2(middle_mat, refer_subset, interpolation_deform_subset, r, P, order=1):
    """One IC-GN iteration (cf. IC_GN2.m). Returns (delta_P, P_next)."""
    N_sub = (2 * r + 1) ** 2
    avg_f = np.sum(refer_subset) / N_sub
    avg_g = np.sum(interpolation_deform_subset) / N_sub
    delta_fg = delta_g2 = delta_f2 = 0.0
    for m in range(2 * r + 1):
        for n in range(2 * r + 1):
            df = refer_subset[m, n] - avg_f
            dg = interpolation_deform_subset[m, n] - avg_g
            delta_fg += df * dg
            delta_g2 += dg ** 2
            delta_f2 += df ** 2
    delta_g = np.sqrt(delta_g2)
    delta_f = np.sqrt(delta_f2)
    F = np.zeros(N_sub)
    idx = 0
    for m in range(2 * r + 1):
        for n in range(2 * r + 1):
            F[idx] = (delta_g / delta_f * (refer_subset[m, n] - avg_f)
                      - (interpolation_deform_subset[m, n] - avg_g))
            idx += 1
    delta_P = middle_mat @ F
    if order == 1:
        W_cur = np.array([[P[1] + 1, P[2], P[0]],
                          [P[4], P[5] + 1, P[3]], [0, 0, 1]])
        dW_inv = np.array([[delta_P[1] + 1, delta_P[2], delta_P[0]],
                           [delta_P[4], delta_P[5] + 1, delta_P[3]], [0, 0, 1]])
        dW = np.linalg.inv(dW_inv)
        W_next = W_cur @ dW
        P_next = np.array([W_next[0, 2], W_next[0, 0] - 1, W_next[0, 1],
                           W_next[1, 2], W_next[1, 0], W_next[1, 1] - 1])
    else:
        P_next = P - delta_P
    return delta_P, P_next


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_subset(img, cx, cy, r):
    sub = np.zeros((2 * r + 1, 2 * r + 1))
    for m in range(cy - r, cy + r + 1):
        for n in range(cx - r, cx + r + 1):
            sub[m - cy + r, n - cx + r] = float(img[m, n])
    return sub


def _brute_search_window(image_def, ref_sub, cx_min, cx_max, cy_min, cy_max, r):
    """Brute-force ZNSSD search within [cx_min..cx_max] x [cy_min..cy_max]."""
    H_d, W_d = image_def.shape
    mrg = r + 3
    best_cc = np.inf
    best_x, best_y = cx_min, cy_min
    for sy in range(cy_min, cy_max + 1):
        for sx in range(cx_min, cx_max + 1):
            if sy - mrg < 0 or sy + mrg >= H_d or sx - mrg < 0 or sx + mrg >= W_d:
                continue
            ds = _extract_subset(image_def, sx, sy, r)
            cc = calc_correlation_coefficient(ref_sub, ds)
            if cc < best_cc:
                best_cc = cc
                best_x, best_y = sx, sy
    return best_x, best_y


def _icgn_refine(image_def, ref_sub, x0, y0, U, V, r, order=1, max_iter=20):
    """IC-GN subpixel refinement given integer-pixel initial guess (U, V)."""
    if order == 1:
        P = np.array([U, 0.0, 0.0, V, 0.0, 0.0])
    else:
        P = np.array([U, 0.0, 0.0, 0.0, 0.0, 0.0,
                      V, 0.0, 0.0, 0.0, 0.0, 0.0])
    mm = middle_mat(ref_sub, r, order)
    ok = True
    for IT_count in range(max_iter):
        dint = interpolation_deform(image_def, P, x0, y0, r)
        dp, P_next = ic_gn2(mm, ref_sub, dint, r, P, order)
        P = P_next
        if order == 1:
            nrm = np.sqrt(dp[0]**2 + (dp[1]*r)**2 + (dp[2]*r)**2
                          + dp[3]**2 + (dp[4]*r)**2 + (dp[5]*r)**2)
        else:
            nrm = np.sqrt(dp[0]**2 + (dp[1]*r)**2 + (dp[2]*r)**2
                          + (dp[3]*r*r)**2 + (dp[4]*r*r)**2 + (dp[5]*r*r)**2
                          + dp[6]**2 + (dp[7]*r)**2 + (dp[8]*r)**2
                          + (dp[9]*r*r)**2 + (dp[10]*r*r)**2 + (dp[11]*r*r)**2)
        if nrm <= 0.01:
            break
        if (P[0] - U)**2 + (P[3 if order == 1 else 6] - V)**2 > 100:
            ok = False
            break
    if IT_count >= max_iter - 1:
        ok = False
    return P, ok


def _sift_global_displacement(image_ref, image_def):
    """Estimate global displacement via SIFT keypoint matching."""
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(image_ref, None)
    kp2, des2 = sift.detectAndCompute(image_def, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0, 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 20:
        return 0.0, 0.0
    matches = sorted(matches, key=lambda m: m.distance)
    dxs = [kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0] for m in matches[:150]]
    dys = [kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1] for m in matches[:150]]
    dxs, dys = np.array(dxs), np.array(dys)
    dx_m, dy_m = np.median(dxs), np.median(dys)
    inliers = (np.abs(dxs - dx_m) < 30) & (np.abs(dys - dy_m) < 30)
    return np.median(dxs[inliers]), np.median(dys[inliers])


# ---------------------------------------------------------------------------
# Single-point matching
# ---------------------------------------------------------------------------

def match_one_point(image_ref, image_def, cx, cy, r, search_radius,
                    search_mode='propagate', order=1, max_iter=None,
                    prior_displacement=None, verbose=False):
    """Match one point using the specified search strategy."""
    if max_iter is None:
        max_iter = 20 if order == 1 else 30

    H_d, W_d = image_def.shape
    mrg = r + 3
    x0 = int(np.fix(cx))
    y0 = int(np.fix(cy))

    if (y0 - r < 0 or y0 + r >= image_ref.shape[0]
            or x0 - r < 0 or x0 + r >= image_ref.shape[1]):
        return np.nan, np.nan, False

    ref_sub = _extract_subset(image_ref, x0, y0, r)

    # ---- Integer-pixel search ----
    if search_mode == 'propagate' and prior_displacement is not None:
        pdx, pdy = prior_displacement
        search_cx = x0 + int(pdx)
        search_cy = y0 + int(pdy)
        best_x, best_y = search_cx, search_cy
        for ring in [3, 5, 8, 12, 20, 30, 60, 120, 200]:
            cx_min = max(mrg, search_cx - ring)
            cx_max = min(W_d - 1 - mrg, search_cx + ring)
            cy_min = max(mrg, search_cy - ring)
            cy_max = min(H_d - 1 - mrg, search_cy + ring)
            rx, ry = _brute_search_window(
                image_def, ref_sub, cx_min, cx_max, cy_min, cy_max, r)
            best_x, best_y = rx, ry
            # Stop when the optimum is not pinned to the window edge.
            if not (rx <= cx_min or rx >= cx_max or ry <= cy_min or ry >= cy_max):
                break
        result_x, result_y = best_x, best_y

    elif search_mode == 'sift':
        dx_glob, dy_glob = prior_displacement if prior_displacement else (0, 0)
        sr = 30
        scx, scy = x0 + int(dx_glob), y0 + int(dy_glob)
        cx_min = max(mrg, scx - sr)
        cx_max = min(W_d - 1 - mrg, scx + sr)
        cy_min = max(mrg, scy - sr)
        cy_max = min(H_d - 1 - mrg, scy + sr)
        result_x, result_y = _brute_search_window(
            image_def, ref_sub, cx_min, cx_max, cy_min, cy_max, r)

    elif search_mode == 'brute':
        result_x, result_y = _brute_search_window(
            image_def, ref_sub,
            max(mrg, x0 - search_radius), min(W_d - 1 - mrg, x0 + search_radius),
            max(mrg, y0 - search_radius), min(H_d - 1 - mrg, y0 + search_radius),
            r)

    else:  # 'fast'
        stride = 2
        best_cc = np.inf
        best_x, best_y = x0, y0
        for dy in range(-search_radius, search_radius + 1, stride):
            for dx in range(-search_radius, search_radius + 1, stride):
                sx, sy = x0 + dx, y0 + dy
                if sy - mrg < 0 or sy + mrg >= H_d or sx - mrg < 0 or sx + mrg >= W_d:
                    continue
                ds = _extract_subset(image_def, sx, sy, r)
                cc = calc_correlation_coefficient(ref_sub, ds)
                if cc < best_cc:
                    best_cc = cc
                    best_x, best_y = sx, sy
        cx_min = max(mrg, best_x - stride)
        cx_max = min(W_d - 1 - mrg, best_x + stride)
        cy_min = max(mrg, best_y - stride)
        cy_max = min(H_d - 1 - mrg, best_y + stride)
        result_x, result_y = _brute_search_window(
            image_def, ref_sub, cx_min, cx_max, cy_min, cy_max, r)

    # ---- IC-GN subpixel refinement ----
    U = result_x - x0
    V = result_y - y0
    P, ok = _icgn_refine(image_def, ref_sub, x0, y0, U, V, r, order, max_iter)

    x_new = cx + P[0]
    y_new = cy + (P[3] if order == 1 else P[6])
    return x_new, y_new, ok


# ---------------------------------------------------------------------------
# Batch matching
# ---------------------------------------------------------------------------

def match_all_points(image_ref, image_def, centers, r=30, search_radius=200,
                     order=1, max_iter=None,
                     search_mode='propagate', verbose=True):
    """Match all circle centers via the specified search strategy.

    Parameters
    ----------
    search_mode : str
        'propagate' — first point brute-force, remainder use the displacement
                       of the spatially nearest matched point as prior with
                       expanding-ring search. (default)
        'brute'     — full brute-force search for every point.
        'sift'      — SIFT displacement prior + 30px window.
        'fast'      — stride-2 coarse-to-fine (fastest, occasional drift).
    """
    if max_iter is None:
        max_iter = 20 if order == 1 else 30
    n_points = centers.shape[0]
    centers_new = np.full((n_points, 2), np.nan)
    ok_count = 0
    t_start = time.time()

    dx_glob, dy_glob = 0.0, 0.0
    anchor_results = []  # list of (cx, cy, du, dv) for propagate mode

    if search_mode == 'sift':
        if verbose:
            print("    Estimating SIFT global displacement ...")
        dx_glob, dy_glob = _sift_global_displacement(image_ref, image_def)
        if verbose:
            print(f"    SIFT displacement: ({dx_glob:.1f}, {dy_glob:.1f})")

    for i in range(n_points):
        cx, cy = centers[i, 0], centers[i, 1]

        # Determine prior for propagate / sift modes
        prior = None
        mode_i = search_mode
        if search_mode == 'sift':
            prior = (dx_glob, dy_glob)
        elif search_mode == 'propagate' and len(anchor_results) > 0:
            # Use displacement of nearest already-matched point as prior
            nearest_du, nearest_dv = anchor_results[0][2], anchor_results[0][3]
            nearest_dist = np.inf
            for ax, ay, adu, adv in anchor_results:
                d = (cx - ax)**2 + (cy - ay)**2
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_du, nearest_dv = adu, adv
            prior = (nearest_du, nearest_dv)
        elif search_mode == 'propagate':
            # First point: no prior, use brute-force
            mode_i = 'brute'

        x_new, y_new, ok = match_one_point(
            image_ref, image_def, cx, cy, r, search_radius,
            search_mode=mode_i, order=order, max_iter=max_iter,
            prior_displacement=prior, verbose=False,
        )
        if ok:
            ok_count += 1
        centers_new[i, 0] = x_new
        centers_new[i, 1] = y_new

        if search_mode == 'propagate':
            anchor_results.append((cx, cy, x_new - cx, y_new - cy))

        if verbose and (i % 5 == 0 or i == n_points - 1):
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n_points - i - 1) if i > 0 else 0
            print(f"    [{i + 1:4d}/{n_points}]  OK={ok_count:<4d}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    return centers_new
