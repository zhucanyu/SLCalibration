"""3D reconstruction: undistortion + linear least-squares triangulation.

Implements the reconstruction step from:
  Zhu et al., Appl. Opt. 61(27), 8050 (2022).
"""

import numpy as np


def undistort_points(points, mtx, dist, f_skew=0.0, n_iters=50):
    """Iterative inverse of the OpenCV distortion model.

    Parameters
    ----------
    points : ndarray (N, 2)
        Distorted image points.
    mtx : ndarray (3, 3)
        Camera intrinsic matrix.
    dist : ndarray (5,)
        Distortion coefficients [k1, k2, p1, p2, k3].
    f_skew : float
        Skew parameter (default 0).
    n_iters : int
        Number of refinement iterations.

    Returns
    -------
    ndarray (N, 2) undistorted points.
    """
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    k1, k2, p1, p2, k3 = dist.ravel()
    N = points.shape[0]
    points_u = np.full((N, 2), np.nan)

    for i in range(N):
        xd, yd = points[i, 0], points[i, 1]
        y11 = (yd - cy) / fy
        x11 = (xd - cx - f_skew * y11) / fx
        r2 = x11 * x11 + y11 * y11
        denom = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xp = (x11 - 2 * p1 * x11 * y11 - p2 * (r2 + 2 * x11 * x11)) / denom
        yp = (y11 - 2 * p2 * x11 * y11 - p1 * (r2 + 2 * y11 * y11)) / denom
        points_u[i, 0] = cx + fx * xp + f_skew * yp
        points_u[i, 1] = cy + fy * yp

        for _ in range(n_iters):
            y11 = (points_u[i, 1] - cy) / fy
            x11 = (points_u[i, 0] - cx - f_skew * y11) / fx
            r2 = x11 * x11 + y11 * y11
            denom = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            xp = ((xd - cx - f_skew * y11) / fx
                  - 2 * p1 * x11 * y11 - p2 * (r2 + 2 * x11 * x11)) / denom
            yp = ((yd - cy) / fy
                  - 2 * p2 * x11 * y11 - p1 * (r2 + 2 * y11 * y11)) / denom
            points_u[i, 0] = cx + fx * xp + f_skew * yp
            points_u[i, 1] = cy + fy * yp
    return points_u


def triangulate_linear_ls(pointsL, pointsR, mtxl, mtxr, R, T):
    """Linear least-squares triangulation from a pair of matched points.

    Constructs the 4x3 system P * M = Q per correspondence and solves
    M = (P^T P)^{-1} P^T Q.

    Parameters
    ----------
    pointsL, pointsR : ndarray (N, 2)
        Undistorted points in the left and right images.
    mtxl, mtxr : ndarray (3, 3)
    R : ndarray (3, 3) rotation (right → left).
    T : ndarray (3,) translation (right → left).

    Returns
    -------
    ndarray (N, 3) 3D points.
    """
    fxl, fyl = mtxl[0, 0], mtxl[1, 1]
    cxl, cyl = mtxl[0, 2], mtxl[1, 2]
    fxr, fyr = mtxr[0, 0], mtxr[1, 1]
    cxr, cyr = mtxr[0, 2], mtxr[1, 2]
    fsl = fsr = 0.0
    N = pointsL.shape[0]
    points3d = np.full((N, 3), np.nan)

    for i in range(N):
        xl, yl = pointsL[i, 0], pointsL[i, 1]
        xr, yr = pointsR[i, 0], pointsR[i, 1]
        P = np.zeros((4, 3))
        Q = np.zeros(4)
        P[0, :] = [-fxl, -fsl, xl - cxl]
        P[1, :] = [0, -fyl, yl - cyl]
        P[2, :] = [(xr - cxr) * R[2, 0] - fxr * R[0, 0] - fsr * R[1, 0],
                   (xr - cxr) * R[2, 1] - fxr * R[0, 1] - fsr * R[1, 1],
                   (xr - cxr) * R[2, 2] - fxr * R[0, 2] - fsr * R[1, 2]]
        P[3, :] = [(yr - cyr) * R[2, 0] - fyr * R[1, 0],
                   (yr - cyr) * R[2, 1] - fyr * R[1, 1],
                   (yr - cyr) * R[2, 2] - fyr * R[1, 2]]
        Q[2] = fxr * T[0] + fsr * T[1] - (xr - cxr) * T[2]
        Q[3] = fyr * T[1] - (yr - cyr) * T[2]
        M, _, _, _ = np.linalg.lstsq(P.T @ P, P.T @ Q, rcond=None)
        points3d[i, :] = M
    return points3d


def center_and_normalize(points3d):
    """Centre and zero-min the point cloud for visualisation."""
    points3d = points3d.copy()
    points3d[:, 2] -= np.mean(points3d[:, 2])
    points3d[:, 2] -= np.min(points3d[:, 2])
    points3d[:, 1] -= np.mean(points3d[:, 1])
    points3d[:, 0] -= np.mean(points3d[:, 0])
    return points3d
