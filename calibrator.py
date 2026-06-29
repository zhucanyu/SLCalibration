"""Stereo calibration for camera–projector system via DIC.

Implements the calibration pipeline described in:
  Zhu et al., Appl. Opt. 61(27), 8050 (2022).

Steps:
  1. Calibrate left camera intrinsics.
  2. Calibrate right "camera" (projector) intrinsics from DIC-matched points.
  3. Stereo calibration with fixed intrinsics.
"""

import os

import cv2
import numpy as np


def load_centers_csv(file_pattern, n_images, input_dir):
    """Load circle centers from CSV files.

    Parameters
    ----------
    file_pattern : str
        Filename pattern with '{}' placeholder, e.g. 'centers{}.csv'.
    n_images : int
    input_dir : str

    Returns
    -------
    list of ndarray (N_i, 2) in float32
    """
    centers_list = []
    for i in range(1, n_images + 1):
        fp = os.path.join(input_dir, file_pattern.format(i))
        data = np.loadtxt(fp, delimiter=',', ndmin=2)
        centers_list.append(data.astype(np.float32))
    return centers_list


def load_reference_centers(filepath):
    """Load world coordinates of calibration target circle centres.

    Returns
    -------
    ndarray (N, 3) in float32
    """
    data = np.loadtxt(filepath, delimiter=',', ndmin=2)
    return data.astype(np.float32)


def stereo_calibrate(centersL_list, centersR_list, obj_points, image_size,
                      fix_intrinsics=True):
    """Run stereo calibration.

    Parameters
    ----------
    centersL_list : list of ndarray
        Left image circle centres (camera view).
    centersR_list : list of ndarray
        Right image circle centres (projector view, from DIC matching).
    obj_points : ndarray (N, 3)
        World coordinates of the calibration target.
    image_size : (width, height)
    fix_intrinsics : bool
        If True, fix intrinsics during stereo calibration.

    Returns
    -------
    mtxl, distl : left camera intrinsics and distortion.
    mtxr, distr : right camera intrinsics and distortion.
    R, T : rotation and translation (right → left).
    reproj_err : stereo reprojection error.
    """
    n_images = len(centersL_list)
    obj_pf = [obj_points.copy().astype(np.float32) for _ in range(n_images)]
    imgL_pf = [c.astype(np.float32) for c in centersL_list]
    imgR_pf = [c.astype(np.float32) for c in centersR_list]

    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
                1000, 1e-12)
    sz = image_size

    # Initial intrinsic guess: principal point at image centre,
    # focal length ~ 1.5 * max(image_dim).
    fx_init = float(max(sz)) * 1.5
    cx_init = float(sz[0]) / 2.0
    cy_init = float(sz[1]) / 2.0
    mtx_init = np.array([[fx_init, 0, cx_init],
                         [0, fx_init, cy_init],
                         [0, 0, 1]], dtype=np.float64)

    # Left camera
    ret_l, mtxl, distl, _, _ = cv2.calibrateCamera(
        obj_pf, imgL_pf, sz, mtx_init.copy(), None, criteria=criteria)
    print(f"  left  reprojection error: {ret_l:.4f}")

    # Right "camera" (projector)
    ret_r, mtxr, distr, _, _ = cv2.calibrateCamera(
        obj_pf, imgR_pf, sz, mtx_init.copy(), None, criteria=criteria)
    print(f"  right reprojection error: {ret_r:.4f}")

    # Stereo
    if fix_intrinsics:
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_INTRINSIC
    else:
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
    reproj_err, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        obj_pf, imgL_pf, imgR_pf,
        mtxl, distl, mtxr, distr, sz,
        criteria=criteria, flags=flags)

    print(f"  stereo reprojection error: {reproj_err:.4f}")
    return mtxl, distl, mtxr, distr, R, T, reproj_err


def save_calib_results(output_dir, mtxl, distl, mtxr, distr, R, T):
    """Write calibration results as CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, 'mtxl.csv'), mtxl,
               delimiter=',', fmt='%.10f')
    np.savetxt(os.path.join(output_dir, 'mtxr.csv'), mtxr,
               delimiter=',', fmt='%.10f')
    np.savetxt(os.path.join(output_dir, 'distl.csv'),
               distl.reshape(-1, 1).T, delimiter=',', fmt='%.10f')
    np.savetxt(os.path.join(output_dir, 'distr.csv'),
               distr.reshape(-1, 1).T, delimiter=',', fmt='%.10f')
    np.savetxt(os.path.join(output_dir, 'R.csv'), R,
               delimiter=',', fmt='%.10f')
    np.savetxt(os.path.join(output_dir, 'T.csv'),
               T.reshape(1, -1), delimiter=',', fmt='%.10f')
    print(f"  calibration results saved to {output_dir}/")
