#!/usr/bin/env python
"""SLCalibration — Structured Light System Calibration via DIC.

Reference:
  Zhu, C., Zhang, Q., Hou, J., & Gao, H.
  "Calibration Method for Structured Light System Based on DIC."
  Applied Optics, Vol. 61, No. 27, pp. 8050-8059 (2022).

Pipeline: detect → match → calibrate → (optional: reconstruct)

Usage
-----
  python pipeline.py -d ./data                          # full pipeline
  python pipeline.py -d ./data -m match                 # DIC matching only
  python pipeline.py -d ./data -m detect                # circle detection only
  python pipeline.py -d ./data -m calib                 # calibration only
  python pipeline.py -d ./data --search_mode sift       # use SIFT-guided search
  python pipeline.py -c config.json                     # load params from JSON
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circle_detector import detect_and_save
from dic_matcher import match_all_points
from calibrator import (load_centers_csv, load_reference_centers,
                         stereo_calibrate, save_calib_results)
from reconstructor import (undistort_points, triangulate_linear_ls,
                            center_and_normalize)
from utils import read_csv, write_csv, imread_unicode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_images(directory, extensions=None):
    if extensions is None:
        extensions = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    return sorted(f for f in os.listdir(directory)
                  if Path(f).suffix.lower() in extensions)


def _hdr(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def _kv(k, v):
    print(f"  {k:<22s}: {v}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_detect(data_dir, board_size, n_images, symmetric=False,
                img_ext='bmp'):
    """Step 1 — Circle detection."""
    _hdr("Step 1/4 — Circle Detection")
    img_dir = os.path.join(data_dir, 'board_pics')
    out_dir = os.path.join(data_dir, 'centers_out')
    os.makedirs(out_dir, exist_ok=True)

    # 优先用无散斑图 (*off.*) 做检测; 没有则用原图
    off_pattern = '{i}off.' + img_ext
    ref_pattern = '{i}.' + img_ext
    target_pattern = off_pattern
    test_path = os.path.join(img_dir, off_pattern.format(i=1))
    if not os.path.exists(test_path):
        target_pattern = ref_pattern
        print("  No *off.* images found, using original images for detection.")

    detect_and_save(target_pattern, n_images, img_dir, out_dir,
                    method='auto', board_size=board_size,
                    symmetric=symmetric)
    return out_dir


def step_match(data_dir, n_images, r=30, search_radius=200, order=1,
               search_mode='propagate', img_ext='bmp'):
    """Step 2 — DIC matching."""
    _hdr("Step 2/4 — DIC Matching (ZNSSD + IC-GN)")
    board_dir = os.path.join(data_dir, 'board_pics')
    speckle_dir = os.path.join(data_dir, 'speckle_pic')
    centers_dir = os.path.join(data_dir, 'centers_out')
    out_dir = os.path.join(data_dir, 'centersnew_out')
    os.makedirs(out_dir, exist_ok=True)

    for d, name in [(board_dir, 'board_pics'),
                    (speckle_dir, 'speckle_pic'),
                    (centers_dir, 'centers_out')]:
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Directory missing: {d} ({name}). Run --mode detect first.")

    speckle_files = _find_images(speckle_dir)
    if not speckle_files:
        raise FileNotFoundError(f"No images in {speckle_dir}")
    speckle_img = imread_unicode(
        os.path.join(speckle_dir, speckle_files[0]), cv2.IMREAD_GRAYSCALE)

    _kv("Speckle image", f"{speckle_files[0]} "
                         f"({speckle_img.shape[1]}x{speckle_img.shape[0]})")
    _kv("Subset radius", r)
    _kv("Search radius", search_radius)
    _kv("Shape function", f"order={order} ({6*order} params)")
    _kv("Search mode", search_mode)

    board_files = _find_images(board_dir)
    _kv("Board images", f"{len(board_files)} files")

    ref_files = [f for f in board_files
                 if 'off' not in Path(f).stem.lower()]
    if len(ref_files) < n_images:
        ref_files = [f for i, f in enumerate(board_files) if i % 2 == 0]
    if len(ref_files) < n_images:
        ref_files = board_files[:n_images]

    for count in range(1, n_images + 1):
        if count - 1 >= len(ref_files):
            print(f"  [SKIP] no reference image for index {count}")
            continue
        ref_fname = ref_files[count - 1]
        ref_img = imread_unicode(os.path.join(board_dir, ref_fname),
                                 cv2.IMREAD_GRAYSCALE)
        print(f"\n  [{count}/{n_images}] {ref_fname} "
              f"({ref_img.shape[1]}x{ref_img.shape[0]})")

        centers_path = os.path.join(centers_dir, f'centers{count}.csv')
        if not os.path.exists(centers_path):
            raise FileNotFoundError(
                f"{centers_path} not found. Run --mode detect first.")
        centers = read_csv(centers_path)
        n_pts = centers.shape[0]
        print(f"    Circles: {n_pts}")

        t0 = time.perf_counter()
        centers_new = match_all_points(
            ref_img, speckle_img, centers,
            r=r, search_radius=search_radius,
            order=order, search_mode=search_mode, verbose=True,
        )
        elapsed = time.perf_counter() - t0
        ok_count = (~np.isnan(centers_new[:, 0])).sum()
        print(f"    Matched: {ok_count}/{n_pts}  ({elapsed:.0f}s)")

        np.savetxt(os.path.join(out_dir, f'centersnew{count}.csv'),
                   centers_new, delimiter=',', fmt='%.10f')
    return out_dir


def step_calibrate(data_dir, n_images):
    """Step 3 — Stereo calibration."""
    _hdr("Step 3/4 — Stereo Calibration")
    centers_dir = os.path.join(data_dir, 'centers_out')
    centersnew_dir = os.path.join(data_dir, 'centersnew_out')
    reference_path = os.path.join(data_dir, 'referencecenters.csv')
    out_dir = os.path.join(data_dir, 'calib_out')

    for p, name in [(centers_dir, 'centers_out/'),
                    (centersnew_dir, 'centersnew_out/'),
                    (reference_path, 'referencecenters.csv')]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {name}. Run previous steps first.")

    centersL = load_centers_csv('centers{}.csv', n_images, centers_dir)
    centersR = load_centers_csv('centersnew{}.csv', n_images, centersnew_dir)
    obj_points = load_reference_centers(reference_path)

    _kv("Image pairs", len(centersL))
    _kv("World points / board", obj_points.shape[0])

    board_dir = os.path.join(data_dir, 'board_pics')
    img_files = _find_images(board_dir)
    if img_files:
        img = imread_unicode(os.path.join(board_dir, img_files[0]))
        image_size = (img.shape[1], img.shape[0])
    else:
        image_size = (2592, 2048)
    _kv("Image size", f"{image_size[0]} x {image_size[1]}")

    mtxl, distl, mtxr, distr, R, T, err = stereo_calibrate(
        centersL, centersR, obj_points, image_size)
    _kv("Stereo reproj. error", f"{err:.4f}")
    save_calib_results(out_dir, mtxl, distl, mtxr, distr, R, T)
    return out_dir


def step_reconstruct(data_dir, visualize=False):
    """Step 4 — 3D reconstruction (validation)."""
    _hdr("Step 4/4 — 3D Reconstruction (validation)")

    calib_dir = os.path.join(data_dir, 'calib_out')
    if not os.path.exists(calib_dir):
        calib_dir = os.path.join(data_dir, 'rebiuld')
    for fn in ('R.csv', 'T.csv', 'mtxl.csv', 'mtxr.csv',
               'distl.csv', 'distr.csv'):
        if not os.path.exists(os.path.join(calib_dir, fn)):
            raise FileNotFoundError(
                f"{fn} missing in {calib_dir}. Run --mode calib first.")

    R = read_csv(os.path.join(calib_dir, 'R.csv'))
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    T = read_csv(os.path.join(calib_dir, 'T.csv')).ravel()
    mtxl = read_csv(os.path.join(calib_dir, 'mtxl.csv'))
    mtxr = read_csv(os.path.join(calib_dir, 'mtxr.csv'))
    distl = read_csv(os.path.join(calib_dir, 'distl.csv'))
    distr = read_csv(os.path.join(calib_dir, 'distr.csv'))

    cl_path = os.path.join(data_dir, 'cl.csv')
    cr_path = os.path.join(data_dir, 'cr.csv')
    if not os.path.exists(cl_path):
        cl_path = os.path.join(calib_dir, 'cl.csv')
        cr_path = os.path.join(calib_dir, 'cr.csv')
    if not os.path.exists(cl_path):
        raise FileNotFoundError(
            "cl.csv / cr.csv not found. "
            "Place match-point files in the data directory or calib_out/.")

    cl = read_csv(cl_path)
    cr = read_csv(cr_path)
    _kv("Match points", cl.shape[0])

    t0 = time.perf_counter()
    pl_u = undistort_points(cl, mtxl, distl)
    pr_u = undistort_points(cr, mtxr, distr)
    _kv("Undistort", f"{time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    pc = triangulate_linear_ls(pl_u, pr_u, mtxl, mtxr, R, T)
    _kv("Triangulate", f"{time.perf_counter() - t0:.2f}s")

    pc = center_and_normalize(pc)
    _kv("X range", f"[{pc[:, 0].min():.2f}, {pc[:, 0].max():.2f}]")
    _kv("Y range", f"[{pc[:, 1].min():.2f}, {pc[:, 1].max():.2f}]")
    _kv("Z range", f"[{pc[:, 2].min():.2f}, {pc[:, 2].max():.2f}]")

    out_dir = os.path.join(data_dir, 'recon_out')
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, 'pc1.csv'), pc)

    if visualize:
        try:
            _show_point_cloud(pc)
        except Exception as e:
            print(f"  [WARN] visualisation failed: {e}")

    return pc


def _show_point_cloud(points3d, sample=5000):
    import matplotlib.pyplot as plt
    idx = np.random.choice(points3d.shape[0],
                           min(sample, points3d.shape[0]), replace=False)
    pc = points3d[idx]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
               c=pc[:, 2], cmap='viridis', s=1, alpha=0.8)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Reconstructed Point Cloud')
    ax.set_box_aspect([1, 1, 0.5])
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _merge_config(args):
    if not args.config or not os.path.exists(args.config):
        return args
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    defaults = {
        'data_dir': '.', 'mode': 'all', 'n_images': 5,
        'board_width': 5, 'board_height': 3, 'symmetric': False,
        'r': 30, 'search_radius': 200, 'order': 1,
        'search_mode': 'propagate', 'img_ext': 'bmp', 'visualize': False,
    }
    for key, default in defaults.items():
        cfg_val = cfg.get(key)
        has_cli = any(a.startswith(f'--{key}') or a.startswith(f'-{key[0]}')
                      for a in sys.argv if not a.startswith('-c'))
        if not has_cli and cfg_val is not None:
            setattr(args, key, cfg_val)
    return args


def main():
    parser = argparse.ArgumentParser(
        description='SLCalibration — Structured Light System Calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py -d ./data
  python pipeline.py -d ./data -m match --search_mode sift
  python pipeline.py -d ./data -n 7 -W 7 -H 5 -o 2
  python pipeline.py -c config.json
        """,
    )
    parser.add_argument('-d', '--data_dir', default='.',
                        help='root data directory')
    parser.add_argument('-m', '--mode', default='all',
                        choices=['all', 'detect', 'match', 'calib', 'recon'])
    parser.add_argument('-n', '--n_images', type=int, default=5)
    parser.add_argument('-W', '--board_width', type=int, default=5)
    parser.add_argument('-H', '--board_height', type=int, default=3)
    parser.add_argument('--symmetric', action='store_true', default=False)
    parser.add_argument('-r', '--r', type=int, default=30,
                        help='subset radius (pixels)')
    parser.add_argument('-s', '--search_radius', type=int, default=200,
                        help='integer search radius (pixels)')
    parser.add_argument('-o', '--order', type=int, default=1,
                        choices=[1, 2],
                        help='shape function order (1 or 2)')
    parser.add_argument('--search_mode', default='propagate',
                        choices=['brute', 'sift', 'propagate', 'fast'],
                        help='integer search strategy')
    parser.add_argument('-c', '--config', default=None,
                        help='JSON config file path')
    parser.add_argument('--img_ext', default='bmp',
                        help='image file extension')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='show point cloud after reconstruction')
    args = parser.parse_args()
    args = _merge_config(args)

    data_dir = os.path.abspath(args.data_dir)
    board_size = (args.board_width, args.board_height)

    # 'all' runs detect→match→calib only; reconstruction is opt-in.
    run_detect = args.mode in ('all', 'detect')
    run_match = args.mode in ('all', 'match')
    run_calib = args.mode in ('all', 'calib')
    run_recon = args.mode == 'recon'

    if not os.path.isdir(os.path.join(data_dir, 'board_pics')):
        print("SLCalibration — Structured Light Calibration via DIC")
        print(f"  Data directory : {data_dir}")
        print(f"  [ERROR] 'board_pics/' not found under data directory.")
        print(f"  Use -d/--data_dir to point at the correct data folder.")
        print(f"  See README.md for the required directory structure.")
        sys.exit(1)
    if run_detect and len(_find_images(os.path.join(data_dir, 'board_pics'))) == 0:
        print(f"  [ERROR] No image files in '{data_dir}/board_pics/'.")
        print(f"  Supported formats: .bmp, .png, .jpg, .tif")
        sys.exit(1)

    print("SLCalibration — Structured Light Calibration via DIC")
    print(f"  Data directory : {data_dir}")
    print(f"  Mode           : {args.mode}")

    t_total = time.perf_counter()
    try:
        if run_detect:
            step_detect(data_dir, board_size, args.n_images,
                        args.symmetric, args.img_ext)
        if run_match:
            step_match(data_dir, args.n_images, args.r, args.search_radius,
                       args.order, args.search_mode, args.img_ext)
        if run_calib:
            step_calibrate(data_dir, args.n_images)
        if run_recon:
            step_reconstruct(data_dir, visualize=args.visualize)

        print(f"\n{'='*60}\n  Done — {time.perf_counter() - t_total:.1f}s\n{'='*60}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\nCheck data directory structure.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
