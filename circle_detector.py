"""Circle centre detection for calibration boards.

Two detection strategies:
  - findCirclesGrid   OpenCV native (may fail on OpenCV >= 4.6).
  - adaptive          Adaptive threshold + contour circularity filter.
                      Falls back automatically when findCirclesGrid fails.
"""

import os

import cv2
import numpy as np

from utils import imread_unicode


# ---------------------------------------------------------------------------
# Strategy A: adaptive threshold + contour circularity
# ---------------------------------------------------------------------------

def detect_circles_adaptive(image_path, area_range=(300, 3000),
                             circularity_min=0.7, adaptive_block=101,
                             adaptive_c=-25):
    """Detect circles via adaptive binarisation and contour filtering.

    Intended for white circles on dark background.  No grid layout required.
    Compatible with all OpenCV 4.x versions.

    Parameters
    ----------
    image_path : str
    area_range : (float, float)
        Min / max contour area for a valid circle.
    circularity_min : float
        Minimum 4*pi*area/perimeter^2 (0 … 1).
    adaptive_block : int
        Block size for adaptive threshold (odd).
    adaptive_c : int
        Constant subtracted from mean (negative -> more foreground).

    Returns
    -------
    centers : ndarray (N, 2)
        Circle centres sorted by y coordinate.
    """
    img = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)

    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, adaptive_block, adaptive_c)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_range[0] or area > area_range[1]:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 10:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < circularity_min:
            continue
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            centers.append((M['m10'] / M['m00'], M['m01'] / M['m00']))

    centers = np.array(centers) if centers else np.empty((0, 2))
    if len(centers) > 0:
        centers = centers[np.argsort(centers[:, 1])]
    return centers


# ---------------------------------------------------------------------------
# Strategy B: OpenCV findCirclesGrid
# ---------------------------------------------------------------------------

def detect_circles_grid(image_path, board_size, symmetric=False,
                        blob_params=None):
    """Detect circles via OpenCV findCirclesGrid.

    Note
    ----
    findCirclesGrid behaviour changed in OpenCV >= 4.6 and may fail on
    images that worked with older versions.  When it fails the 'auto'
    method falls back to adaptive detection automatically.
    """
    img = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)

    params = cv2.SimpleBlobDetector_Params()
    bp = blob_params or {}
    params.minThreshold = bp.get('minThreshold', 10)
    params.maxThreshold = bp.get('maxThreshold', 200)
    params.blobColor = bp.get('blobColor', 255)
    params.maxArea = bp.get('maxArea', 5000)
    params.minArea = bp.get('minArea', 10)
    params.minCircularity = bp.get('minCircularity', 0.8)

    detector = cv2.SimpleBlobDetector_create(params)
    pattern = (cv2.CALIB_CB_SYMMETRIC_GRID if symmetric
               else cv2.CALIB_CB_ASYMMETRIC_GRID)
    found, centers = cv2.findCirclesGrid(img, board_size, flags=pattern,
                                         blobDetector=detector)
    return centers if found else None, found


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def detect_circles(image_path, method='auto', board_size=None,
                   symmetric=False, blob_params=None, **adaptive_kwargs):
    """Detect circle centres.  Tries findCirclesGrid first when *method*
    is 'auto' and *board_size* is given; falls back to adaptive detection.

    Parameters
    ----------
    image_path : str
    method : {'auto', 'grid', 'adaptive'}
    board_size : (int, int) or None
        Required for 'grid' / 'auto' methods.
    symmetric : bool
        Use symmetric circle pattern.
    blob_params : dict or None
        Passed through to SimpleBlobDetector_Params.
    adaptive_kwargs :
        Passed through to detect_circles_adaptive.

    Returns
    -------
    centers : ndarray (N, 2)
    """
    if method == 'auto':
        if board_size is not None:
            c, found = detect_circles_grid(image_path, board_size,
                                           symmetric, blob_params)
            if found and c is not None and c.shape[0] > 0:
                return c
        centers = detect_circles_adaptive(image_path, **adaptive_kwargs)
        if centers.shape[0] == 0:
            raise RuntimeError(f"Circle detection failed: {image_path}")
        return centers

    if method == 'grid':
        c, found = detect_circles_grid(image_path, board_size,
                                       symmetric, blob_params)
        if not found or c is None or c.shape[0] == 0:
            raise RuntimeError(f"findCirclesGrid failed: {image_path}")
        return c

    if method == 'adaptive':
        centers = detect_circles_adaptive(image_path, **adaptive_kwargs)
        if centers.shape[0] == 0:
            raise RuntimeError(f"Adaptive detection failed: {image_path}")
        return centers

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def detect_and_save(file_pattern, n_images, input_dir, output_dir,
                    method='auto', board_size=None, symmetric=False,
                    blob_params=None, save_vis=True, **adaptive_kwargs):
    """Batch circle detection, writes CSV files and optional visualisation.

    Parameters
    ----------
    file_pattern : str
        Filename pattern with '{i}' placeholder, e.g. '{i}.bmp'.
    n_images : int
    input_dir, output_dir : str
    save_vis : bool
        Save annotated images as 'centers{i}_vis.png'.
    """
    os.makedirs(output_dir, exist_ok=True)
    ok_count = 0
    for i in range(1, n_images + 1):
        img_path = os.path.join(input_dir, file_pattern.format(i=i))
        if not os.path.exists(img_path):
            for ext in ('.png', '.bmp', '.jpg', '.tif'):
                base_name = file_pattern.format(i=i).rsplit('.', 1)[0]
                alt = os.path.join(input_dir, base_name + ext)
                if os.path.exists(alt):
                    img_path = alt
                    break
            else:
                print(f"  [SKIP] image not found: {file_pattern.format(i=i)}")
                continue

        try:
            centers = detect_circles(img_path, method=method,
                                     board_size=board_size,
                                     symmetric=symmetric,
                                     blob_params=blob_params,
                                     **adaptive_kwargs)
            ok_count += 1
            print(f"  [OK] image {i}: {centers.shape[0]} circles")
            np.savetxt(os.path.join(output_dir, f"centers{i}.csv"),
                       centers, delimiter=',', fmt='%.10f')

            if save_vis:
                img_color = imread_unicode(img_path, cv2.IMREAD_COLOR)
                for (cx, cy) in centers:
                    cv2.drawMarker(img_color, (int(cx), int(cy)),
                                   (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.imwrite(os.path.join(output_dir, f"centers{i}_vis.png"),
                            img_color)
        except Exception as e:
            print(f"  [FAIL] image {i}: {e}")

    if ok_count == 0:
        raise RuntimeError(
            f"No circle centres detected in any image under '{input_dir}'. "
            f"Check the directory path (--data_dir), image extension "
            f"(--img_ext), and whether images are present.")
    print(f"  {ok_count}/{n_images} images succeeded")
