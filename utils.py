"""Utilities: CSV I/O and Unicode-safe image loading for Windows."""

import numpy as np
import cv2


def read_csv(filename, delimiter=','):
    return np.loadtxt(filename, delimiter=delimiter, ndmin=2)


def write_csv(filename, data, fmt='%.10f'):
    np.savetxt(filename, data, delimiter=',', fmt=fmt)


def write_csv_int(filename, data):
    np.savetxt(filename, data, delimiter=',', fmt='%d')


def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    """Read an image from a path that may contain non-ASCII characters.

    OpenCV's ``cv2.imread`` does not support Unicode paths on Windows.
    This function reads the file via numpy and decodes it with OpenCV.
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img
