"""Lens map."""

from itertools import product
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np

from lens_distortion_utils.constants import INF, INTERPOLATION_MASK_SIZE


def _fill_pixels(
    cur: int,
    tot: int,
    width: int,
    height: int,
    lens: np.ndarray,
    neigh_mask: list[tuple[int, int]],
    indices: np.ndarray | list,
) -> np.ndarray:
    """Return filled pixels."""
    split = 1 / tot
    i = int(cur * split * len(indices))
    f = int((cur + 1) * split * len(indices))
    lensc = np.full((height, width, 2), INF, dtype=lens.dtype)
    for px in indices[i:f]:
        neighs = np.add(px, neigh_mask)
        neighs = np.maximum(neighs, [0, 0])
        neighs = np.minimum(neighs, [height - 1, width - 1])
        good_neighs = np.argwhere(
            lens[neighs[:, 0], neighs[:, 1], 0] != INF,
        ).flatten()
        neighs = neighs[good_neighs]
        if len(neighs) == 0:
            continue
        lensc[px[0], px[1]] = np.average(
            lens[neighs[:, 0], neighs[:, 1]],
            axis=0,
        )
    return lensc


def _field_of_view(focal_len: float, resolution: int) -> float:
    x = resolution / (2 * focal_len)
    square_root = np.sqrt(1 + x**2)
    return float(2 * np.arccos(1 / square_root))


def compute_field_of_view(
    camera_matrix_rect: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Return field of view."""
    (fx, _, _), (_, fy, _), *_ = camera_matrix_rect.tolist()
    return _field_of_view(fx, width), _field_of_view(fy, height)


# FIXME: [DTSW-6529] this takes a lot of time to compute with the DD24
# camera parameters, should be fixed and, ideally, vectorized
def compute_lens_maps(
    camera_matrix: np.ndarray,
    camera_matrix_rect: np.ndarray,
    distortion_parameters: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lens maps."""
    # create rectification map
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_parameters,
        None,
        camera_matrix_rect,
        (width, height),
        cv2.CV_32FC1,
    )
    inv_mapx = np.full_like(mapx, INF, dtype=np.float32)
    inv_mapy = np.full_like(mapy, INF, dtype=np.float32)
    width_range = range(width)
    height_range = range(height)
    for u, v in product(width_range, height_range):
        u1, v1 = int(mapx[v, u]), int(mapy[v, u])
        if 0 < u1 < width and 0 < v1 < height:
            inv_mapx[v1, u1] = u
            inv_mapy[v1, u1] = v
    # fill holes in lens map
    lens = np.dstack((inv_mapx, inv_mapy))
    mask_half_size = np.floor(INTERPOLATION_MASK_SIZE / 2)
    mask_half_size_integer = int(mask_half_size)
    num_threads = cpu_count()
    while True:
        # find pixels in map with no corresponding rectified pixel
        empty_pixels = np.argwhere(lens[:, :, 0] == INF)
        indices: np.ndarray | list = empty_pixels if len(empty_pixels) else []
        # there are no missing pixels, we are done here
        if len(indices) == 0:
            break
        # compute mask
        mask_range = range(
            -mask_half_size_integer,
            mask_half_size_integer + 1,
            1,
        )
        mask_range_list = list(mask_range)
        mask_range_list_product = product(mask_range_list, mask_range_list)
        neigh_mask = list(mask_range_list_product)
        # split vectors generation job into num_threads workers
        args = [
            (i, num_threads, width, height, lens, neigh_mask, indices)
            for i in range(num_threads)
        ]
        processes = cpu_count()
        with Pool(processes) as pool:
            res = pool.starmap(_fill_pixels, args)
        # combine results
        for ma in res:
            lens = np.ma.where(ma == INF, lens, ma)
        mask_half_size_integer += 1
    return lens[:, :, 0], lens[:, :, 1]


def compute_lens_uv_textures(
    camera_matrix: np.ndarray,
    camera_matrix_rect: np.ndarray,
    distortion_parameters: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lens UV textures."""
    lens_u, lens_v = compute_lens_maps(
        camera_matrix,
        camera_matrix_rect,
        distortion_parameters,
        width,
        height,
    )
    # normalize to [0, 1]
    min_u, max_u = np.min(lens_u), np.max(lens_u)
    min_v, max_v = np.min(lens_v), np.max(lens_v)
    texture_u = (lens_u - min_u) / (max_u - min_u)
    texture_v = (lens_v - min_v) / (max_v - min_v)
    # flip V axis
    texture_v = 1 - texture_v
    # expand to R16 single channel image format
    texture_u *= np.power(2, 16) - 1
    texture_v *= np.power(2, 16) - 1
    texture_u = texture_u.astype(np.uint16)
    texture_v = texture_v.astype(np.uint16)
    return texture_u, texture_v


def compute_pinhole_camera_matrix(
    camera_matrix: np.ndarray,
    distortion_parameters: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Return pinhole camera matrix."""
    # find optimal rectified pinhole camera
    rect_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_parameters,
        (width, height),
        1,
        (width, height),
    )
    return rect_camera_matrix
