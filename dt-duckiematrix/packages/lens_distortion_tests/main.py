"""Main function."""

import logging
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from lens_distortion_utils.constants import INF
from lens_distortion_utils.lens_map import (
    compute_lens_maps,
    compute_lens_uv_textures,
)
from matplotlib import pyplot as plt

from lens_distortion_tests.constants import camera_info_dict
from lens_distortion_tests.test_library import fov

raw_img_fpath = (
    Path(__file__).parent.resolve() / "data" / "raw_7.5cm_both_centered.png"
)
raw_img_orig = cv2.imread(raw_img_fpath)
rect_img_fpath = (
    Path(__file__).parent.resolve()
    / "data"
    / "rect_7.5cm_both_centered_in_raw.png"
)
rect_img_orig = cv2.imread(rect_img_fpath)
# set error pixel to red color
raw_img_orig[0, 0] = [255, 0, 0]
rect_img_orig[0, 0] = [255, 0, 0]
np.set_printoptions(suppress=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main() -> None:
    """Run lens distortion tests."""
    camera_matrix = np.array(
        camera_info_dict["camera_matrix"]["data"],
    ).reshape((3, 3))
    distortion_coefficients = np.array(
        camera_info_dict["distortion_coefficients"]["data"],
    )
    height = camera_info_dict["image_height"]
    width = camera_info_dict["image_width"]
    # find optimal rectified pinhole camera
    rect_camera_matrix_alpha0, roi_alpha0 = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coefficients,
        (width, height),
        0,
    )
    rect_camera_matrix_alpha1, roi_alpha1 = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coefficients,
        (width, height),
        1,
    )
    (fx, _, cx), (_, fy, cy), *_ = camera_matrix.tolist()
    logger.info(
        "raw\n\tfx:%s, fy:%s, cx:%s, cy:%s",
        int(fx),
        int(fy),
        int(cx),
        int(cy),
    )
    (fx, _, cx), (_, fy, cy), *_ = rect_camera_matrix_alpha0
    logger.info(
        "alpha: 0\n\tfx:%s, fy:%s, cx:%s, cy:%s",
        int(fx),
        int(fy),
        int(cx),
        int(cy),
    )
    hfov, vfov = np.rad2deg(fov(fx, width)), np.rad2deg(fov(fy, height))
    logger.info(
        "\thFoV: %sdeg, vFoV: %sdeg, aspect_ratio: %s",
        int(hfov),
        int(vfov),
        np.round(hfov / vfov, 2),
    )
    logger.info("\troi:%s", roi_alpha0)
    (fx, _, cx), (_, fy, cy), *_ = rect_camera_matrix_alpha1
    logger.info(
        "alpha: 1\n\tfx:%s, fy:%s, cx:%s, cy:%s",
        int(fx),
        int(fy),
        int(cx),
        int(cy),
    )
    hfov, vfov = np.rad2deg(fov(fx, width)), np.rad2deg(fov(fy, height))
    logger.info(
        "\thFoV: %sdeg, vFoV: %sdeg, aspect_ratio: %s",
        int(hfov),
        int(vfov),
        np.round(hfov / vfov, 2),
    )
    logger.info("\troi:%s", roi_alpha1)
    # create rectification map
    mapx_alpha0, mapy_alpha0 = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_coefficients,
        None,
        rect_camera_matrix_alpha0,
        (width, height),
        cv2.CV_32FC1,
    )
    mapx_alpha1, mapy_alpha1 = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_coefficients,
        None,
        rect_camera_matrix_alpha1,
        (width, height),
        cv2.CV_32FC1,
    )
    # rectify image
    rect_img_alpha0 = cv2.remap(
        raw_img_orig,
        mapx_alpha0,
        mapy_alpha0,
        cv2.INTER_NEAREST,
    )
    rect_img_alpha1 = cv2.remap(
        raw_img_orig,
        mapx_alpha1,
        mapy_alpha1,
        cv2.INTER_NEAREST,
    )
    # render
    _, axs = plt.subplots(2, 4)
    axs[0, 0].imshow(raw_img_orig)
    axs[0, 0].set_title("Raw (original)")
    axs[0, 1].imshow(rect_img_alpha0)
    axs[0, 1].set_title("Rect (alpha: 0)")
    axs[0, 2].imshow(rect_img_alpha1)
    axs[0, 2].set_title("Rect (alpha: 1)")
    inv_mapx_alpha1 = np.full_like(mapx_alpha1, INF, dtype=np.float32)
    inv_mapy_alpha1 = np.full_like(mapy_alpha1, INF, dtype=np.float32)
    for u, v in product(range(width), range(height)):
        u1, v1 = int(mapx_alpha1[v, u]), int(mapy_alpha1[v, u])
        if 0 < u1 < width and 0 < v1 < height:
            inv_mapx_alpha1[v1, u1] = u
            inv_mapy_alpha1[v1, u1] = v
    inv_mapx_sparse_alpha1 = inv_mapx_alpha1.copy()
    inv_mapy_sparse_alpha1 = inv_mapy_alpha1.copy()
    for v, u in np.argwhere(inv_mapx_sparse_alpha1 == INF):
        inv_mapx_sparse_alpha1[v, u] = 0
    for v, u in np.argwhere(inv_mapy_sparse_alpha1 == INF):
        inv_mapy_sparse_alpha1[v, u] = 0
    # set error pixel to red color
    rect_img_alpha1[0, 0] = [255, 0, 0]
    raw_img_alpha1 = cv2.remap(
        rect_img_alpha1.astype(np.float32),
        inv_mapx_sparse_alpha1,
        inv_mapy_sparse_alpha1,
        cv2.INTER_NEAREST,
    ).astype(np.uint8)
    axs[0, 3].imshow(raw_img_alpha1)
    axs[0, 3].set_title("Reconstructed Raw (sparse inverse map, alpha: 1)")
    lens_u, lens_v = compute_lens_maps(
        camera_matrix,
        rect_camera_matrix_alpha1,
        distortion_coefficients,
        width,
        height,
    )
    raw_img_filled_alpha1 = cv2.remap(
        rect_img_alpha1.astype(np.float32),
        lens_u,
        lens_v,
        cv2.INTER_NEAREST,
    ).astype(np.uint8)
    axs[1, 0].imshow(raw_img_filled_alpha1)
    axs[1, 0].set_title("Reconstructed Raw (dense inverse map, alpha: 1)")
    rect_img_filled_alpha1 = cv2.remap(
        raw_img_filled_alpha1.astype(np.float32),
        mapx_alpha1,
        mapy_alpha1,
        cv2.INTER_NEAREST,
    ).astype(np.uint8)
    axs[1, 1].imshow(rect_img_filled_alpha1)
    axs[1, 1].set_title("Reconstructed Rect (dense inverse map, alpha: 1)")
    txt_u, txt_v = compute_lens_uv_textures(
        camera_matrix,
        rect_camera_matrix_alpha1,
        distortion_coefficients,
        width,
        height,
    )
    txt_u_rgb = texture_to_rgb(txt_u, 0)
    axs[1, 2].imshow(txt_u_rgb)
    axs[1, 2].set_title("U Map (alpha: 1)")
    txt_v_rgb = texture_to_rgb(txt_v, 1)
    axs[1, 3].imshow(txt_v_rgb)
    axs[1, 3].set_title("V Map (alpha: 1)")
    # hide x labels and tick labels for top plots and y ticks for right
    # plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.subplots_adjust(0, 0, 1, 1, 0.05, 0)
    plt.show()


def texture_to_rgb(texture: np.ndarray, channel: int) -> np.ndarray:
    """Return RGB from texture."""
    empty_channel = np.zeros_like(texture, dtype=np.uint8)
    rgb = np.dstack([empty_channel, empty_channel, empty_channel])
    rgb[:, :, channel] = (texture / (np.power(2, 16) / 255)).astype(np.uint8)
    return rgb


if __name__ == "__main__":
    main()
