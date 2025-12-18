"""Test library."""

import logging
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import yaml
from lens_distortion_utils.lens_map import (
    compute_lens_maps,
    compute_lens_uv_textures,
    compute_pinhole_camera_matrix,
)
from matplotlib import pyplot as plt

TWO_16 = 65535

data_dir = Path(__file__).parent.resolve() / "data"
camera_desc_fpath = Path(data_dir) / "camera_desc.yaml"
grid_fpath = Path(data_dir) / "grid_10h.png"
original_img = cv2.imread(grid_fpath)
with Path(camera_desc_fpath).open() as fin:
    camera_desc = yaml.safe_load(fin)
np.set_printoptions(suppress=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def fov(focal_len: float, resolution: int) -> float:
    """Return field of view."""
    x = resolution / (2 * focal_len)
    return 2 * np.arccos(1 / np.sqrt(1 + x**2))


def main() -> None:
    """Run test library."""
    # get camera parameters
    camera_matrix = np.array(camera_desc["camera_matrix"])
    distortion_parameters = np.array(camera_desc["distortion_parameters"])
    width, height = camera_desc["width"], camera_desc["height"]
    # compute camera matrix of underlying pinhole camera
    camera_matrix_rect = compute_pinhole_camera_matrix(
        camera_matrix,
        distortion_parameters,
        width,
        height,
    )
    # create lens textures
    lens_map_u, lens_map_v = compute_lens_maps(
        camera_matrix,
        camera_matrix_rect,
        distortion_parameters,
        width,
        height,
    )
    lens_texture_u, lens_texture_v = compute_lens_uv_textures(
        camera_matrix,
        camera_matrix_rect,
        distortion_parameters,
        width,
        height,
    )
    lens_texture_u = lens_texture_u.astype(float)
    lens_texture_v = lens_texture_v.astype(float)
    # render rect
    _, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(original_img)
    axs[0, 0].set_title("Original")
    axs[1, 0].imshow(original_img)
    axs[1, 0].set_title("Original")
    # apply distortion
    raw_uv_img = np.zeros_like(original_img)
    raw_map_img = np.zeros_like(original_img)
    if (height, width) not in (lens_texture_u.shape, lens_texture_v.shape):
        raise ValueError
    # distort image (uv_texture)
    for u, v in product(range(width), range(height)):
        u1 = int(np.floor((lens_texture_u[v, u] / TWO_16) * width))
        v1 = int(np.floor((lens_texture_v[v, u] / TWO_16) * height))
        if u1 < 0 or u1 >= width:
            continue
        if v1 < 0 or v1 >= height:
            continue
        raw_uv_img[v, u] = original_img[v1 * -1, u1]
    # distort image (lens_map)
    for u, v in product(range(width), range(height)):
        u1 = int(lens_map_u[v, u])
        v1 = int(lens_map_v[v, u])
        if u1 < 0 or u1 >= width:
            continue
        if v1 < 0 or v1 >= height:
            continue
        raw_map_img[v, u] = original_img[v1, u1]
    # show distorted image
    axs[0, 1].imshow(raw_map_img)
    axs[0, 1].set_title("Distorted (Map)")
    axs[1, 1].imshow(raw_uv_img)
    axs[1, 1].set_title("Distorted (UV)")
    # undistort image
    alpha = 0
    # - find optimal rectified pinhole camera
    rect_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_parameters,
        (width, height),
        alpha,
    )
    (fx, _, cx), (_, fy, cy), *_ = camera_matrix.tolist()
    logger.info(
        "raw\n\tfx:%s, fy:%s, cx:%s, cy:%s",
        int(fx),
        int(fy),
        int(cx),
        int(cy),
    )
    (fx, _, cx), (_, fy, cy), *_ = rect_camera_matrix
    logger.info(
        "alpha: 0\n\tfx:%s, fy:%s, cx:%s, cy:%s",
        int(fx),
        int(fy),
        int(cx),
        int(cy),
    )
    hfov, vfov = np.rad2deg(fov(fx, width)), np.rad2deg(fov(fy, height))
    logger.info(
        "\thFoV: %sdeg, vFoV: %sdeg",
        int(hfov),
        int(vfov),
    )
    # create rectification map
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_parameters,
        None,
        rect_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    # rectify image
    rect_map_img = cv2.remap(raw_map_img, mapx, mapy, cv2.INTER_NEAREST)
    rect_uv_img = cv2.remap(raw_uv_img, mapx, mapy, cv2.INTER_NEAREST)
    # show rectified image
    axs[0, 2].imshow(rect_map_img)
    axs[0, 2].set_title("Rectified (Map)")
    axs[1, 2].imshow(rect_uv_img)
    axs[1, 2].set_title("Rectified (UV)")
    # hide x labels and tick labels for top plots and y ticks for right
    # plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.98, 0.05, 0)
    plt.show()


if __name__ == "__main__":
    main()
