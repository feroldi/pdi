#!/usr/bin/env python3

import imageio
import numpy as np

from pathlib import Path


def load_image(image_directory: Path) -> np.ndarray:
    return imageio.imread(image_directory)


def save_image(image: np.ndarray, image_directory: Path):
    imageio.imwrite(image_directory, image)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    def f(pixel):
        r, g, b = pixel
        pixel = 0.21 * r + 0.72 * g + 0.07 * b
        return pixel

    return _transform(f, image, output_shape=image.shape[:2])


def binarize(image: np.ndarray, threshold=127) -> np.ndarray:
    def f(pixel):
        return 255 if pixel > threshold else 0

    return _transform(f, image)


def reverse(image: np.ndarray) -> np.ndarray:
    def f(pixel):
        return 255 - pixel

    return _transform(f, image)


def nonlinear_filter(image: np.ndarray, transform, window_shape=(3, 3)) -> np.ndarray:
    window_w, window_h = window_shape
    image_w, image_h = image.shape
    image_padded = _pad_matrix(matrix=image, width=window_w // 2, height=window_h // 2)
    output_image = np.zeros(image.shape, dtype=image.dtype)
    for i in range(image_w):
        for j in range(image_h):
            neighbors = _neighbors(i, j, *window_shape, image_padded)
            pixel = transform(neighbors)
            output_image[i, j] = pixel
    return output_image


def correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    normal_kernel = normalize_kernel(kernel)

    def apply_kernel(neighbors):
        pixel = np.sum(neighbors * normal_kernel)
        return np.clip(pixel, 0, 255)

    return nonlinear_filter(image, apply_kernel, window_shape=normal_kernel.shape)


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return correlate(image, kernel[::-1, ::-1])


def mean_filter(image: np.ndarray, window_shape=(3, 3)) -> np.ndarray:
    def mean(neighbors):
        return sum(neighbors) // len(neighbors)

    return nonlinear_filter(image, mean, window_shape)


def median_filter(image: np.ndarray, window_shape=(3, 3)) -> np.ndarray:
    def median(neighbors):
        return sorted(neighbors.flatten())[(len(neighbors) - 1) // 2]

    return nonlinear_filter(image, median, window_shape)


def mode_filter(image: np.ndarray, window_shape=(3, 3)) -> np.ndarray:
    def mode(neighbors):
        return max(histogram(neighbors.flatten()))

    return nonlinear_filter(image, mode, window_shape)


def k_order(image: np.ndarray, k, window_shape=(3, 3)) -> np.ndarray:
    assert k < (window_shape[0] * window_shape[1])

    def k_order(neighbors):
        return sorted(neighbors.flatten())[k]

    return nonlinear_filter(image, k_order, window_shape)


def sobel_filter(image: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = convolve(image, kernel_x)
    gy = convolve(image, kernel_y)
    g = np.hypot(gx, gy)
    g *= 255 / np.max(g)
    return g.astype(np.uint8)


def prewitt_filter(image: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    gx = convolve(image, kernel_x)
    gy = convolve(image, kernel_y)
    g = np.hypot(gx, gy)
    g *= 255 / np.max(g)
    return g.astype(np.uint8)


def roberts_filter(image: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    gx = convolve(image, kernel_x)
    gy = convolve(image, kernel_y)
    g = np.hypot(gx, gy)
    g *= 255 / np.max(g)
    return g.astype(np.uint8)


def smoothen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(3, 3)
    return convolve(image, kernel)


def gaussian_blur_smoothen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([0, 1, 0, 1, 4, 1, 0, 1, 0]).reshape(3, 3)
    return convolve(image, kernel)


def laplacian_edge_detection(image: np.ndarray) -> np.ndarray:
    kernel = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape(3, 3)
    return convolve(image, kernel)


def sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape(3, 3)
    return convolve(image, kernel)


def intensified_sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([-1, -1, -1, -1, 9, -1, -1, -1, -1]).reshape(3, 3)
    return convolve(image, kernel)


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel_sum = np.sum(kernel)
    divider = kernel_sum if kernel_sum != 0 else 1
    return kernel / divider


def color_quantize(image: np.ndarray, k) -> np.ndarray:
    assert k >= 2 and k <= 128
    divider = 256 // k

    def f(pixel):
        pixel += divider - (pixel % divider) - 1
        assert pixel >= 0 and pixel <= 255
        return pixel

    return _transform(f, image)


def histogram(image: np.ndarray):
    assert len(image.shape) == 2
    image1d = image.flatten()
    hg = {}
    for pixel in image1d:
        hg[pixel] = hg[pixel] + 1 if pixel in hg else 1
    return hg


def split_channels(image: np.ndarray):
    r_chan, g_chan, b_chan = np.dsplit(image, image.shape[-1])
    return (r_chan[..., 0], g_chan[..., 0], b_chan[..., 0])


def merge_channels(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    return np.dstack((red, green, blue))


def rotate180(kernel: np.ndarray) -> np.ndarray:
    return np.rot90(kernel, k=2)


def _pad_matrix(matrix: np.ndarray, width, height) -> np.ndarray:
    pads = ((width, width), (height, height))
    return np.pad(matrix, pad_width=pads, mode="symmetric")


def _neighbors(i, j, w, h, matrix):
    indexes = np.ix_(range(i, i + w), range(j, j + h))
    return matrix[indexes]


def _transform(f, input_matrix: np.ndarray, output_shape=None) -> np.ndarray:
    if not output_shape:
        output_shape = input_matrix.shape
    output_matrix = np.zeros(output_shape, dtype=input_matrix.dtype)
    for i, row in enumerate(input_matrix):
        for j, pixel in enumerate(row):
            output_matrix[i, j] = f(pixel)
    return output_matrix
