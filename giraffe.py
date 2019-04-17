#!/usr/bin/env python3

import imageio
import numpy as np

from pathlib import Path


def load_image(image_directory: Path) -> np.ndarray:
    matrix = imageio.imread(image_directory)
    # Removes alpha channel if there is any
    no_alpha_matrix = matrix[:, :, :3]
    return no_alpha_matrix


def save_image(image: np.ndarray, image_directory: Path):
    imageio.imwrite(image_directory, image)


def grayscale(image: np.ndarray) -> np.ndarray:
    def f(r, g, b):
        pixel = 0.21 * r + 0.72 * g + 0.07 * b
        return [pixel] * 3

    return _transform(f, image)


def binarize(image: np.ndarray, threshold=127) -> np.ndarray:
    def f(r, g, b):
        r = 255 if r > threshold else 0
        g = 255 if g > threshold else 0
        b = 255 if b > threshold else 0
        return [r, g, b]

    return _transform(f, image)


def reverse(image: np.ndarray) -> np.ndarray:
    def f(r, g, b):
        return [255 - r, 255 - g, 255 - b]

    return _transform(f, image)


def nonlinear_filter_channel(
    channel: np.ndarray, transform, window_shape=(3, 3)
) -> np.ndarray:
    window_w, window_h = window_shape
    image_w, image_h = channel.shape
    image_padded = _pad_matrix(
        matrix=channel, width=window_w // 2, height=window_h // 2
    )
    output_image = np.zeros(channel.shape, dtype=channel.dtype)
    for i in range(image_w):
        for j in range(image_h):
            neighbors = _neighbors(i, j, *window_shape, image_padded)
            pixel = transform(neighbors)
            output_image[i, j] = pixel
    return output_image


def nonlinear_filter(image: np.ndarray, transform, window_shape=(3, 3)) -> np.ndarray:
    (red, green, blue) = split_channels(image)
    red = nonlinear_filter_channel(red, transform, window_shape=window_shape)
    green = nonlinear_filter_channel(green, transform, window_shape=window_shape)
    blue = nonlinear_filter_channel(blue, transform, window_shape=window_shape)
    return np.dstack((red, green, blue))


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
        return max(channel_histogram(neighbors.flatten()))

    return nonlinear_filter(image, mode, window_shape)


def sobel_filter(image: np.ndarray) -> np.ndarray:
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    conv_x = convolve(image, sobel_x)
    conv_y = convolve(image, sobel_y)

    r_x, g_x, b_x = split_channels(conv_x)
    r_y, g_y, b_y = split_channels(conv_y)

    r = np.sqrt(r_x * r_x + r_y * r_y)
    g = np.sqrt(g_x * g_x + g_y * g_y)
    b = np.sqrt(b_x * b_x + b_y * b_y)

    return np.dstack((r, g, b)).astype(np.uint8)


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


def histogram(image: np.ndarray):
    hists = []
    for channel in split_channels(image):
        hist = channel_histogram(channel.flatten())
        hists.append(hist)
    return hists


def channel_histogram(channel: np.ndarray):
    hg = {}
    for i in channel:
        hg[i] = hg[i] + 1 if i in hg else 1
    return hg


def split_channels(image: np.ndarray):
    r_chan, g_chan, b_chan = np.dsplit(image, image.shape[-1])
    return (r_chan[..., 0], g_chan[..., 0], b_chan[..., 0])


def rotate180(kernel: np.ndarray) -> np.ndarray:
    return np.rot90(kernel, k=2)


def _pad_matrix(matrix: np.ndarray, width, height) -> np.ndarray:
    pads = ((width, width), (height, height))
    return np.pad(matrix, pad_width=pads, mode="symmetric")


def _neighbors(i, j, w, h, matrix):
    indexes = np.ix_(range(i, i + w), range(j, j + h))
    return matrix[indexes]


def _transform(f, input_matrix: np.ndarray) -> np.ndarray:
    output_matrix = np.zeros(input_matrix.shape, dtype=input_matrix.dtype)
    for i, row in enumerate(input_matrix):
        for j, pixel in enumerate(row):
            output_matrix[i, j] = f(*pixel)
    return output_matrix
