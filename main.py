#!/usr/bin/env python3

import argparse
import imageio
import io
import sys
import numpy as np
import math

# passa baixa -- detecção de bordas -- sobel, prewitt, roberts

# binarização

# passa alta -- remover granulação de imagens -- moda mediana media

def grayscale(r, g, b, *a):
    pixel = 0.21 * r + 0.72 * g + 0.07 * b
    return [pixel] * 3 + [*a]


def binary(k):
    def do(r, g, b, *a):
        r, g, b, *a = grayscale(r, g, b, *a)
        pixel = 255 if r > k else 0
        return [pixel] * 3 + [*a]
    return do


def reverse(r, g, b, *a):
    return [255 - r, 255 - g, 255 - b] + [*a]


def apply_transform(f, input_matrix):
    output_matrix = input_matrix.copy()
    for i, row in enumerate(input_matrix):
        for j, pixel in enumerate(row):
            output_matrix[i, j] = f(*pixel)
    return output_matrix


def process_matrix(input_matrix):
    return apply_transform(reverse, input_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser.add_argument('-i', metavar='/path/to/input/image', dest='image_input_path')
    parser.add_argument('-o', metavar='/path/to/output/image', dest='image_output_path')

    parser_grayscale = subparsers.add_parser('grayscale')

    parser_binary = subparsers.add_parser('binary')
    parser_binary.add_argument('-k', type=int)

    args = parser.parse_args()
    input_matrix = imageio.imread(args.image_input_path)
    output_matrix = process_matrix(input_matrix)
    imageio.imwrite(args.image_output_path, output_matrix)
