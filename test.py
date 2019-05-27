#!/usr/bin/env python3

import giraffe as gf
import io
import matplotlib.pyplot as plt
import numpy as np
import pathlib

def save_histogram(image: np.ndarray, path: pathlib.Path):
    assert len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 3)
    figsize = [18.5, 10.5]

    if len(image.shape) == 2:
        hist = gf.histogram(image)
        fig, ax = plt.subplots(1)
        fig.set_size_inches(*figsize)
        ax.bar(hist.keys(), hist.values(), color='black')
    else:
        r, g, b = gf.split_channels(image)
        r, g, b = gf.histogram(r), gf.histogram(g), gf.histogram(b)
        fig, (ar, ag, ab) = plt.subplots(3)
        fig.set_size_inches(*figsize)
        ar.bar(r.keys(), r.values(), color='red')
        ag.bar(g.keys(), g.values(), color='green')
        ab.bar(b.keys(), b.values(), color='blue')

    fig.savefig(path, bbox_inches='tight')
    plt.close(fig=fig)


def test():
    image = gf.load_image('lenna.png')
    r, g, b = gf.split_channels(image)
    r = gf.color_quantize(r, k=8)
    g = gf.color_quantize(g, k=8)
    b = gf.color_quantize(b, k=8)
    out = gf.merge_channels(r, g, b)
    gf.save_image(out, 'color-quantize.png')
    save_histogram(out, 'histogram.png')

def test_ascii():
    image = gf.load_image('lenna.png')
    image = gf.rgb_to_grayscale(image)
    image = gf.equalize_tone(image)
    ascii_art = gf.ascii_art(image, block_size=(8, 5))
    with io.open('lenna.txt', 'w') as f:
        gf.print_ascii_art(ascii_art, out=f)


def test_filter():
    image = gf.load_image('lenna.png')
    r, g, b = gf.split_channels(image)
    r = gf.equalize_tone(r)
    g = gf.equalize_tone(g)
    b = gf.equalize_tone(b)
    out = gf.merge_channels(r, g, b)
    gf.save_image(out, 'equalization.png')


def apply_to_channels(f, image):
    channels = []
    for channel in gf.split_channels(image):
        channels.append(f(channel))
    return gf.merge_channels(*channels)

def test_homework2():
    image = gf.load_image('lenna.png')
    rot90_l = apply_to_channels(gf.rotate90_left, image)
    rot90_r = apply_to_channels(gf.rotate90_right, image)
    mirror_h = apply_to_channels(gf.mirror_horizontal, image)
    mirror_v = apply_to_channels(gf.mirror_vertical, image)
    gf.save_image(rot90_l, 'examples/rotate90_left.png')
    gf.save_image(rot90_r, 'examples/rotate90_right.png')
    gf.save_image(mirror_h, 'examples/mirror_horizontal.png')
    gf.save_image(mirror_v, 'examples/mirror_vertical.png')


test_ascii()
