#!/usr/bin/env python3

import giraffe as gf
import numpy as np
import matplotlib.pyplot as plt
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
    image = gf.load_image('lenna_32.png')
    grayscale = gf.rgb_to_grayscale(image)
    ascii_art = gf.ascii_art(grayscale)
    ascii_art_display = "\n".join(map("".join, ascii_art))
    print(ascii_art_display)

def test_notebook():
    image = gf.load_image('notebook.jpg')
    out = gf.rgb_to_grayscale(image)
    out = gf.intensified_sharpen(out)
    out = gf.binarize(out, threshold=60)
    gf.save_image(out, 'notebook-binarized.png')

def test_mode_filter():
    image = gf.load_image('lenna.png')
    r, g, b = gf.split_channels(image)
    r = gf.mode_filter(r)
    g = gf.mode_filter(g)
    b = gf.mode_filter(b)
    out = gf.merge_channels(r, g, b)
    gf.save_image(out, 'mode-filter.png')

test_mode_filter()
