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

image = gf.load_image('lenna.png')

r, g, b = gf.split_channels(image)

r = gf.color_quantize(r, k=8)
g = gf.color_quantize(g, k=8)
b = gf.color_quantize(b, k=8)

out = gf.merge_channels(r, g, b)

gf.save_image(out, 'color-quantize-8.png')

save_histogram(out, 'histogram.png')
