#!/usr/bin/env python3

import giraffe as gf
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image):
    r, g, b = gf.histogram(image)
    fig, (axr, axg, axb) = plt.subplots(3)
    axr.bar(r.keys(), r.values(), color='red')
    axg.bar(g.keys(), g.values(), color='green')
    axb.bar(b.keys(), b.values(), color='blue')
    plt.show()

image = gf.load_image('lenna.png')

out = gf.mode_filter(image)

gf.save_image(out, 'out.png')

plot_histogram(out)
