import colorsys

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, colors, colorbar
from matplotlib.colors import LinearSegmentedColormap

from typing import List


def get_marker():
    """
    Returns
    -------
    A list with all available different marker types in the matplotlib. E.g.: '.', ',', 'o', 'v', '^', '<', '>',
    """
    from matplotlib.lines import Line2D
    marker = list(Line2D.markers.keys())
    return marker


def get_maximal_different_colors(num_colors: int):
    """

    Parameters
    ----------
    num_colors

    Returns
    -------
    list of colors
    """
    assert num_colors < 13, "we only have 12 colors here."
    colors  = [(166, 206, 227), (31, 120, 180), (178, 223, 138), (51, 160, 44), (251, 154, 153),
               (227, 26, 28), (253, 191, 111), (255, 127, 0), (202, 178, 214), (106, 61, 154),
               (255, 255, 153), (177, 89, 40)]
    colors = [tuple(v / 255 for v in c) for c in colors]
    return colors[:num_colors]


def get_diverging_colors(num_colors: int):
    """
    Get a color map containing with ``num_color`` diverging colors

    Parameters
    ----------
    num_colors : int

    Returns
    -------
    color map
    """
    return sns.color_palette('RdBu', num_colors)


def get_evenly_spaced_colors(num_colors: int):
    """
    Get a color map containing with ``num_color`` colors in a ordered fashion.

    Parameters
    ----------
    num_colors : int

    Returns
    -------
    color map
    """
    return sns.hls_palette(num_colors, l=.5, s=.8)


def get_cmap(n, name='hsv'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    """
    return plt.cm.get_cmap(name, n)


def get_rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


def cool_13_colors() -> List[str]:
    """
    This is a set of 13 cool colors.

    Returns
    -------
    List[str]
    """
    return ["#E9C46A", "#F4A261", "#4C2505", "#E73B0A", "#2A9D8F", "#9EE5DD", "#264653", "#8FBBCC", "#d7191c",
            "#fdae61", "#ffffbf", "#abdda4", "#2b83ba"]
