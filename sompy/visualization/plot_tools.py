import math

import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.collections import RegularPolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_hex_map(d_matrix, titles=[], colormap=cm.gray, shape=[1, 1], comp_width=5, hex_shrink=1.0, fig=None,
                 colorbar=True):
    """
    Plot hexagon map where each neuron is represented by a hexagon. The hexagon
    color is given by the distance between the neurons (D-Matrix)

    Args:
    - grid: Grid dictionary (keys: centers, x, y ),
    - d_matrix: array contaning the distances between each neuron
    - w: width of the map in inches
    - title: map title

    Returns the Matplotlib SubAxis instance
    """

    d_matrix = np.flip(d_matrix, axis=0)

    def create_grid_coordinates(x, y):
        coordinates = [x for row in -1 * np.array(list(range(x))) for x in
                       list(zip(np.arange(((row) % 2) * 0.5, y + ((row) % 2) * 0.5), [0.8660254 * (row)] * y))]
        return (np.array(list(reversed(coordinates))), x, y)

    if d_matrix.ndim < 3:
        d_matrix = np.expand_dims(d_matrix, 2)

    if len(titles) != d_matrix.shape[2]:
        titles = [""] * d_matrix.shape[2]

    n_centers, x, y = create_grid_coordinates(*d_matrix.shape[:2])

    # Size of figure in inches
    if fig is None:
        xinch, yinch = comp_width * shape[1], comp_width * (x / y) * shape[0]
        fig = plt.figure(figsize=(xinch, yinch), dpi=72.)

    for comp, title in zip(range(d_matrix.shape[2]), titles):
        ax = fig.add_subplot(shape[0], shape[1], comp + 1, aspect='equal')

        # Get pixel size between two data points
        xpoints = n_centers[:, 0]
        ypoints = n_centers[:, 1]
        ax.scatter(xpoints, ypoints, s=0.0, marker='s')
        ax.axis([min(xpoints) - 1., max(xpoints) + 1.,
                 min(ypoints) - 1., max(ypoints) + 1.])
        xy_pixels = ax.transData.transform(np.vstack([xpoints, ypoints]).T)
        xpix, ypix = xy_pixels.T

        # discover radius and hexagon
        apothem = hex_shrink * (xpix[1] - xpix[0]) / math.sqrt(3)
        area_inner_circle = math.pi * (apothem ** 2)
        dm = d_matrix[:, :, comp].reshape(np.multiply(*d_matrix.shape[:2]))
        collection_bg = RegularPolyCollection(
            numsides=6,  # a hexagon
            rotation=0,
            sizes=(area_inner_circle,),
            array=dm,
            cmap=colormap,
            offsets=n_centers,
            transOffset=ax.transData,
        )
        ax.add_collection(collection_bg, autolim=True)

        ax.axis('off')
        ax.autoscale_view()
        ax.set_title(title)#, fontdict={"fontsize": 3 * comp_width})
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(collection_bg, cax=cax)
        if not colorbar:
            cbar.remove()

        #cbar.ax.tick_params(labelsize=3 * comp_width)

    return ax, list(reversed(n_centers))