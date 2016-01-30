import matplotlib
from . import MatplotView
from matplotlib import pyplot as plt
import numpy as np


class DotMapView(MatplotView):

    def init_figure(self, dim, cols):
        no_row_in_plot = dim/cols + 1
        no_col_in_plot = dim if no_row_in_plot <= 1 else cols
        h = .1
        w = .1
        self.width = no_col_in_plot*2.5*(1+w)
        self.height = no_row_in_plot*2.5*(1+h)
        self.prepare()

    def plot(self, data, coords, msz0, msz1, colormap, dlen, dim, rows, cols):
        for i in range(dim):
            plt.subplot(rows, cols, i+1)

            # This uses the colors uniquely for each record, while in normal views, it is based on the values
            # within each dimensions. This is important when we are dealing with time series. Where we don't want
            # to normalize colors within each time period, rather we like to see the patterns of each data
            # records in time.
            mn = np.min(data[:, :], axis=1)
            mx = np.max(data[:, :], axis=1)

            for j in range(dlen):
                sc = plt.scatter(coords[j, 1],
                                 msz0-1-coords[j, 0],
                                 c=data[j, i],
                                 vmax=mx[j], vmin=mn[j],
                                 s=90,
                                 marker='.',
                                 edgecolor='None',
                                 cmap=colormap,
                                 alpha=1)

            eps = .0075
            plt.xlim(0-eps, msz1-1+eps)
            plt.ylim(0-eps, msz0-1+eps)
            plt.xticks([])
            plt.yticks([])

    def show(self, som, which_dim='all', colormap=None, cols=None):
        colormap = plt.cm.get_cmap(colormap) if colormap else plt.cm.get_cmap('RdYlBu_r')

        data = som.data_raw
        msz0, msz1 = som.codebook.mapsize
        coords = som.bmu_ind_to_xy(som.project_data(data))[:, :2]
        cols = cols if cols else 8  # 8 is arbitrary
        rows = data.shape[1]/cols+1

        if which_dim == 'all':
            dim = data.shape[0]
            self.init_figure(dim, cols)
            self.plot(data, coords, msz0, msz1, colormap, data.shape[0], data.shape[1], rows, cols)

        else:
            dim = 1 if type(which_dim) is int else len(which_dim)
            self.init_figure(dim, cols)
            self.plot(data, coords, msz0, msz1, colormap, data.shape[0], len(which_dim), rows, cols)

        plt.tight_layout()
        plt.subplots_adjust(hspace=.16, swspace=.05)

