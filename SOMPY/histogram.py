from .view import MatplotView
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import numpy as np


class Hist2d(MatplotView):

    def _fill_hist(self, x, y, mapsize, data_coords, what='train'):
        x = np.arange(.5, mapsize[1]+.5, 1)
        y = np.arange(.5, mapsize[0]+.5, 1)
        X, Y = np.meshgrid(x, y)

        if what == 'train':
            a = plt.hist2d(x, y, bins=(mapsize[1], mapsize[0]), alpha=.0, cmap=cm.jet)
            area = a[0].T*12
            plt.scatter(data_coords[:, 1], mapsize[0] - .5 - data_coords[:, 0],
                        s=area.flatten(), alpha=.9, c='None', marker='o', cmap='jet', linewidths=3, edgecolor='r')

        else:
            a = plt.hist2d(x, y, bins=(mapsize[1], mapsize[0]), alpha=.0, cmap=cm.jet, norm=LogNorm())
            area = a[0].T*50
            plt.scatter(data_coords[:, 1] + .5, mapsize[0] - .5 - data_coords[:, 0],
                        s=area, alpha=0.9, c='None', marker='o', cmap='jet', linewidths=3, edgecolor='r')
            plt.scatter(data_coords[:, 1]+.5, mapsize[0]-.5-data_coords[:, 0],
                        s=area, alpha=0.2, c='b', marker='o', cmap='jet', linewidths=3, edgecolor='r')

        plt.xlim(0, mapsize[1])
        plt.ylim(0, mapsize[0])

    def show(self, som, data=None):
        #First Step: show the hitmap of all the training data
        coord = som.bmu_ind_to_xy(som.project_data(som.data_raw))

        self.prepare()

        ax = self._fig.add_subplot(111)
        ax.xaxis.set_ticks([i for i in range(0, som.codebook.mapsize[1])])
        ax.yaxis.set_ticks([i for i in range(0, som.codebook.mapsize[0])])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.grid(True, linestyle='-', linewidth=.5)

        self._fill_hist(coord[:, 1], coord[:, 0], som.codebook.mapsize, som.bmu_ind_to_xy(np.arange(som.codebook.nnodes)))

        if data:
            coord_d = som.bmu_ind_to_xy(som.project_data(data))
            self._fill_hist(coord[:, 1], coord[:, 0], som.codebook.mapsize, coord_d, 'data')

        plt.show()

