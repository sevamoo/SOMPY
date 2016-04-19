import matplotlib
from .view import MatplotView
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


class MapView(MatplotView):

    def _calculate_figure_params(self, som, which_dim):
        codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)
        dim = som._dim
        indtoshow, sV, sH = None, None, None

        if which_dim == 'all':
            indtoshow = np.arange(0, dim).T
            ratio = float(dim)/float(dim)
            ratio = np.max((.35, ratio))
            sH, sV = 16, 16*ratio*1

        elif type(which_dim) == int:
            dim = 1
            indtoshow = np.zeros(1)
            indtoshow[0] = int(which_dim)
            sH, sV = 6, 6

        elif type(which_dim) == list:
            max_dim = codebook.shape[1]
            dim = len(which_dim)
            ratio = float(dim)/float(max_dim)
            ratio = np.max((.35, ratio))
            indtoshow = np.asarray(which_dim).T
            sH, sV = 16, 16*ratio*1

        no_row_in_plot = dim/6 + 1  # 6 is arbitrarily selected
        no_col_in_plot = dim if no_row_in_plot <= 1 else 6
        axisNum = 0

        width = sH
        height = sV

        return width, height, indtoshow, no_row_in_plot, no_col_in_plot, axisNum


class View2D(MapView):

    def show(self, som, which_dim='all'):
        self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot, axisNum = self._calculate_figure_params(som, which_dim)
        self.prepare()
        codebook = som.codebook.matrix

        norm = matplotlib.colors.normalize(vmin=np.mean(codebook.flatten())-1*np.std(codebook.flatten()),
                                           vmax=np.mean(codebook.flatten())+1*np.std(codebook.flatten()),
                                           clip=True)
        while axisNum < som._dim:
            axisNum += 1
            ax = plt.subplot(no_row_in_plot, no_col_in_plot, axisNum)
            ind = int(indtoshow[axisNum-1])
            mp = codebook[:, ind].reshape(som.codebook.mapsize[0], som.codebook.mapsize[1])
            pl = plt.pcolor(mp[::-1], norm=norm)
            plt.axis([0, som.codebook.mapsize[0], 0, som.codebook.mapsize[1]])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.colorbar(pl)

        plt.show()


class View2DPacked(MapView):

    def _set_axis(self, ax, msz0, msz1):
        plt.axis([0, msz0, 0, msz1])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.xaxis.set_ticks([i for i in range(0, msz1)])
        ax.yaxis.set_ticks([i for i in range(0, msz0)])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.grid(True, linestyle='-', linewidth=0.5, color='k')

    def show(self, som, what='codebook', which_dim='all', CMAP=None, col_sz=None):
        self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot, axisNum = self._calculate_figure_params(som, which_dim)
        codebook = som.codebook.matrix

        no_col_in_plot = som._dim if no_row_in_plot <= 1 else col_sz or no_col_in_plot
        CMAP = CMAP or plt.cm.get_cmap('RdYlBu_r')
        msz0, msz1 = som.codebook.mapsize

        if what == 'codebook':
            h = .1
            w = .1
            self.width = no_col_in_plot*2.5*(1+w)
            self.height = no_row_in_plot*2.5*(1+h)
            self.prepare()

            while axisNum < som._dim:
                axisNum += 1

                ax = self._fig.add_subplot(no_row_in_plot, no_col_in_plot, axisNum)
                ind = int(indtoshow[axisNum-1])
                mp = codebook[:, ind].reshape(msz0, msz1)
                plt.imshow(mp[::-1], norm=None, cmap=CMAP)

                self._set_axis(ax, msz0, msz1)

        if what == 'cluster':
            codebook = som.cluster_labels if hasattr(som, 'cluster_labels') else som.cluster()

            h = .2
            w = .001
            self.width = msz0/2
            self.height = msz1/2
            self.prepare()

            ax = self._fig.add_subplot(1, 1, 1)
            mp = codebook[:].reshape(msz0, msz1)
            plt.imshow(mp[::-1], cmap=CMAP)

            self._set_axis(ax, msz0, msz1)

        plt.subplots_adjust(hspace=h, wspace=w)


class View1D(MapView):

    def show(self, som, what='codebook', which_dim='all'):
        self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot, axisNum = self._calculate_figure_params(som, which_dim)
        self.prepare()

        codebook = som.codebook.matrix

        while axisNum < som._dim:
            axisNum += 1
            ax = plt.subplot(no_row_in_plot, no_col_in_plot, axisNum)
            ind = int(indtoshow[axisNum-1])
            mp = codebook[:, ind]
            plt.plot(mp, '-k', linewidth=0.8)

        plt.show()
