import matplotlib
from .view import MatplotView
from matplotlib import pyplot as plt
import numpy as np


class MapView(MatplotView):

    def _calculate_figure_params(self, som, which_dim, col_sz):
        codebook = som._normalizer.denormalize_by(som.data_raw,
                                                  som.codebook.matrix)

        indtoshow, sV, sH = None, None, None

        if which_dim == 'all':
            dim = som._dim
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

        no_row_in_plot = dim / col_sz + 1  # 6 is arbitrarily selected
        if no_row_in_plot <= 1:
            no_col_in_plot = dim
        else:
            no_col_in_plot = col_sz

        axis_num = 0

        width = sH
        height = sV

        return (width, height, indtoshow, no_row_in_plot, no_col_in_plot,
                axis_num)


class View2D(MapView):

    def show(self, som, what='codebook', which_dim='all', cmap=None,
             col_sz=None):
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, which_dim, col_sz)
        self.prepare()
        codebook = som.codebook.matrix

        norm = matplotlib.colors.Normalize(
            vmin=np.mean(codebook.flatten()) - 1 * np.std(codebook.flatten()),
            vmax=np.mean(codebook.flatten()) + 1 * np.std(codebook.flatten()),
            clip=True)

        while axis_num < len(indtoshow):
            axis_num += 1
            ax = plt.subplot(no_row_in_plot, no_col_in_plot, axis_num)
            ind = int(indtoshow[axis_num-1])
            mp = codebook[:, ind].reshape(som.codebook.mapsize[0],
                                          som.codebook.mapsize[1])
            pl = plt.pcolor(mp[::-1], norm=norm)
            plt.axis([0, som.codebook.mapsize[0], 0, som.codebook.mapsize[1]])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.colorbar(pl)

        plt.show()


class View2DPacked(MapView):

    def _set_axis(self, ax, msz0, msz1):
        plt.axis([0, msz0, 0, msz1])
        plt.axis('off')
        ax.axis('off')

    def show(self, som, what='codebook', which_dim='all', cmap=None,
             col_sz=None):
        if col_sz is None:
            col_sz = 6
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, which_dim, col_sz)
        codebook = som.codebook.matrix

        cmap = cmap or plt.cm.get_cmap('RdYlBu_r')
        msz0, msz1 = som.codebook.mapsize
        compname = som.component_names
        if what == 'codebook':
            h = .1
            w = .1
            self.width = no_col_in_plot*2.5*(1+w)
            self.height = no_row_in_plot*2.5*(1+h)
            self.prepare()

            while axis_num < len(indtoshow):
                axis_num += 1
                ax = self._fig.add_subplot(no_row_in_plot, no_col_in_plot,
                                           axis_num)
                ax.axis('off')
                ind = int(indtoshow[axis_num-1])
                mp = codebook[:, ind].reshape(msz0, msz1)
                plt.imshow(mp[::-1], norm=None, cmap=cmap)
                self._set_axis(ax, msz0, msz1)

                if self.show_text is True:
                    plt.title(compname[0][ind])
                    font = {'size': self.text_size}
                    plt.rc('font', **font)

        if what == 'cluster':
            try:
                codebook = getattr(som, 'cluster_labels')
            except:
                codebook = som.cluster()

            h = .2
            w = .001
            self.width = msz0/2
            self.height = msz1/2
            self.prepare()

            ax = self._fig.add_subplot(1, 1, 1)
            mp = codebook[:].reshape(msz0, msz1)
            plt.imshow(mp[::-1], cmap=cmap)

            self._set_axis(ax, msz0, msz1)

        plt.subplots_adjust(hspace=h, wspace=w)


class View1D(MapView):

    def show(self, som, what='codebook', which_dim='all', cmap=None,
             col_sz=None):
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, which_dim, col_sz)
        self.prepare()

        codebook = som.codebook.matrix

        while axis_num < len(indtoshow):
            axis_num += 1
            plt.subplot(no_row_in_plot, no_col_in_plot, axis_num)
            ind = int(indtoshow[axis_num-1])
            mp = codebook[:, ind]
            plt.plot(mp, '-k', linewidth=0.8)

        plt.show()
