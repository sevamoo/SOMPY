import matplotlib
from . import MatplotView
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


class MapView(MatplotView):

    def _setup_figure(self, som, which_dim):
        codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)
        dim = som._dim
        indtoshow, sV, sH = None, None, None

        if which_dim == 'all':
            indtoshow = np.arange(0, dim).T
            ratio = float(dim)/float(dim)
            ratio = np.max((.35, ratio))
            sH, sV = 16,16*ratio*1

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

        self.width = sH
        self.height = sV
        self.prepare()

        no_row_in_plot = dim/6 + 1  # 6 is arbitrarily selected
        no_col_in_plot = dim if no_row_in_plot <= 1 else 6
        axisNum = 0

        return indtoshow, no_row_in_plot, no_col_in_plot, axisNum


class View2D(MapView):

    def show(self, som, which_dim='all'):
        indtoshow, no_row_in_plot, no_col_in_plot, axisNum = self._setup_figure(som, which_dim)
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
            plt.title(som.component_names[0][ind])
            font = {'size': self.text_size*self.height/no_col_in_plot}
            plt.rc('font', **font)
            plt.axis('off')
            plt.axis([0, som.codebook.mapsize[0], 0, som.codebook.mapsize[1]])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.colorbar(pl)

        plt.show()


class View2DPacked(MapView):

    def view_2d_Pack(self, som, what='codebook', which_dim='all', grid='yes', text='yes', CMAP=None, col_sz=None):
        indtoshow, no_row_in_plot, no_col_in_plot, axisNum = self._setup_figure(som, which_dim)
        codebook = som.codebook.matrix

        no_col_in_plot = som._dim if no_row_in_plot <= 1 else col_sz
        CMAP = CMAP or plt.cm.get_cmap('RdYlBu_r')
        msz0, msz1 = som.codebook.mapsize

        if what == 'codebook':
            h = .1
            w = .1
            fig = plt.figure(figsize=(no_col_in_plot*2.5*(1+w), no_row_in_plot*2.5*(1+h)))
            DD = pd.Series(data=codebook.flatten()).describe(percentiles=[.03, .05, .1, .25, .3, .4, .5, .6, .7, .8, .9, .95, .97])
            norm = matplotlib.colors.Normalize(vmin=DD.ix['3%'], vmax=DD.ix['97%'], clip=False)

            while axisNum < som._dim:
                axisNum += 1

                ax = fig.add_subplot(no_row_in_plot, no_col_in_plot, axisNum)
                ind = int(indtoshow[axisNum-1])
                mp = codebook[:, ind].reshape(msz0, msz1)

                if grid == 'yes':
                   pl = plt.pcolor(mp[::-1], cmap=CMAP)

                elif grid == 'no':
                    plt.imshow(mp[::-1], norm=None, cmap=CMAP)
                    plt.axis('off')

                if text == 'yes':
                    plt.title(som.component_names[0][ind])
                    font = {'size': self.text_size}
                    plt.rc('font', **font)

                plt.axis([0, msz0, 0, msz1])
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.xaxis.set_ticks([i for i in range(0,msz1)])
                ax.yaxis.set_ticks([i for i in range(0,msz0)])
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
                ax.grid(True, linestyle='-', linewidth=0.5,color='k')

            plt.subplots_adjust(hspace=h, wspace=w)

        if what == 'cluster':
            if hasattr(som, 'cluster_labels'):
                codebook = som.cluster_labels

            else:
                #print 'clustering based on default parameters...'
                codebook = som.cluster()

            h = .2
            w = .001
            fig = plt.figure(figsize=(msz0/2, msz1/2))

            ax = fig.add_subplot(1, 1, 1)
            mp = codebook[:].reshape(msz0, msz1)
            if grid == 'yes':
                plt.imshow(mp[::-1], cmap=CMAP)

            elif grid == 'no':
                plt.imshow(mp[::-1], cmap=CMAP)
                plt.axis('off')

            if text == 'yes':
                plt.title('clusters')
                font = {'size': self.text_size}
                plt.rc('font', **font)

            plt.axis([0, msz0, 0, msz1])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.xaxis.set_ticks([i for i in range(0, msz1)])
            ax.yaxis.set_ticks([i for i in range(0, msz0)])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.grid(True, linestyle='-', linewidth=0.5, color='k')
            plt.subplots_adjust(hspace=h, wspace=w)


class View1D(MapView):

    def show(self, som, what='codebook', which_dim='all'):
        indtoshow, no_row_in_plot, no_col_in_plot, axisNum = self._setup_figure(som, which_dim)
        codebook = som.codebook.matrix

        while axisNum < som._dim:
            axisNum += 1
            ax = plt.subplot(no_row_in_plot, no_col_in_plot, axisNum)
            ind = int(indtoshow[axisNum-1])
            mp = codebook[:, ind]
            plt.plot(mp, '-k', linewidth=0.8)
            plt.title(som.component_names[0][ind])
            font = {'size': self.text_size*self.height/no_col_in_plot}
            plt.rc('font', **font)

        plt.show()
