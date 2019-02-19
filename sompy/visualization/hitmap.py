import numpy as np
from matplotlib import pyplot as plt

from .mapview import MapView
from .plot_tools import plot_hex_map


class HitMapView(MapView):

    def _set_labels(self, cents, ax, labels, onlyzeros, fontsize, hex=False):
        for i, txt in enumerate(labels):
            if onlyzeros == True:
                if txt > 0:
                    txt = ""
            c = cents[i] if hex else (cents[i, 1] + 0.5, cents[-(i + 1), 0] + 0.5)
            ax.annotate(txt, c, va="center", ha="center", size=fontsize)

    def show(self, som, data=None, anotate=True, onlyzeros=False, labelsize=7, cmap="jet"):
        org_w = self.width
        org_h = self.height
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, 1, 1)
        self.width /= (self.width / org_w) if self.width > self.height else (self.height / org_h)
        self.height /= (self.width / org_w) if self.width > self.height else (self.height / org_h)
        try:
            clusters = getattr(som, 'cluster_labels')
        except:
            clusters = som.cluster()

        # codebook = getattr(som, 'cluster_labels', som.cluster())
        msz = som.codebook.mapsize

        self.prepare()
        if som.codebook.lattice == "rect":
            ax = self._fig.add_subplot(111)

            if data:
                proj = som.project_data(data)
                cents = som.bmu_ind_to_xy(proj)
                if anotate:
                    # TODO: Fix position of the labels
                    self._set_labels(cents, ax, clusters[proj], onlyzeros, labelsize, hex=False)

            else:
                cents = som.bmu_ind_to_xy(np.arange(0, msz[0] * msz[1]))
                if anotate:
                    # TODO: Fix position of the labels
                    self._set_labels(cents, ax, clusters, onlyzeros, labelsize, hex=False)

            plt.imshow(clusters.reshape(msz[0], msz[1])[::], alpha=.5)

        elif som.codebook.lattice == "hexa":
            ax, cents = plot_hex_map(np.flip(clusters.reshape(msz[0], msz[1])[::], axis=1),
                                     fig=self._fig, colormap=cmap, colorbar=False)
            if anotate:
                self._set_labels(cents, ax, clusters, onlyzeros, labelsize, hex=True)
