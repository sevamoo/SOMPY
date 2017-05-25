from .view import MatplotView
from matplotlib import pyplot as plt
import numpy as np


class HitMapView(MatplotView):

    def _set_labels(self, cents, ax, labels):
        for i, txt in enumerate(labels):
            ax.annotate(txt, (cents[i, 1], cents[i, 0]), size=10, va="center")

    def show(self, som, data=None):

        try:
            codebook = getattr(som, 'cluster_labels')
        except:
            codebook = som.cluster()

        # codebook = getattr(som, 'cluster_labels', som.cluster())
        msz = som.codebook.mapsize

        self.prepare()
        ax = self._fig.add_subplot(111)

        if data:
            proj = som.project_data(data)
            cents = som.bmu_ind_to_xy(proj)
            self._set_labels(cents, ax, codebook[proj])

        else:
            cents = som.bmu_ind_to_xy(np.arange(0, msz[0]*msz[1]))
            self._set_labels(cents, ax, codebook)

        plt.imshow(codebook.reshape(msz[0], msz[1])[::], alpha=.5)
        plt.show()

        return cents
