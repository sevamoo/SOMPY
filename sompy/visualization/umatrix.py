from .mapview import MapView
from .plot_tools import plot_hex_map
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.feature import blob_log

from warnings import warn
from math import sqrt
import numpy as np
import scipy


def rectxy_to_hexaxy(coord, X, Y):
    """Convert rectangular grid xy coordinates to hexagonal grid xy coordinates.
    Useful for plotting additional data on top of hexagonal grid.

    Args:
        coord (array): array with rectangular grid xy coordinates
        X (array): mapsize shaped array with hexagonal grid x coordinates 
        Y (array): mapsize shaped array with hexagonal grid y coordinates 

    Returns:
        [array]: array of coord's shape with hexagonal grid xy coordinates
    """
    out = np.vstack(([X[tuple(i)] for i in coord], [Y[tuple(i)] for i in coord])).T
    return out

class UMatrixView(MapView):
    def build_u_matrix(self, som, distance=1, row_normalized=False):
        UD2 = som.calculate_map_dist()
        Umatrix = np.zeros((som.codebook.nnodes, 1))
        codebook = som.codebook.matrix
        if row_normalized:
            vector = som._normalizer.normalize_by(codebook.T, codebook.T).T
        else:
            vector = codebook

        for i in range(som.codebook.nnodes):
            codebook_i = vector[i][np.newaxis, :]
            neighborbor_ind = UD2[i][0:] <= distance
            neighborbor_codebooks = vector[neighborbor_ind]
            neighborbor_dists = scipy.spatial.distance_matrix(
                codebook_i, neighborbor_codebooks)
            Umatrix[i] = np.sum(neighborbor_dists) / (neighborbor_dists.shape[1] - 1)

        return Umatrix.reshape(som.codebook.mapsize)

    def _set_contour(self, umat, ax, X=None, Y=None, hex=False):
        mn = np.min(umat.flatten())
        md = np.median(umat.flatten())
        if hex:
            ax.contour(X, Y, umat, np.linspace(mn, md, 15), 
                linewidths=0.7, cmap=plt.cm.get_cmap('Blues'))
        else:
            ax.contour(umat, np.linspace(mn, md, 15), 
                linewidths=0.7, cmap=plt.cm.get_cmap('Blues'))

    def _set_show_data(self, X, Y, ax):
        ax.scatter(X, Y, s=2, alpha=1., c='Gray',
                marker='o', cmap='jet', linewidths=3, edgecolor='Gray')

    def _set_labels(self, labels, X, Y, ax):
        for label, x, y in zip(labels, X, Y):
            ax.annotate(str(label), xy=(x, y),
                horizontalalignment='center',
                verticalalignment='center')

    def _set_blob(self, umat, coord, ax, X=None, Y=None, hex=False):
        # 'Laplacian of Gaussian'
        image = 1 / umat
        blobs = blob_log(image, max_sigma=5, num_sigma=4, threshold=.152)
        blobs[:, 2] = blobs[:, 2] * sqrt(2)
        if hex:
            blobs[:, :2] = rectxy_to_hexaxy(blobs[:, :2].astype(int), X, Y)
        else:
            blobs[:, :2] = np.flip(blobs[:, :2], axis=1)
        sel_points = list()

        for blob in blobs:
            row, col, r = blob
            c = plt.Circle((row, col), r, color='red', linewidth=2,
                        fill=False)
            ax.add_patch(c)

            dist = scipy.spatial.distance_matrix(
                coord, np.array([row, col])[np.newaxis, :])
            sel_point = dist <= r
            ax.plot(coord[:, 0][sel_point[:, 0]],
                coord[:, 1][sel_point[:, 0]], '.r')
            sel_points.append(sel_point[:, 0])
            if hex:
                ax.set_xlim([-0.5, umat.shape[1]])
                ax.set_ylim([-(((umat.shape[0] - 1) * sqrt(3)/2) + 1/sqrt(3)), 1/sqrt(3)])
            else:
                ax.set_xlim([-0.5, umat.shape[1] - 0.5])
                ax.set_ylim([umat.shape[0] - 0.5, -0.5])
            
    
    def show(self, som, distance=1, row_normalized=False, show_data=False,
            contour=False, blob=False, labels=False):
        # Setting figure parameters
        org_w = self.width
        org_h = self.height
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
            axis_num) = self._calculate_figure_params(som, 1, 1)
        self.width /= (self.width/org_w) if self.width > self.height else (self.height/org_h)
        self.height /= (self.width / org_w) if self.width > self.height else (self.height / org_h)
        self.prepare()
        plt.rc('figure', titlesize=self.text_size)
        colormap = plt.get_cmap('RdYlBu_r')

        # Setting figure data
        if som.codebook.lattice == "hexa" and distance < sqrt(3):
            warn("For hexagonal lattice, distance < sqrt(3) produces a null U-matrix.")
        umat = self.build_u_matrix(som, distance=distance, row_normalized=row_normalized)
        msz = som.codebook.mapsize
        proj = som._bmu[0]
        coord = som.bmu_ind_to_xy(proj)[:, :2]
        sel_points = list()

        if som.codebook.lattice == "rect":
            ax = self._fig.add_subplot(111)
            ax.imshow(umat, cmap=colormap, alpha=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cm.ScalarMappable(cmap=colormap), cax=cax, orientation='vertical')
            coord = np.flip(coord, axis=1)

            if contour:
                self._set_contour(umat, ax, hex=False)
            
            if blob:
                self._set_blob(umat, coord, ax, hex=False)
        elif som.codebook.lattice == "hexa":
            ax, cents = plot_hex_map(umat, colormap=colormap, fig=self._fig, colorbar=True)
            X = np.flip(np.array(cents)[:, 0].reshape(msz[0], msz[1]), axis=1)
            Y = np.flip(np.array(cents)[:, 1].reshape(msz[0], msz[1]), axis=1)
            coord = rectxy_to_hexaxy(coord, X, Y)

            if contour:
                self._set_contour(umat, ax, X, Y, hex=True)

            if blob:
                self._set_blob(umat, coord, ax, X, Y, hex=True)
        else:
            raise ValueError(
                'lattice argument of SOM object should be either "rect" or "hexa".')

        if show_data:
            self._set_show_data(coord[:, 0], coord[:, 1], ax)

        if labels:
            labels = som.build_data_labels()
            self._set_labels(labels, coord[:, 0], coord[:, 1], ax)
        
        ratio = float(msz[0])/(msz[0]+msz[1])
        self._fig.set_size_inches((1-ratio)*15, ratio*15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=.00, wspace=.000)

        plt.show()
        return sel_points, umat
