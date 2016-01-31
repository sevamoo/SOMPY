import matplotlib
from .view import MatplotView
from matplotlib import pyplot as plt
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show
from math import sqrt
import numpy as np
import scipy


class UMatrixView(MatplotView):

    def build_u_matrix(self, som, distance=1, row_normalized=True):
        UD2 = som.calculate_map_dist()
        Umatrix = np.zeros((som.codebook.nnodes, 1))
        codebook = som.codebook.matrix
        vector = som._normalizer.normalize_by(codebook.T, codebook.T, method='var').T if row_normalized else codebook

        for i in range(som.codebook.nnodes):
            codebook_i = vector[i][np.newaxis, :]
            neighborbor_ind = UD2[i][0:] <= distance
            neighborbor_codebooks = vector[neighborbor_ind]
            Umatrix[i] = scipy.spatial.distance_matrix(codebook_i, neighborbor_codebooks).mean()

        return Umatrix.reshape(som.codebook.mapsize)

    def show(self, som, distance2=1, row_normalized=False, show_data=True, contooor=True, blob=False):
        umat = self.build_u_matrix(som, distance=distance2, row_normalized=row_normalized)
        msz = som.codebook.mapsize
        proj = som.project_data(som.data_raw)
        coord = som.bmu_ind_to_xy(proj)

        fig, ax = plt.subplots(1, 1)
        im = imshow(umat, cmap=plt.cm.get_cmap('RdYlBu_r'), alpha=1)

        if contooor:
            mn = np.min(umat.flatten())
            mx = np.max(umat.flatten())
            std = np.std(umat.flatten())
            md = np.median(umat.flatten())
            mx = md + 0*std
            cset = contour(umat, np.linspace(mn, mx, 15), linewidths=0.7, cmap=plt.cm.get_cmap('Blues'))

        if show_data:
            plt.scatter(coord[:, 1], coord[:, 0], s=2, alpha=1., c='Gray', marker='o', cmap='jet', linewidths=3, edgecolor='Gray')
            plt.axis('off')

        ratio = float(msz[0])/(msz[0]+msz[1])
        fig.set_size_inches((1-ratio)*15, ratio*15)
        plt.tight_layout()
        plt.subplots_adjust(hspace=.00, wspace=.000)
        sel_points = list()

        if blob:
            from skimage.color import rgb2gray
            from skimage.feature import blob_log

            image = 1/umat
            image_gray = rgb2gray(image)

            #'Laplacian of Gaussian'
            blobs = blob_log(image, max_sigma=5, num_sigma=4, threshold=.152)
            blobs[:, 2] = blobs[:, 2] * sqrt(2)
            imshow(umat, cmap=plt.cm.get_cmap('RdYlBu_r'), alpha=1)
            sel_points = list()

            for blob in blobs:
                row, col, r = blob
                c = plt.Circle((col, row), r, color='red', linewidth=2, fill=False)
                ax.add_patch(c)
                dist = scipy.spatial.distance_matrix(coord[:, :2], np.array([row, col])[np.newaxis, :])
                sel_point = dist <= r
                plt.plot(coord[:, 1][sel_point[:, 0]], coord[:, 0][sel_point[:, 0]], '.r')
                sel_points.append(sel_point[:, 0])

        plt.show()
        return sel_points, umat

