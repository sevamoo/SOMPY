# -*- coding: utf-8 -*-


# Author: Vahid Moosavi (sevamoo@gmail.com)
#         Chair For Computer Aided Architectural Design, ETH  Zurich
#         Future Cities Lab
#         www.vahidmoosavi.com

# Contributor: Sebastian Packmann (sebastian.packmann@gmail.com)


import tempfile
import os
import itertools
import timeit
import sys

import numpy as np
import numexpr as ne
import scipy.spatial as spdist
import pandas as pd

from time import time
from scipy.sparse import csr_matrix
from sklearn import neighborbors
from sklearn.externals.joblib import Parallel, delayed, load, dump
from sklearn.decomposition import RandomizedPCA, PCA

from codebook import Codebook


class SOM(object):

    def __init__(self,
                 data,
                 mapsize=None,
                 mask=None,
                 mapshape='planar',
                 lattice='rect',
                 normalization='var',
                 initialization='pca',
                 neighbor='gaussian',
                 name='sompy'):

        # available mapshapes ['planar','toroid','cylinder']
        # available lattices ['hexa','rect']
        # available normalizations ['var']
        # available initializations ['pca', 'random']
        # available neighborhood ['Gaussian','manhattan','bubble','cut_gaussian','epanechicov']
        # available algorithms = ['seq','batch']
        # available alfa_types = ['linear','inv','power']

        self.name = name
        self.data_raw = data
        self.neighbor = neighbor
        self.mapshape = mapshape
        self.mask = mask or np.ones([1, self._dim])
        self.initialization = initialization

        self._data = normalize(data, method=normalization) if normalization == 'var' else data
        self._dim = data.shape[1]
        self._dlen = data.shape[0]

        self._dlabel = None
        self._codebook = None
        self._bmu = None

        # TODO: These 3 are not used anywhere
        self.algtype = 'batch'
        self.alfaini = 'inv'
        self.alfafinal = .005

        self.codebook = Codebook(mapsize, lattice)

        self._component_names = self.build_component_names()
        self._distance_matrix = self.calculate_map_dist()

        #self.set_data_labels()  # slow for large data sets

    @property
    def component_names(self):
        return self._component_names

    @component_names.setter
    def component_names(self, compnames):
        try:
            if self._dim == len(compnames):
                self._component_names = np.asarray(compnames)[np.newaxis, :]
            else:
                print 'compname should have the same size'
        except:
            pass
            print 'no data yet: please first set training data to the SOM'

    def build_component_names(self):
        component_names = None

        try:
            cc = ['Variable-' + str(i+1) for i in range(0, self._dim)]
            component_names = np.asarray(cc)[np.newaxis, :]
        except:
            pass
            print 'no data yet: please first set training data to the SOM'

        return component_names


    @property
    def data_labels(self):
        return self._dlabel

    @data_labels.setter
    def data_labels(self, labels):
        """
        Set labels of the training data, it should be in the format of a list of strings
        """
        try:
            if labels.shape == (1, self._dlen):
                label = labels.T
            elif labels.shape == (self._dlen, 1):
                label = labels
            elif labels.shape == (self._dlen,):
                label = labels[:, np.newaxis]
            else:
                print 'wrong label format'
                label = None

            self._dlabel = label

        except:
            pass
            print 'no data yet: please first set training data to the SOM'

    def build_data_labels(self):
        dlabels = None
        try:
            cc = ['dlabel-' + str(i) for i in range(0, self._dlen)]
            dlabels = np.asarray(cc)[:, np.newaxis]
        except:
            pass
            print 'no data yet: please first set training data to the SOM'

        return dlabels

    def calculate_map_dist(self):
        """
        Calculates the grid distance, which will be used during the training steps.
        It supports only planar grids for the moment
        """
        nnodes = self.codebook.nnodes

        bmus_distance_matrix = np.zeros((nnodes, nnodes))
        for i in range(nnodes):
            bmus_distance_matrix[i] = self.codebook.grid_dist(i).reshape(1, nnodes)

        return bmus_distance_matrix

    def train(self, n_job=1, shared_memory='no', verbose='on'):
        t0 = time()

        #print 'data len is %d and data dimension is %d' % (self._dlen, self._dim)
        #print 'map size is %d, %d' %(self._mapsize[0], self._mapsize[1])
        #print 'array size in log10 scale' , np.log10(self._dlen*self._nnodes*self._dim)
        #print 'nomber of jobs in parallel: ', n_job 
        #######################################
        #initialization
        if verbose == 'on':
            print 
            print 'initialization method = %s, initializing..' % self.initialization
            print
            t0 = time()

        if self.initialization == 'random':
            self.codebook.random_initialization(self._data)

        elif self.initialization == 'pca':
            self.codebook.pca_linear_initialization(self._data)

        if verbose == 'on':
            print 'initialization done in %f seconds' % round(time()-t0, 3)

        self.rough_train(njob=n_job, shared_memory=shared_memory, verbose=verbose)

        self.finetune_train(njob=n_job, shared_memory=shared_memory, verbose=verbose)

        err = np.mean(self._bmu)[1]

        if verbose == 'on':
            ts = round(time() - t0, 3)
            print
            print "Total time elapsed: %f seconds" % ts
            print "final quantization error: %f" % err

        if verbose == 'final':
            ts = round(time() - t0, 3)
            print
            print "Total time elapsed: %f seconds" % ts
            print "final quantization error: %f" % err

    def _calculate_ms_and_mpd(self):
        mn = np.min(self.codebook.mapsize)
        max_s = max(self.codebook.mapsize[0], self.codebook.mapsize[1])

        mpd = float(self.codebook.nnodes*10)/float(self._dlen) if mn == 1 else float(self.codebook.nnodes)/float(self._dlen)
        ms = max_s/2.0 if mn == 1 else max_s

        return ms, mpd

    def rough_train(self, njob=1, shared_memory='no', verbose='on'):
        if verbose == 'on':
            print 'rough training...'

        ms, mpd = self._calculate_ms_and_mpd()

        trainlen, radiusin, radiusfin = int(np.ceil(30*mpd)), None, None

        if self.initialization == 'random':
            radiusin = max(1, np.ceil(ms/3.))
            radiusfin = max(1, radiusin/6.)

        elif self.initialization == 'pca':
            radiusin = max(1, np.ceil(ms/8.))
            radiusfin = max(1, radiusin/4.)

        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory, verbose)

    def finetune_train(self, njob=1, shared_memory='no', verbose='on'):
        if verbose == 'on':
            print 'finetune training...'

        ms, mpd = self._calculate_ms_and_mpd()

        trainlen, radiusin, radiusfin = None, None, None

        if self.initialization == 'random':
            trainlen = int(np.ceil(50*mpd))
            radiusin = max(1, ms/12.)  # from radius fin in rough training
            radiusfin = max(1, radiusin/25.)
            #radiusin = max(1, ms/2.)  # from radius fin in rough training
            #radiusfin = max(1, radiusin/2.)

        elif self.initialization == 'pca':
            trainlen = int(np.ceil(40*mpd))
            radiusin = max(1, np.ceil(ms/8.)/4)
            radiusfin = 1  # max(1, ms/128)

        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory, verbose)

    def _batchtrain(self, trainlen, radiusin, radiusfin, njob=1, shared_memory='no', verbose='on'):
        t0 = time()

        radius = np.linspace(radiusin, radiusfin, trainlen)

        if shared_memory == 'yes':
            data = self._data
            data_folder = tempfile.mkdtemp()
            data_name = os.path.join(data_folder, 'data')
            dump(data, data_name)
            data = load(data_name, mmap_mode='r')

        else:
            data = self._data

        bmu = None
        neighborhood = None
        # X2 is part of euclidean distance (x-y)^2 = x^2 +y^2 - 2xy that we use for each data row in bmu finding.
        # Since it is a fixed value we can skip it during bmu finding for each data point,
        # but later we need it calculate quantification error
        fixed_euclidean_x2 = np.einsum('ij,ij->i', data, data)

        if verbose == 'on':
            print 'radius_ini: %f , radius_final: %f, trainlen: %d' % (radiusin, radiusfin, trainlen)

        for i in range(trainlen):
            if self.neighbor == 'gaussian':
                neighborhood = np.exp(-1.0*self._distance_matrix/(2.0*radius[i]**2)).reshape(self._nnodes, self._nnodes)

            if self.neighbor == 'bubble':
                neighborhood = l(radius[i], np.sqrt(self._distance_matrix.flatten())).reshape(self._nnodes, self._nnodes) + .000000000001

            t1 = time()
            bmu = self.find_bmu(data, njb=njob)

            t2 = time()
            self._codebook = self.update_codebook_voronoi(data, bmu, neighborhood)
            #print 'updating nodes: ', round (time()- t2, 3)

            if verbose == 'on':
                qerror = (i+1, round(time() - t1, 3), np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2)))
                print "epoch: %d ---> elapsed time:  %f, quantization error: %f " % qerror

        bmu[1] = np.sqrt(bmu[1] + fixed_euclidean_x2)

        self._bmu = bmu

    def find_bmu(self, input_matrix, njb=1):
        """
        Finds the best matching unit (bmu) for each input data from the input matrix. It does all at once parallelizing
        the calculation instead of going through each input and running it against the codebook.

        @param input_matrix numpy matrix representing inputs as rows and features/dimention as cols
        @param njob number of jobs to parallelize the search
        """
        dlen = input_matrix.shape[0]
        y2 = np.einsum('ij,ij->i', self._codebook, self._codebook)

        t_temp = time()

        parallelizer = Parallel(n_jobs=njb, pre_dispatch='3*n_jobs')
        chunk_bmu_finder = delayed(self._chunk_based_bmu_find)

        row_chunk = lambda part: part * dlen // njb
        col_chunk = lambda part: min((part+1)*dlen // njb, dlen)

        b = parallelizer(chunk_bmu_finder(input_matrix[row_chunk(i):col_chunk(i)], self._codebook, y2) for i in xrange(njb))

        #print 'bmu finding: %f seconds ' %round(time() - t_temp, 3)
        t1 = time()
        bmu = np.asarray(list(itertools.chain(*b))).T
        #print 'bmu to array: %f seconds' %round(time() - t1, 3)
        del b
        return bmu

    @staticmethod
    def _chunk_based_bmu_find(input_matrix, codebook, y2):
        """
        Finds the corresponding bmus to the input matrix.

        @param input_matrix a matrix of input data, representing input vector as rows, and vectors features/dimention as cols
                            when parallelizing the search, the input_matrix can be a sub matrix from the bigger matrix
        @param codebook matrix of weights to be used for the bmu search
        @param y2 <not sure>
        """
        dlen = input_matrix.shape[0]
        nnodes = codebook.shape[0]
        bmu = np.empty((dlen, 2))

        # It seems that small batches for large dlen is really faster:
        # that is because of ddata in loops and n_jobs. for large data it slows down due to memory needs in parallel
        blen = min(50, dlen)
        i0 = 0

        while i0+1 <= dlen:
            low = i0
            high = min(dlen, i0+blen)
            i0 = i0+blen
            ddata = input_matrix[low:high+1]
            d = np.dot(codebook, ddata.T)
            d *= -2
            d += y2.reshape(nnodes, 1)
            bmu[low:high+1, 0] = np.argmin(d, axis=0)
            bmu[low:high+1, 1] = np.min(d, axis=0)
            del ddata

        return bmu

    def update_codebook_voronoi(self, training_data, bmu, neighborhood):
        """
        Updates the weights of each node in the codebook that belongs to the bmu's neighborhood.

        First finds the Voronoi set of each node. It needs to calculate a smaller matrix.
        Super fast comparing to classic batch training algorithm, it is based on the implemented algorithm in
        som toolbox for Matlab by Helsinky university

        @param training_data input matrix with input vectors as rows and vector features as cols
        @param bmu best matching unit for each input data
        @param neighborhood matrix representing the neighborhood of each bmu

        @return An updated codebook that incorporates the learnings from the input data
        """
        # bmu has shape of 2, dlen, Where first row has bmu indexes
        # we construct ud2 from precomputed UD2 : ud2 = UD2[bmu[0,:]]
        # TODO: Comment above, ud2 is not used here

        #fig = plt.hist(bmu[0],bins=100)
        #plt.show()

        inds = bmu[0].astype(int)
        row = inds
        col = np.arange(self._dlen)
        val = np.tile(1, self._dlen)
        P = csr_matrix((val, (row, col)), shape=(self._nnodes, self._dlen))
        S = P.dot(training_data)

        # neighborhood has nnodes*nnodes and S has nnodes*dim  ---> Nominator has nnodes*dim
        nom = neighborhood.T.dot(S)
        nV = P.sum(axis=1).reshape(1, self._nnodes)
        denom = nV.dot(neighborhood.T).reshape(self._nnodes, 1)
        new_codebook = np.divide(nom, denom)

        return np.around(new_codebook, decimals=6)

    def project_data(self, data):
        """
        Projects a data set to a trained SOM. It is based on nearest neighborhood search module of scikitlearn,
        but it is not that fast.
        """
        clf = neighborbors.KNeighborsClassifier(n_neighborbors=1)
        labels = np.arange(0, self._codebook.shape[0])
        clf.fit(self._codebook, labels)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of original data
        data = normalize_by(self.data_raw, data, method='var')
        #data = normalize(data, method='var')
        #plt.hist(data[:,2])

        return clf.predict(data)

    def predict_by(self, data, target, k=5, wt='distance'):
        # here it is assumed that target is the last column in the codebook
        # and data has dim-1 columns
        dim = self._codebook.shape[1]
        ind = np.arange(0,dim)
        indX = ind[ind != target]
        x = self._codebook[:, indX]
        y = self._codebook[:, target]
        n_neighborbors = k
        clf = neighborbors.KNeighborsRegressor(n_neighborbors, weights=wt)
        clf.fit(x, y)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of original data
        dimdata = data.shape[1]

        if dimdata == dim:
            # data[:, target] == 0 mmmm whas this meant to be an assignment?
            data = normalize_by(self.data_raw, data, method='var')
            data = data[:, indX]

        elif dimdata == dim-1:
            data = normalize_by(self.data_raw[:, indX], data, method='var')

        predicted_values = clf.predict(data)
        predicted_values = denormalize_by(self.data_raw[:, target], predicted_values)
        return predicted_values

    def predict(self, x_test, k=5, wt='distance'):
        """
        Similar to SKlearn we assume that we have X_tr, Y_tr and X_test. Here it is assumed that Target is the last
        column in the codebook and data has dim-1 columns

        @param x_test input vector
        @param k number of neighbors to use
        @param wt method to use for the weights (more detail in KNeighborsRegressor docs)
        """
        target = self.data_raw.shape[1]-1
        x_train = self._codebook[:, :target]
        y_train = self._codebook[:, target]
        clf = neighborbors.KNeighborsRegressor(k, weights=wt)
        clf.fit(x_train, y_train)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of original data
        x_test = normalize_by(self.data_raw[:, :target], x_test, method='var')
        predicted_values = clf.predict(x_test)

        return denormalize_by(self.data_raw[:, target], predicted_values)

    def find_k_nodes(self, data, k=5):
        from sklearn.neighborbors import NearestNeighbors
        # we find the k most similar nodes to the input vector
        neighbor = NearestNeighbors(n_neighborbors=k)
        neighbor.fit(self._codebook)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of original data
        return neighbor.kneighborbors(normalize_by(self.data_raw, data, method='var'))

    def bmu_ind_to_xy(self, bmu_ind):
        """
        Translates a best matching unit index to the corresponding matrix x,y coordinates
        """
        rows = self._mapsize[0]
        cols = self._mapsize[1]

        # bmu should be an integer between 0 to no_nodes
        out = np.zeros((bmu_ind.shape[0], 3))
        out[:, 2] = bmu_ind
        out[:, 0] = rows-1-bmu_ind / cols
        out[:, 0] = bmu_ind / cols
        out[:, 1] = bmu_ind % cols

        return out.astype(int)

    def cluster(self, method='Kmeans', n_clusters=8):
        import sklearn.cluster as clust
        return clust.KMeans(n_clusters=n_clusters).fit_predict(denormalize_by(self.data_raw, self._codebook, n_method='var'))

    def predict_probability(self, data, target, k=5):
        # here it is assumed that Target is the last column in the codebook #and data has dim-1 columns
        dim = self._codebook.shape[1]
        ind = np.arange(0,dim)
        indx = ind[ind != target]
        x = self._codebook[:, indx]
        y = self._codebook[:, target]

        clf = neighborbors.KNeighborsRegressor(k, weights='distance')
        clf.fit(x, y)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of original data
        dimdata = data.shape[1]

        if dimdata == dim: 
            #data[:,Target] == 0  # mmm assignment?
            data = normalize_by(self.data_raw, data, method='var')
            data = data[:, indx]

        elif dimdata == dim-1:
            data = normalize_by(self.data_raw[:, indx], data, method='var')

        weights, ind = clf.kneighborbors(data, n_neighborbors=k, return_distance=True)
        weights = 1./weights
        sum_ = np.sum(weights, axis=1)
        weights = weights/sum_[:, np.newaxis]
        labels = np.sign(self._codebook[ind, target])
        labels[labels >= 0] = 1

        #for positives
        pos_prob = labels.copy()
        pos_prob[pos_prob < 0] = 0
        pos_prob *= weights
        pos_prob = np.sum(pos_prob, axis=1)[:, np.newaxis]

        #for negatives
        neg_prob = labels.copy()
        neg_prob[neg_prob > 0] = 0
        neg_prob = neg_prob * weights * -1
        neg_prob = np.sum(neg_prob, axis=1)[:, np.newaxis]

        #predicted_values = clf.predict(data)
        #predicted_values = denormalize_by(data_raw[:,Target], predicted_values)
        return np.concatenate((pos_prob, neg_prob), axis=1)

    def node_activation(self, data, target=None, wt='distance'):
        weights, ind = None, None

        if not target:
            clf = neighborbors.KNeighborsClassifier(n_neighborbors=self._nnodes)
            labels = np.arange(0, self._codebook.shape[0])
            clf.fit(self._codebook, labels)

            # The codebook values are all normalized
            # we can normalize the input data based on mean and std of original data
            data = normalize_by(self.data_raw, data, method='var')
            weights, ind = clf.kneighborbors(data)

            ##Softmax function
            weights = 1./weights
            #S_  = np.sum(np.exp(weights),axis=1)[:,np.newaxis]
            #weights = np.exp(weights)/S_

        return weights, ind

    def linear_init(self):
        """
        We initialize the map, just by using the first two first eigen vals and eigenvectors
        Further, we create a linear combination of them in the new map by giving values from -1 to 1 in each

        X = UsigmaWT
        XTX = Wsigma^2WT
        T = XW = Usigma

        // Transformed by W EigenVector, can be calculated by multiplication PC matrix by eigenval too
        // Further, we can get lower ranks by using just few of the eigen vevtors

        T(2) = U(2)sigma(2) = XW(2) ---> 2 is the number of selected eigenvectors

        (*) Note that 'X' is the covariance matrix of original data

        """
        cols = self._mapsize[1]
        coord = None
        pca_components = None

        if np.min(self._mapsize) > 1:
            coord = np.zeros((self._nnodes, 2))
            pca_components = 2

            for i in range(0, self._nnodes):
                coord[i, 0] = int(i / cols)  # x
                coord[i, 1] = int(i % cols)  # y

        elif np.min(self._mapsize) == 1:
            coord = np.zeros((self._nnodes, 1))
            pca_components = 1

            for i in range(0, self._nnodes):
                coord[i, 0] = int(i % cols)  # y

        mx = np.max(coord, axis=0)
        mn = np.min(coord, axis=0)
        coord = (coord - mn)/(mx-mn)
        coord = (coord - .5)*2
        me = np.mean(self._data, 0)
        data = (self._data - me)
        self._codebook = np.tile(me, (self._nnodes, 1))

        pca = RandomizedPCA(n_components=pca_components)  # Randomized PCA is scalable
        pca.fit(data)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
        eigvec = ((eigvec.T/norms)*eigval).T

        for j in range(self._nnodes):
            for i in range(eigvec.shape[0]):
                self._codebook[j, :] = self._codebook[j, :] + coord[j, i]*eigvec[i, :]

        return np.around(self._codebook, decimals=6)


def _mean_and_standard_dev(data):
    return np.mean(data, axis=0), np.std(data, axis=0)


def normalize(data, method='var'):
    #methods  = ['var','range','log','logistic','histD','histC']
    #status = ['done', 'undone']
    normalized_data = data
    me, st = _mean_and_standard_dev(data)

    if method == 'var':
        normalized_data = (data-me)/st

    return normalized_data


def normalize_by(data_raw, data, method='var'):
    #methods  = ['var','range','log','logistic','histD','histC']
    #status = ['done', 'undone']
    # to have the mean and std of the original data, by which SOM is trained
    normalized_data = data
    me, st = _mean_and_standard_dev(data_raw)

    if method == 'var':
        normalized_data = (data-me)/st

    return normalized_data


def denormalize_by(data_by, n_vect, n_method='var'):
    denormalized_data = n_vect
    me, st = _mean_and_standard_dev(data_by)

    if n_method == 'var':
        denormalized_data = n_vect * st + me

    else:
        print 'data is not normalized before'

    return denormalized_data


def l(a, b):
    c = np.zeros(b.shape)
    c[a-b >= 0] = 1
    return c
