# -*- coding: utf-8 -*-

# Author: Vahid Moosavi (sevamoo@gmail.com)
#         Chair For Computer Aided Architectural Design, ETH  Zurich
#         Future Cities Lab
#         www.vahidmoosavi.com

# Contributor: Sebastian Packmann (sebastian.packmann@gmail.com)


import tempfile
import os
import itertools
import logging

import numpy as np

from time import time
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
from sklearn import neighbors
from joblib import Parallel, delayed, load, dump
import sys

from .decorators import timeit
from .codebook import Codebook
from .neighborhood import NeighborhoodFactory
from .normalization import NormalizerFactory


class ComponentNamesError(Exception):
    pass


class LabelsError(Exception):
    pass


class SOMFactory(object):

    @staticmethod
    def build(data,
              mapsize=None,
              mask=None,
              mapshape='planar',
              lattice='rect',
              normalization='var',
              initialization='pca',
              neighborhood='gaussian',
              training='batch',
              name='sompy',
              component_names=None):
        """
        :param data: data to be clustered, represented as a matrix of n rows,
            as inputs and m cols as input features
        :param neighborhood: neighborhood object calculator.  Options are:
            - gaussian
            - bubble
            - manhattan (not implemented yet)
            - cut_gaussian (not implemented yet)
            - epanechicov (not implemented yet)

        :param normalization: normalizer object calculator. Options are:
            - var

        :param mapsize: tuple/list defining the dimensions of the som.
            If single number is provided is considered as the number of nodes.
        :param mask: mask
        :param mapshape: shape of the som. Options are:
            - planar
            - toroid (not implemented yet)
            - cylinder (not implemented yet)

        :param lattice: type of lattice. Options are:
            - rect
            - hexa

        :param initialization: method to be used for initialization of the som.
            Options are:
            - pca
            - random

        :param name: name used to identify the som
        :param training: Training mode (seq, batch)
        """
        if normalization:
            normalizer = NormalizerFactory.build(normalization)
        else:
            normalizer = None
        neighborhood_calculator = NeighborhoodFactory.build(neighborhood)
        return SOM(data, neighborhood_calculator, normalizer, mapsize, mask,
                   mapshape, lattice, initialization, training, name, component_names)


class SOM(object):

    def __init__(self,
                 data,
                 neighborhood,
                 normalizer=None,
                 mapsize=None,
                 mask=None,
                 mapshape='planar',
                 lattice='rect',
                 initialization='pca',
                 training='batch',
                 name='sompy',
                 component_names=None):
        """
        Self Organizing Map

        :param data: data to be clustered, represented as a matrix of n rows,
            as inputs and m cols as input features
        :param neighborhood: neighborhood object calculator.
        :param normalizer: normalizer object calculator.
        :param mapsize: tuple/list defining the dimensions of the som. If
            single number is provided is considered as the number of nodes.
        :param mask: mask
        :param mapshape: shape of the som.
        :param lattice: type of lattice.
        :param initialization: method to be used for initialization of the som.
        :param name: name used to identify the som
        :param training: Training mode (seq, batch)
        """
        self._data = normalizer.normalize(data) if normalizer else data
        self._normalizer = normalizer
        self._dim = data.shape[1]
        self._dlen = data.shape[0]
        self._dlabel = None
        self._bmu = None

        self.name = name
        self.data_raw = data
        self.neighborhood = neighborhood
        self.mapshape = mapshape
        self.initialization = initialization
        self.mask = mask or np.ones([1, self._dim])
        mapsize = self.calculate_map_size(lattice) if not mapsize else mapsize
        self.codebook = Codebook(mapsize, lattice)
        self.training = training
        self._component_names = self.build_component_names() if component_names is None else [component_names]
        self._distance_matrix = self.calculate_map_dist()

    @property
    def component_names(self):
        return self._component_names

    @component_names.setter
    def component_names(self, compnames):
        if self._dim == len(compnames):
            self._component_names = np.asarray(compnames)[np.newaxis, :]
        else:
            raise ComponentNamesError('Component names should have the same '
                                      'size as the data dimension/features')

    def build_component_names(self):
        cc = ['Variable-' + str(i+1) for i in range(0, self._dim)]
        return np.asarray(cc)[np.newaxis, :]

    @property
    def data_labels(self):
        return self._dlabel

    @data_labels.setter
    def data_labels(self, labels):
        """
        Set labels of the training data, it should be in the format of a list
        of strings
        """
        if labels.shape == (1, self._dlen):
            label = labels.T
        elif labels.shape == (self._dlen, 1):
            label = labels
        elif labels.shape == (self._dlen,):
            label = labels[:, np.newaxis]
        else:
            raise LabelsError('wrong label format')

        self._dlabel = label

    def build_data_labels(self):
        cc = ['dlabel-' + str(i) for i in range(0, self._dlen)]
        return np.asarray(cc)[:, np.newaxis]

    def calculate_map_dist(self):
        """
        Calculates the grid distance, which will be used during the training
        steps. It supports only planar grids for the moment
        """
        nnodes = self.codebook.nnodes

        distance_matrix = np.zeros((nnodes, nnodes))
        for i in range(nnodes):
            distance_matrix[i] = self.codebook.grid_dist(i).reshape(1, nnodes)
        return distance_matrix

    @timeit()
    def train(self,
              n_job=1,
              shared_memory=False,
              verbose='info',
              train_rough_len=None,
              train_rough_radiusin=None,
              train_rough_radiusfin=None,
              train_finetune_len=None,
              train_finetune_radiusin=None,
              train_finetune_radiusfin=None,
              train_len_factor=1,
              maxtrainlen=np.Inf):
        """
        Trains the som

        :param n_job: number of jobs to use to parallelize the traning
        :param shared_memory: flag to active shared memory
        :param verbose: verbosity, could be 'debug', 'info' or None
        :param train_len_factor: Factor that multiply default training lenghts (similar to "training" parameter in the matlab version). (lbugnon)
        """
        logging.root.setLevel(
            getattr(logging, verbose.upper()) if verbose else logging.ERROR)

        logging.info(" Training...")
        logging.debug((
            "--------------------------------------------------------------\n"
            " details: \n"
            "      > data len is {data_len} and data dimension is {data_dim}\n"
            "      > map size is {mpsz0},{mpsz1}\n"
            "      > array size in log10 scale is {array_size}\n"
            "      > number of jobs in parallel: {n_job}\n"
            " -------------------------------------------------------------\n")
            .format(data_len=self._dlen,
                    data_dim=self._dim,
                    mpsz0=self.codebook.mapsize[0],
                    mpsz1=self.codebook.mapsize[1],
                    array_size=np.log10(
                        self._dlen * self.codebook.nnodes * self._dim),
                    n_job=n_job))

        if self.initialization == 'random':
            self.codebook.random_initialization(self._data)

        elif self.initialization == 'pca':
            self.codebook.pca_linear_initialization(self._data)

        self.rough_train(njob=n_job, shared_memory=shared_memory, trainlen=train_rough_len,
                         radiusin=train_rough_radiusin, radiusfin=train_rough_radiusfin,trainlen_factor=train_len_factor,maxtrainlen=maxtrainlen)
        self.finetune_train(njob=n_job, shared_memory=shared_memory, trainlen=train_finetune_len,
                            radiusin=train_finetune_radiusin, radiusfin=train_finetune_radiusfin,trainlen_factor=train_len_factor,maxtrainlen=maxtrainlen)
        logging.debug(
            " --------------------------------------------------------------")
        logging.info(" Final quantization error: %f" % np.mean(self._bmu[1]))

    def _calculate_ms_and_mpd(self):
        mn = np.min(self.codebook.mapsize)
        max_s = max(self.codebook.mapsize[0], self.codebook.mapsize[1])

        if mn == 1:
            mpd = float(self.codebook.nnodes*10)/float(self._dlen)
        else:
            mpd = float(self.codebook.nnodes)/float(self._dlen)
        ms = max_s/2.0 if mn == 1 else max_s

        return ms, mpd

    def rough_train(self, njob=1, shared_memory=False, trainlen=None, radiusin=None, radiusfin=None,trainlen_factor=1,maxtrainlen=np.Inf):
        logging.info(" Rough training...")

        ms, mpd = self._calculate_ms_and_mpd()
        #lbugnon: add maxtrainlen
        trainlen = min(int(np.ceil(30*mpd)),maxtrainlen) if not trainlen else trainlen
        #print("maxtrainlen %d",maxtrainlen)
        #lbugnon: add trainlen_factor
        trainlen=int(trainlen*trainlen_factor)
        
        if self.initialization == 'random':
            radiusin = max(1, np.ceil(ms/3.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin/6.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            radiusin = max(1, np.ceil(ms/8.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin/4.) if not radiusfin else radiusfin

        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory)

    def finetune_train(self, njob=1, shared_memory=False, trainlen=None, radiusin=None, radiusfin=None,trainlen_factor=1,maxtrainlen=np.Inf):
        logging.info(" Finetune training...")

        ms, mpd = self._calculate_ms_and_mpd()

        #lbugnon: add maxtrainlen
        if self.initialization == 'random':
            trainlen = min(int(np.ceil(50*mpd)),maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, ms/12.)  if not radiusin else radiusin # from radius fin in rough training
            radiusfin = max(1, radiusin/25.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            trainlen = min(int(np.ceil(40*mpd)),maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, np.ceil(ms/8.)/4) if not radiusin else radiusin
            radiusfin = 1 if not radiusfin else radiusfin # max(1, ms/128)

        #print("maxtrainlen %d",maxtrainlen)
        
        #lbugnon: add trainlen_factor
        trainlen=int(trainlen_factor*trainlen)
        
            
        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory)

    def _batchtrain(self, trainlen, radiusin, radiusfin, njob=1,
                    shared_memory=False):
        radius = np.linspace(radiusin, radiusfin, trainlen)

        if shared_memory:
            data = self._data
            data_folder = tempfile.mkdtemp()
            data_name = os.path.join(data_folder, 'data')
            dump(data, data_name)
            data = load(data_name, mmap_mode='r')

        else:
            data = self._data

        bmu = None

        # X2 is part of euclidean distance (x-y)^2 = x^2 +y^2 - 2xy that we use
        # for each data row in bmu finding.
        # Since it is a fixed value we can skip it during bmu finding for each
        # data point, but later we need it calculate quantification error
        fixed_euclidean_x2 = np.einsum('ij,ij->i', data, data)

        logging.info(" radius_ini: %f , radius_final: %f, trainlen: %d\n" %
                     (radiusin, radiusfin, trainlen))

        for i in range(trainlen):
            t1 = time()
            neighborhood = self.neighborhood.calculate(
                self._distance_matrix, radius[i], self.codebook.nnodes)
            bmu = self.find_bmu(data, njb=njob)
            self.codebook.matrix = self.update_codebook_voronoi(data, bmu,
                                                                neighborhood)

            #lbugnon: ojo! aca el bmy[1] a veces da negativo, y despues de eso se rompe...hay algo raro ahi
            qerror = (i + 1, round(time() - t1, 3),
                      np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2))) #lbugnon: ojo aca me tiró un warning, revisar (commit sinc: 965666d3d4d93bcf48e8cef6ea2c41a018c1cb83 )
            #lbugnon
            #ipdb.set_trace()
            #
            logging.info(
                " epoch: %d ---> elapsed time:  %f, quantization error: %f\n" %
                qerror)
            if np.any(np.isnan(qerror)):
                logging.info("nan quantization error, exit train\n")
                
                #sys.exit("quantization error=nan, exit train")

        bmu = self.find_bmu(data, njb=njob)
        bmu[1] = np.sqrt(bmu[1] + fixed_euclidean_x2)
        self._bmu = bmu

    @timeit(logging.DEBUG)
    def find_bmu(self, input_matrix, njb=1, nth=1):
        """
        Finds the best matching unit (bmu) for each input data from the input
        matrix. It does all at once parallelizing the calculation instead of
        going through each input and running it against the codebook.

        :param input_matrix: numpy matrix representing inputs as rows and
            features/dimension as cols
        :param njb: number of jobs to parallelize the search
        :returns: the best matching unit for each input
        """
        dlen = input_matrix.shape[0]
        y2 = np.einsum('ij,ij->i', self.codebook.matrix, self.codebook.matrix)
        if njb == -1:
            njb = cpu_count()

        pool = Pool(njb)
        chunk_bmu_finder = _chunk_based_bmu_find

        def row_chunk(part):
            return part * dlen // njb

        def col_chunk(part):
            return min((part+1)*dlen // njb, dlen)

        chunks = [input_matrix[row_chunk(i):col_chunk(i)] for i in range(njb)]
        b = pool.map(lambda chk: chunk_bmu_finder(chk, self.codebook.matrix, y2, nth=nth), chunks)
        pool.close()
        pool.join()
        bmu = np.asarray(list(itertools.chain(*b))).T
        del b
        return bmu

    @timeit(logging.DEBUG)
    def update_codebook_voronoi(self, training_data, bmu, neighborhood):
        """
        Updates the weights of each node in the codebook that belongs to the
        bmu's neighborhood.

        First finds the Voronoi set of each node. It needs to calculate a
        smaller matrix.
        Super fast comparing to classic batch training algorithm, it is based
        on the implemented algorithm in som toolbox for Matlab by Helsinky
        University.

        :param training_data: input matrix with input vectors as rows and
            vector features as cols
        :param bmu: best matching unit for each input data. Has shape of
            (2, dlen) where first row has bmu indexes
        :param neighborhood: matrix representing the neighborhood of each bmu

        :returns: An updated codebook that incorporates the learnings from the
            input data
        """
        row = bmu[0].astype(int)
        col = np.arange(self._dlen)
        val = np.tile(1, self._dlen)
        P = csr_matrix((val, (row, col)), shape=(self.codebook.nnodes,
                       self._dlen))
        S = P.dot(training_data)

        # neighborhood has nnodes*nnodes and S has nnodes*dim
        # ---> Nominator has nnodes*dim
        nom = neighborhood.T.dot(S)
        nV = P.sum(axis=1).reshape(1, self.codebook.nnodes)
        denom = nV.dot(neighborhood.T).reshape(self.codebook.nnodes, 1)
        new_codebook = np.divide(nom, denom)

        return np.around(new_codebook, decimals=6)

    def project_data(self, data):
        """
        Projects a data set to a trained SOM. It is based on nearest
        neighborhood search module of scikitlearn, but it is not that fast.
        """
        clf = neighbors.KNeighborsClassifier(n_neighbors=1)
        labels = np.arange(0, self.codebook.matrix.shape[0])
        clf.fit(self.codebook.matrix, labels)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        data = self._normalizer.normalize_by(self.data_raw, data)

        return clf.predict(data)

    def predict_by(self, data, target, k=5, wt='distance'):
        # here it is assumed that target is the last column in the codebook
        # and data has dim-1 columns
        dim = self.codebook.matrix.shape[1]
        ind = np.arange(0, dim)
        indX = ind[ind != target]
        x = self.codebook.matrix[:, indX]
        y = self.codebook.matrix[:, target]
        n_neighbors = k
        clf = neighbors.KNeighborsRegressor(n_neighbors, weights=wt)
        clf.fit(x, y)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        dimdata = data.shape[1]

        if dimdata == dim:
            data[:, target] = 0
            data = self._normalizer.normalize_by(self.data_raw, data)
            data = data[:, indX]

        elif dimdata == dim-1:
            data = self._normalizer.normalize_by(self.data_raw[:, indX], data)

        predicted_values = clf.predict(data)
        predicted_values = self._normalizer.denormalize_by(
            self.data_raw[:, target], predicted_values)
        return predicted_values

    def predict(self, x_test, k=5, wt='distance'):
        """
        Similar to SKlearn we assume that we have X_tr, Y_tr and X_test. Here
        it is assumed that target is the last column in the codebook and data
        has dim-1 columns

        :param x_test: input vector
        :param k: number of neighbors to use
        :param wt: method to use for the weights
            (more detail in KNeighborsRegressor docs)
        :returns: predicted values for the input data
        """
        target = self.data_raw.shape[1]-1
        x_train = self.codebook.matrix[:, :target]
        y_train = self.codebook.matrix[:, target]
        clf = neighbors.KNeighborsRegressor(k, weights=wt)
        clf.fit(x_train, y_train)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        x_test = self._normalizer.normalize_by(
            self.data_raw[:, :target], x_test)
        predicted_values = clf.predict(x_test)

        return self._normalizer.denormalize_by(
            self.data_raw[:, target], predicted_values)

    def find_k_nodes(self, data, k=5):
        from sklearn.neighbors import NearestNeighbors
        # we find the k most similar nodes to the input vector
        neighbor = NearestNeighbors(n_neighbors=k)
        neighbor.fit(self.codebook.matrix)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        return neighbor.kneighbors(
            self._normalizer.normalize_by(self.data_raw, data))

    def bmu_ind_to_xy(self, bmu_ind):
        """
        Translates a best matching unit index to the corresponding
        matrix x,y coordinates.

        :param bmu_ind: node index of the best matching unit
            (number of node from top left node)
        :returns: corresponding (x,y) coordinate
        """
        rows = self.codebook.mapsize[0]
        cols = self.codebook.mapsize[1]

        # bmu should be an integer between 0 to no_nodes
        out = np.zeros((bmu_ind.shape[0], 3))
        out[:, 2] = bmu_ind
        out[:, 0] = rows-1-bmu_ind / cols
        out[:, 0] = bmu_ind / cols
        out[:, 1] = bmu_ind % cols

        return out.astype(int)

    def cluster(self, n_clusters=8, random_state=0):
        import sklearn.cluster as clust
        cl_labels = clust.KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(
            self._normalizer.denormalize_by(self.data_raw,
                                            self.codebook.matrix))
        self.cluster_labels = cl_labels
        return cl_labels

    def predict_probability(self, data, target, k=5):
        """
        Predicts probability of the input data to be target

        :param data: data to predict, it is assumed that 'target' is the last
            column in the codebook, so data hould have dim-1 columns
        :param target: target to predict probability
        :param k: k parameter on KNeighborsRegressor
        :returns: probability of data been target
        """
        dim = self.codebook.matrix.shape[1]
        ind = np.arange(0, dim)
        indx = ind[ind != target]
        x = self.codebook.matrix[:, indx]
        y = self.codebook.matrix[:, target]

        clf = neighbors.KNeighborsRegressor(k, weights='distance')
        clf.fit(x, y)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        dimdata = data.shape[1]

        if dimdata == dim:
            data[:, target] = 0
            data = self._normalizer.normalize_by(self.data_raw, data)
            data = data[:, indx]

        elif dimdata == dim-1:
            data = self._normalizer.normalize_by(self.data_raw[:, indx], data)

        weights, ind = clf.kneighbors(data, n_neighbors=k,
                                      return_distance=True)
        weights = 1./weights
        sum_ = np.sum(weights, axis=1)
        weights = weights/sum_[:, np.newaxis]
        labels = np.sign(self.codebook.matrix[ind, target])
        labels[labels >= 0] = 1

        # for positives
        pos_prob = labels.copy()
        pos_prob[pos_prob < 0] = 0
        pos_prob *= weights
        pos_prob = np.sum(pos_prob, axis=1)[:, np.newaxis]

        # for negatives
        neg_prob = labels.copy()
        neg_prob[neg_prob > 0] = 0
        neg_prob = neg_prob * weights * -1
        neg_prob = np.sum(neg_prob, axis=1)[:, np.newaxis]

        return np.concatenate((pos_prob, neg_prob), axis=1)

    def node_activation(self, data, target=None, wt='distance'):
        weights, ind = None, None

        if not target:
            clf = neighbors.KNeighborsClassifier(
                n_neighbors=self.codebook.nnodes)
            labels = np.arange(0, self.codebook.matrix.shape[0])
            clf.fit(self.codebook.matrix, labels)

            # The codebook values are all normalized
            # we can normalize the input data based on mean and std of
            # original data
            data = self._normalizer.normalize_by(self.data_raw, data)
            weights, ind = clf.kneighbors(data)

            # Softmax function
            weights = 1./weights

        return weights, ind

    def calculate_topographic_error(self):
        bmus1 = self.find_bmu(self.data_raw, njb=1, nth=1)
        bmus2 = self.find_bmu(self.data_raw, njb=1, nth=2)
        topographic_error = None
        if self.codebook.lattice=="rect":
            bmus_gap = np.abs((self.bmu_ind_to_xy(np.array(bmus1[0]))[:, 0:2] - self.bmu_ind_to_xy(np.array(bmus2[0]))[:, 0:2]).sum(axis=1))
            topographic_error = np.mean(bmus_gap != 1)
        elif self.codebook.lattice=="hexa":
            dist_matrix_1 = self.codebook.lattice_distances[bmus1[0].astype(int)].reshape(len(bmus1[0]), -1)
            topographic_error = (np.array(
                [distances[bmu2] for bmu2, distances in zip(bmus2[0].astype(int), dist_matrix_1)]) > 2).mean()
        return(topographic_error)

    def calculate_quantization_error(self):
        neuron_values = self.codebook.matrix[self.find_bmu(self._data)[0].astype(int)]
        quantization_error = np.mean(np.abs(neuron_values - self._data))
        return quantization_error

    def calculate_map_size(self, lattice):
        """
        Calculates the optimal map size given a dataset using eigenvalues and eigenvectors. Matlab ported
        :lattice: 'rect' or 'hex'
        :return: map sizes
        """
        D = self.data_raw.copy()
        dlen = D.shape[0]
        dim = D.shape[1]
        munits = np.ceil(5 * (dlen ** 0.5))
        A = np.ndarray(shape=[dim, dim]) + np.Inf

        for i in range(dim):
            D[:, i] = D[:, i] - np.mean(D[np.isfinite(D[:, i]), i])

        for i in range(dim):
            for j in range(dim):
                c = D[:, i] * D[:, j]
                c = c[np.isfinite(c)]
                A[i, j] = sum(c) / len(c)
                A[j, i] = A[i, j]

        VS = np.linalg.eig(A)
        eigval = sorted(np.linalg.eig(A)[0])
        if eigval[-1] == 0 or eigval[-2] * munits < eigval[-1]:
            ratio = 1
        else:
            ratio = np.sqrt(eigval[-1] / eigval[-2])

        if lattice == "rect":
            size1 = min(munits, round(np.sqrt(munits / ratio)))
        else:
            size1 = min(munits, round(np.sqrt(munits / ratio*np.sqrt(0.75))))

        size2 = round(munits / size1)

        return [int(size1), int(size2)]


# Since joblib.delayed uses Pickle, this method needs to be a top level
# method in order to be pickled
# Joblib is working on adding support for cloudpickle or dill which will allow
# class methods to be pickled
# when that that comes out we can move this to SOM class
def _chunk_based_bmu_find(input_matrix, codebook, y2, nth=1):
    """
    Finds the corresponding bmus to the input matrix.

    :param input_matrix: a matrix of input data, representing input vector as
                         rows, and vectors features/dimention as cols
                         when parallelizing the search, the input_matrix can be
                         a sub matrix from the bigger matrix
    :param codebook: matrix of weights to be used for the bmu search
    :param y2: <not sure>
    """
    dlen = input_matrix.shape[0]
    nnodes = codebook.shape[0]
    bmu = np.empty((dlen, 2))

    # It seems that small batches for large dlen is really faster:
    # that is because of ddata in loops and n_jobs. for large data it slows
    # down due to memory needs in parallel
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
        bmu[low:high+1, 0] = np.argpartition(d, nth, axis=0)[nth-1]
        bmu[low:high+1, 1] = np.partition(d, nth, axis=0)[nth-1]
        del ddata

    return bmu



