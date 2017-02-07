import numpy as np
from sklearn.decomposition import PCA# RandomizedPCA (randomizedpca is deprecated)
from .decorators import timeit


class InvalidNodeIndexError(Exception):
    pass


class InvalidMapsizeError(Exception):
    pass


class Codebook(object):

    def __init__(self, mapsize, lattice='rect'):
        self.lattice = lattice

        if 2 == len(mapsize):
            _size = [1, np.max(mapsize)] if 1 == np.min(mapsize) else mapsize

        elif 1 == len(mapsize):
            _size = [1, mapsize[0]]
            print('input was considered as the numbers of nodes')
            print('map size is [{dlen},{dlen}]'.format(dlen=int(mapsize[0]/2)))
        else:
            raise InvalidMapsizeError(
                "Mapsize is expected to be a 2 element list or a single int")

        self.mapsize = _size
        self.nnodes = mapsize[0]*mapsize[1]
        self.matrix = np.asarray(self.mapsize)
        self.initialized = False

    @timeit()
    def random_initialization(self, data):
        """
        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """
        mn = np.tile(np.min(data, axis=0), (self.nnodes, 1))
        mx = np.tile(np.max(data, axis=0), (self.nnodes, 1))
        self.matrix = mn + (mx-mn)*(np.random.rand(self.nnodes, data.shape[1]))
        self.initialized = True

    @timeit()
    def pca_linear_initialization(self, data):
        """
        We initialize the map, just by using the first two first eigen vals and
        eigenvectors
        Further, we create a linear combination of them in the new map by
        giving values from -1 to 1 in each

        X = UsigmaWT
        XTX = Wsigma^2WT
        T = XW = Usigma

        // Transformed by W EigenVector, can be calculated by multiplication
        // PC matrix by eigenval too
        // Further, we can get lower ranks by using just few of the eigen
        // vevtors

        T(2) = U(2)sigma(2) = XW(2) ---> 2 is the number of selected
        eigenvectors

        (*) Note that 'X' is the covariance matrix of original data

        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """
        cols = self.mapsize[1]
        coord = None
        pca_components = None

        if np.min(self.mapsize) > 1:
            coord = np.zeros((self.nnodes, 2))
            pca_components = 2

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i / cols)  # x
                coord[i, 1] = int(i % cols)  # y

        elif np.min(self.mapsize) == 1:
            coord = np.zeros((self.nnodes, 1))
            pca_components = 1

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i % cols)  # y

        mx = np.max(coord, axis=0)
        mn = np.min(coord, axis=0)
        coord = (coord - mn)/(mx-mn)
        coord = (coord - .5)*2
        me = np.mean(data, 0)
        data = (data - me)
        tmp_matrix = np.tile(me, (self.nnodes, 1))

        # Randomized PCA is scalable
        #pca = RandomizedPCA(n_components=pca_components)
        pca = PCA(n_components=pca_components, svd_solver='randomized')

        pca.fit(data)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
        eigvec = ((eigvec.T/norms)*eigval).T

        for j in range(self.nnodes):
            for i in range(eigvec.shape[0]):
                tmp_matrix[j, :] = tmp_matrix[j, :] + coord[j, i]*eigvec[i, :]

        self.matrix = np.around(tmp_matrix, decimals=6)
        self.initialized = True

    def grid_dist(self, node_ind):
        """
        Calculates grid distance based on the lattice type.

        :param node_ind: number between 0 and number of nodes-1. Depending on
                         the map size, starting from top left
        :returns: matrix representing the distance matrix
        """
        if self.lattice == 'rect':
            return self._rect_dist(node_ind)

        elif self.lattice == 'hexa':
            return self._hexa_dist(node_ind)

    def _hexa_dist(self, node_ind):
        raise NotImplementedError()

    def _rect_dist(self, node_ind):
        """
        Calculates the distance of the specified node to the other nodes in the
        matrix, generating a distance matrix

        Ej. The distance matrix for the node_ind=5, that corresponds to
        the_coord (1,1)
           array([[2, 1, 2, 5],
                  [1, 0, 1, 4],
                  [2, 1, 2, 5],
                  [5, 4, 5, 8]])

        :param node_ind: number between 0 and number of nodes-1. Depending on
                         the map size, starting from top left
        :returns: matrix representing the distance matrix
        """
        rows = self.mapsize[0]
        cols = self.mapsize[1]
        dist = None

        # bmu should be an integer between 0 to no_nodes
        if 0 <= node_ind <= (rows*cols):
            node_col = int(node_ind % cols)
            node_row = int(node_ind / cols)
        else:
            raise InvalidNodeIndexError(
                "Node index '%s' is invalid" % node_ind)

        if rows > 0 and cols > 0:
            r = np.arange(0, rows, 1)[:, np.newaxis]
            c = np.arange(0, cols, 1)
            dist2 = (r-node_row)**2 + (c-node_col)**2

            dist = dist2.ravel()
        else:
            raise InvalidMapsizeError(
                "One or both of the map dimensions are invalid. "
                "Cols '%s', Rows '%s'".format(cols=cols, rows=rows))

        return dist
