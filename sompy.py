# -*- coding: utf-8 -*-


# Vahid Moosavi 2015 08 08 10:50 am
#sevamoo@gmail.com
#Chair For Computer Aided Architectural Design, ETH  Zurich
# Future Cities Lab
#www.vahidmoosavi.com

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numexpr as ne
from time import time
import scipy.spatial as spdist
import timeit
import sys
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals.joblib import load, dump
import tempfile
import shutil
import os
import itertools
from scipy.sparse import csr_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn import neighborbors
from matplotlib.colors import LogNorm
from matplotlib import cm
import matplotlib
import pandas as pd


class SOM(object):

    def __init__(self,
                 data,
                 mapsize=None,
                 mask=None,
                 mapshape='planar',
                 lattice='rect',
                 normalization='var',
                 initialization='pca',
                 neighbor='Guassian',
                 name='sompy'):
        # available mapshapes ['planar','toroid','cylinder']
        # available lattices ['hexa','rect']
        # available normalizations ['var']
        # available initializations ['pca', 'random']
        # available neighborhood ['Guassian','manhatan','bubble','cut_gaussian','epanechicov']
        # available algorithms = ['seq','batch']
        # available alfa_types = ['linear','inv','power']

        self.name = name
        self.data_raw = data
        self.data = self.normalize(data, method=normalization) if normalization == 'var' else data
        self.dlabel = None
        self.codebook = None
        self.bmu = None
        self.dim = data.shape[1]
        self.dlen = data.shape[0]
        self.initialization = initialization

        # Estos 3 no se usan.. que onda?
        self.algtype = 'batch'
        self.alfaini = 'inv'
        self.alfafinal = .005

        self.neighbor = neighbor
        self.mapshape = mapshape
        self.lattice = lattice
        self.mask = mask or np.ones([1, self.dim])
        self.mapsize, self.nnodes = self.calculate_mapsize_and_nnodes(mapsize)
        self._component_names = self.build_component_names()
        self.UD2 = self.calculate_map_dist()

        #self.set_data_labels()  # slow for large data sets

    def calculate_mapsize_and_nnodes(self, defined_mapsize=None):
        mapsize, nnodes = None, None

        if defined_mapsize:
            if 2 == len(defined_mapsize):
                mapsize = [1, np.max(defined_mapsize)] if 1 == np.min(defined_mapsize) else defined_mapsize

            elif 1 == len(defined_mapsize):
                mapsize = [1, defined_mapsize[0]]
                print 'input was considered as the numbers of nodes'
                print 'map size is [{dlen},{dlen}]'.format(dlen=int(defined_mapsize[0]/2))

            nnodes = mapsize[0]*mapsize[1]
        else:
            tmp = int(round(np.sqrt(self.dlen)))
            nnodes = tmp
            mapsize = [int(3./5*nnodes), int(2./5*nnodes)]

        return mapsize, nnodes

    @property
    def component_names(self):
        return self._component_names
    
    @component_names.setter
    def component_names(self, compnames):
        try:
            if self.dim == len(compnames):
                self._component_names = np.asarray(compnames)[np.newaxis, :]
            else:
                print 'compname should have the same size'
        except:
            pass
            print 'no data yet: plesae first set trainign data to the SOM'

    def build_component_names(self):
        component_names = None

        try:
            cc = list()
            for i in range(0, self.dim):
                cc.append('Variable-' + str(i+1))
                component_names = np.asarray(cc)[np.newaxis, :]
        except:
            pass
            print 'no data yet: plesae first set trainign data to the SOM'

        return component_names

    #Set labels of the training data, it should be in the format of a list of strings
    def build_data_labels(self):
        dlabels = None
        try:
            cc = list()
            for i in range(0, self.dlen):
                cc.append('dlabel-'+ str(i))
                dlabels = np.asarray(cc)[:, np.newaxis]
        except:
            pass
            print 'no data yet: plesae first set trainign data to the SOM'

        return dlabels

    @property
    def data_labels(self):
        return self.dlabel

    @data_labels.setter
    def data_labels(self, labels):
        try:
            if labels.shape == (1, self.dlen):
                label = labels.T
            elif labels.shape == (self.dlen, 1):
                label = labels
            elif labels.shape == (self.dlen,):
                label = labels[:, np.newaxis]
            else:
                print 'wrong lable format'

            self.dlabel = label

        except:
            pass
            print 'no data yet: plesae first set trainign data to the SOM'

    #calculating the grid distance, which will be called during the training steps
    #currently just works for planar grids
    def calculate_map_dist(self):
        cd = self.nnodes
        UD2 = np.zeros((cd, cd))
        for i in range(cd):
            UD2[i, :] = self.grid_dist(i).reshape(1, cd)

        return UD2

    #Main loop of training
    def train(self, trainlen=None, n_job=1, shared_memory='no', verbose='on'):
        t0 = time()
        data = self.data
        nnodes = self.nnodes
        dlen = self.dlen
        dim = self.dim
        mapsize = self.mapsize
        mem = np.log10(dlen*nnodes*dim)
        #print 'data len is %d and data dimension is %d' % (dlen, dim)
        #print 'map size is %d, %d' %(mapsize[0], mapsize[1])
        #print 'array size in log10 scale' , mem 
        #print 'nomber of jobs in parallel: ', n_job 
        #######################################
        #initialization
        if verbose == 'on':
            print 
            print 'initialization method = %s, initializing..' % self.initialization
            print
            t0 = time()

        self.init_map()

        if verbose == 'on':
            print 'initialization done in %f seconds' % round(time()-t0, 3)

        ########################################
        #rough training
        if verbose == 'on':
            print

        self.batchtrain(njob=n_job, phase='rough', shared_memory='no', verbose=verbose)

        if verbose == 'on':
            print
        #######################################
        #Finetuning
        if verbose == 'on':
            print

        self.batchtrain(njob=n_job, phase='finetune', shared_memory='no', verbose=verbose)
        err = np.mean(self.bmu)[1]

        if verbose == 'on':
#         or verbose == 'off':
#             print
            ts = round(time() - t0, 3)
            print
            print "Total time elapsed: %f secodns" % ts
            print "final quantization error: %f" % err

        if verbose == 'final':
#         or verbose == 'off':
#             print
            ts = round(time() - t0, 3)
            print
            print "Total time elapsed: %f secodns" % ts
            print "final quantization error: %f" % err

    def para_bmu_find(self, x, y, njb=1):
        dlen = x.shape[0]
        Y2 = np.einsum('ij,ij->i', y, y)

        t_temp  = time()

        parallelizer = Parallel(n_jobs=njb, pre_dispatch='3*n_jobs')
        func = delayed(self.chunk_based_bmu_find)
        b = parallelizer(func(self, x[i*dlen // njb:min((i+1)*dlen // njb, dlen)], y, Y2) for i in xrange(njb))

        #print 'bmu finding: %f seconds ' %round(time() - t_temp, 3)
        t1 = time()
        bmu = np.asarray(list(itertools.chain(*b))).T
        #print 'bmu to array: %f seconds' %round(time() - t1, 3)
        del b
        return bmu

    #First finds the Voronoi set of each node. It needs to calculate a smaller matrix. Super fast comparing to classic batch training algorithm
    # it is based on the implemented algorithm in som toolbox for Matlab by Helsinky university
    def update_codebook_voronoi(self, training_data, bmu, H, radius):
        #bmu has shape of 2,dlen, where first row has bmuinds
        # we construct ud2 from precomputed UD2 : ud2 = UD2[bmu[0,:]]
        nnodes = self.nnodes
        dlen = self.dlen
        dim = self.dim

        New_Codebook = np.empty((nnodes, dim))
        inds = bmu[0].astype(int)
#         print 'bmu', bmu[0]
#         fig = plt.hist(bmu[0],bins=100)
#         plt.show()
        row = inds
        col = np.arange(dlen)
        val = np.tile(1, dlen)
        P = csr_matrix((val, (row, col)), shape=(nnodes, dlen))
        S = np.empty((nnodes, dim))
        S = P.dot(training_data)
        #assert( S.shape == (nnodes, dim))
        #assert( H.shape == (nnodes, nnodes))

        # H has nnodes*nnodes and S has nnodes*dim  ---> Nominator has nnodes*dim
        #print Nom
        Nom = np.empty((nnodes, nnodes))
        Nom = H.T.dot(S)
        #assert( Nom.shape == (nnodes, dim))
        nV = np.empty((1, nnodes))
        nV = P.sum(axis=1).reshape(1, nnodes)
#         print 'nV', nV
#         print 'H'
#         print  H
        #assert(nV.shape == (1, nnodes))
        Denom = np.empty((nnodes, 1))
        Denom = nV.dot(H.T).reshape(nnodes, 1)
#         print 'Denom'
#         print  Denom
        #assert( Denom.shape == (nnodes, 1))
        New_Codebook = np.divide(Nom, Denom)
#         print 'codebook'
#         print New_Codebook.sum(axis=1)
        #assert (New_Codebook.shape == (nnodes,dim))
        #setattr(som, 'codebook', New_Codebook)
        return np.around(New_Codebook, decimals=6)

    # we will call this function in parallel for different number of jobs
    def chunk_based_bmu_find(self, x, y, Y2):
        dim = x.shape[1]
        dlen = x.shape[0]
        nnodes = y.shape[0]
        bmu = np.empty((dlen, 2))
        #it seems that smal batches for large dlen is really faster:
        # that is because of ddata in loops and n_jobs. for large data it slows down due to memory needs in parallel
        blen = min(50, dlen)
        i0 = 0
        d = None
        t = time()
        while i0+1 <= dlen:
            Low = i0
            High = min(dlen, i0+blen)
            i0 = i0+blen
            ddata = x[Low:High+1]
            d = np.dot(y, ddata.T)
            d *= -2
            d += Y2.reshape(nnodes, 1)
            bmu[Low:High+1, 0] = np.argmin(d, axis=0)
            bmu[Low:High+1, 1] = np.min(d, axis=0)
            del ddata

        return bmu

    #Batch training which is called for rought training as well as finetuning
    def batchtrain(self, njob=1, phase=None, shared_memory='no', verbose='on'):
        t0 = time()
        nnodes = self.nnodes
        dlen = self.dlen
        dim = self.dim
        mapsize = self.mapsize

        #############################################
        # seting the parameters
        initialization = self.initialization

        mn = np.min(mapsize)
        if mn == 1:
            mpd = float(nnodes*10)/float(dlen)
        else:
            mpd = float(nnodes)/float(dlen)

        max_s = max(mapsize[0], mapsize[1])
        ms = max_s/2.0 if mn == 1 else max_s

        #Based on somtoolbox, Matlab
        #case 'train',    sTrain.trainlen = ceil(50*mpd);
        #case 'rough',    sTrain.trainlen = ceil(10*mpd);
        #case 'finetune', sTrain.trainlen = ceil(40*mpd);
        if phase == 'rough':
            #training length
            trainlen = int(np.ceil(30*mpd))
            #radius for updating
            if initialization == 'random':
                radiusin = max(1, np.ceil(ms/3.))
                radiusfin = max(1, radiusin/6.)
    #         	radiusin = max(1, np.ceil(ms/1.))
    #         	radiusfin = max(1, radiusin/2.)
            elif initialization == 'pca':
                radiusin = max(1, np.ceil(ms/8.))
                radiusfin = max(1, radiusin/4.)
        elif phase == 'finetune':
            #train lening length

            #radius for updating
            if initialization == 'random':
                trainlen = int(np.ceil(50*mpd))
                radiusin = max(1, ms/12.) #from radius fin in rough training
                radiusfin = max(1, radiusin/25.)

    #             radiusin = max(1, ms/2.) #from radius fin in rough training
    #             radiusfin = max(1, radiusin/2.)
            elif initialization == 'pca':
                trainlen = int(np.ceil(40*mpd))
                radiusin = max(1, np.ceil(ms/8.)/4)
                radiusfin = 1#max(1, ms/128)

        radius = np.linspace(radiusin, radiusfin, trainlen)
        ##################################################

        UD2 = self.UD2
        New_Codebook_V = np.empty((nnodes, dim))
        New_Codebook_V = self.codebook

        #print 'data is in shared memory?', shared_memory
        if shared_memory == 'yes':
            data = self.data
            data_folder = tempfile.mkdtemp()
            data_name = os.path.join(data_folder, 'data')
            dump(data, data_name)
            data = load(data_name, mmap_mode='r')
        else:
            data = self.data

        #X2 is part of euclidean distance (x-y)^2 = x^2 +y^2 - 2xy that we use for each data row in bmu finding.
        #Since it is a fixed value we can skip it during bmu finding for each data point, but later we need it calculate quantification error
        X2 = np.einsum('ij,ij->i', data, data)

        if verbose == 'on':
            print '%s training...' % phase
            print 'radius_ini: %f , radius_final: %f, trainlen: %d' %(radiusin, radiusfin, trainlen)

        neighbor_func = self.neighbor

        for i in range(trainlen):
            if neighbor_func == 'guassian':
                #in case of Guassian neighborborhood
                H = np.exp(-1.0*UD2/(2.0*radius[i]**2)).reshape(nnodes, nnodes)

            if neighbor_func == 'bubble':
                # in case of Bubble function
    #         	print radius[i], UD2.shape
    #         	print UD2
                H = l(radius[i], np.sqrt(UD2.flatten())).reshape(nnodes, nnodes) + .000000000001
    #         	print H

            t1 = time()
            bmu = self.para_bmu_find(data, New_Codebook_V, njb=njob)

            if verbose == 'on':
                print
            #updating the codebook
            t2 = time()
            New_Codebook_V = self.update_codebook_voronoi(data, bmu, H, radius)
            #print 'updating nodes: ', round (time()- t2, 3)

            if verbose == 'on':
                print "epoch: %d ---> elapsed time:  %f, quantization error: %f " % (i+1, round(time() - t1, 3), np.mean(np.sqrt(bmu[1] + X2)))

        self.codebook = New_Codebook_V

        bmu[1] = np.sqrt(bmu[1] + X2)
        self.bmu = bmu

    def grid_dist(self, bmu_ind):
        """
        som and bmu_ind
        depending on the lattice "hexa" or "rect" we have different grid distance
        functions.
        bmu_ind is a number between 0 and number of nodes-1. depending on the map size
        bmu_coord will be calculated and then distance matrix in the map will be returned
        """
        try:
            lattice = self.lattice
        except:
            lattice = 'hexa'
            print 'lattice not found! Lattice as hexa was set'

        if lattice == 'rect':
            return self.rect_dist(bmu_ind)
        elif lattice == 'hexa':
            try:
                msize = self.mapsize
                rows = msize[0]
                cols = msize[1]
            except:
                rows = 0.
                cols = 0.
                pass

            #needs to be implemented
            print 'to be implemented', rows, cols
            return np.zeros((rows, cols))

    def rect_dist(self,bmu):
        #the way we consider the list of nodes in a planar grid is that node0 is on top left corner,
        #nodemapsz[1]-1 is top right corner and then it goes to the second row.
        #no. of rows is map_size[0] and no. of cols is map_size[1]
        try:
            msize = self.mapsize
            rows = msize[0]
            cols = msize[1]
        except:
            pass

        #bmu should be an integer between 0 to no_nodes
        if 0 <= bmu <= (rows*cols):
            c_bmu = int(bmu%cols)
            r_bmu = int(bmu/cols)
        else:
          print 'wrong bmu'

        #calculating the grid distance
        if np.logical_and(rows>0 , cols>0):
            r, c = np.arange(0, rows, 1)[:, np.newaxis], np.arange(0, cols, 1)
            dist2 = (r-r_bmu)**2 + (c-c_bmu)**2
            return dist2.ravel()
        else:
            print 'please consider the above mentioned errors'
            return np.zeros((rows, cols)).ravel()


    #to project a data set to a trained SOM and find the index of bmu 
    #It is based on nearest neighborborhood search module of scikitlearn, but it is not that fast.
    def project_data(self, data):
        codebook = self.codebook
        data_raw = self.data_raw
        clf = neighborbors.KNeighborsClassifier(n_neighborbors=1)
        labels = np.arange(0, codebook.shape[0])
        clf.fit(codebook, labels)

        # the codebook values are all normalized
        #we can normalize the input data based on mean and std of original data
        data = normalize_by(data_raw, data, method='var')
        #data = normalize(data, method='var')
        #plt.hist(data[:,2])
        Predicted_labels = clf.predict(data)
        return Predicted_labels


    def predict_by(self, data, Target, K =5, wt= 'distance'):
        """
        ‘uniform’
        """
        # here it is assumed that Target is the last column in the codebook
        #and data has dim-1 columns
        codebook = getattr(self, 'codebook')
        data_raw = getattr(self,'data_raw')
        dim = codebook.shape[1]
        ind = np.arange(0,dim)
        indX = ind[ind != Target]
        X = codebook[:,indX]
        Y = codebook[:,Target]
        n_neighborbors = K
        clf = neighborbors.KNeighborsRegressor(n_neighborbors, weights = wt)
        clf.fit(X, Y)
        # the codebook values are all normalized
        #we can normalize the input data based on mean and std of original data
        dimdata = data.shape[1]
        if dimdata == dim: 
            data[:,Target] == 0   
            data = normalize_by(data_raw, data, method='var')
            data = data[:,indX]
        elif dimdata == dim -1:          
            data = normalize_by(data_raw[:,indX], data, method='var')       
            #data = normalize(data, method='var')
        Predicted_values = clf.predict(data)
        Predicted_values = denormalize_by(data_raw[:,Target], Predicted_values)
        return Predicted_values


    def predict(self, X_test, K =5, wt= 'distance'):
        """
        ‘uniform’
        """
        #Similar to SKlearn we assume that we have X_tr, Y_tr and X_test
        # here it is assumed that Target is the last column in the codebook
        #and data has dim-1 columns
        codebook = getattr(self, 'codebook')
        data_raw = getattr(self,'data_raw')
        dim = codebook.shape[1]
        Target = data_raw.shape[1]-1
        X_train = codebook[:,:Target]
        Y_train= codebook[:,Target]
        n_neighborbors = K
        clf = neighborbors.KNeighborsRegressor(n_neighborbors, weights = wt)
        clf.fit(X_train, Y_train)
        # the codebook values are all normalized
        #we can normalize the input data based on mean and std of original data
        X_test = normalize_by(data_raw[:,:Target], X_test, method='var')
        Predicted_values = clf.predict(X_test)
        Predicted_values = denormalize_by(data_raw[:,Target], Predicted_values)
        return Predicted_values


    def find_K_nodes(self, data, K =5):
        from sklearn.neighborbors import NearestNeighbors
        # we find the k most similar nodes to the input vector
        codebook = getattr(self, 'codebook')
        neighbor = NearestNeighbors(n_neighborbors = K)
        neighbor.fit(codebook) 
        data_raw = getattr(self,'data_raw')
        # the codebook values are all normalized
        #we can normalize the input data based on mean and std of original data
        data = normalize_by(data_raw, data, method='var')
        return neighbor.kneighborbors(data)  

    def ind_to_xy(self, bm_ind):
        msize =  getattr(self, 'mapsize')
        rows = msize[0]
        cols = msize[1]
        #bmu should be an integer between 0 to no_nodes
        out = np.zeros((bm_ind.shape[0],3))
        out[:,2] = bm_ind
        out[:,0] = rows-1-bm_ind/cols
        out[:,0] = bm_ind/cols
        out[:,1] = bm_ind%cols
        return out.astype(int)

    def cluster(self,method='Kmeans',n_clusters=8):
        import sklearn.cluster as clust
        km= clust.KMeans(n_clusters=n_clusters)
        labels = km.fit_predict(denormalize_by(self.data_raw, self.codebook, n_method = 'var'))
        setattr(self,'cluster_labels',labels)
        return labels

    ###################################
    #visualize map
    def view_map(self, what = 'codebook', which_dim = 'all', pack= 'Yes', text_size = 2.8,save='No', save_dir = 'empty',grid='No',text='Yes',cmap='None',COL_SiZe=6):

        mapsize = getattr(self, 'mapsize')
        if np.min(mapsize) >1:
            if pack == 'No':
                self.view_2d(self, text_size, which_dim = which_dim, what = what)
            else:
#         		print 'hi'
                self.view_2d_Pack(self, text_size, which_dim = which_dim,what = what,save = save, save_dir = save_dir, grid=grid,text=text,CMAP=cmap,col_sz=COL_SiZe)

        elif np.min(mapsize) == 1:
            self.view_1d(self, text_size, which_dim = which_dim, what = what)

    ################################################################################
    # Initialize map codebook: Weight vectors of SOM
    def init_map(self):
        dim = 0
        n_nod = 0
        if  getattr(self, 'initialization')=='random':
            #It produces random values in the range of min- max of each dimension based on a uniform distribution
            mn = np.tile(np.min(getattr(self,'data'), axis =0), (getattr(self, 'nnodes'),1))
            mx = np.tile(np.max(getattr(self,'data'), axis =0), (getattr(self, 'nnodes'),1))
            setattr(self, 'codebook', mn + (mx-mn)*(np.random.rand(getattr(self, 'nnodes'), getattr(self, 'dim'))))
        elif getattr(self, 'initialization') == 'pca':
            codebooktmp = self.lininit() #it is based on two largest eigenvalues of correlation matrix
            setattr(self, 'codebook', codebooktmp)
        else:
            print 'please select a corect initialization method'
            print 'set a correct one in SOM. current SOM.initialization:  ', getattr(self, 'initialization')
            print "possible init methods:'random', 'pca'"


    def hit_map(self,data=None):
        #First Step: show the hitmap of all the training data

#     	print 'None'
        data_tr = getattr(self, 'data_raw')
        proj = self.project_data(data_tr)
        msz =  getattr(self, 'mapsize')
        coord = self.ind_to_xy(proj)

        #this is not an appropriate way, but it works
#     	coord[:,0] = msz[0]-coord[:,0]

        ###############################
        fig = plt.figure(figsize=(msz[1]/5,msz[0]/5))
        ax = fig.add_subplot(111)
        ax.xaxis.set_ticks([i for i in range(0,msz[1])])
        ax.yaxis.set_ticks([i for i in range(0,msz[0])])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.grid(True,linestyle='-', linewidth=.5)
        a = plt.hist2d(coord[:,1], coord[:,0], bins=(msz[1],msz[0]),alpha=.0,cmap=cm.jet)
        # clbar  = plt.colorbar()
        x = np.arange(.5,msz[1]+.5,1)
        y = np.arange(.5,msz[0]+.5,1)
        X, Y = np.meshgrid(x, y)
        area = a[0].T*12

        # plt.scatter(coord[:,1]+.5, msz[0]-.5-coord[:,0], s=area, alpha=0.9,c='None',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
#     	plt.scatter(coord[:,1]+.5, msz[0]-.5-coord[:,0], s=area, alpha=0.2,c='b',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
        coord = self.ind_to_xy(np.arange(self.nnodes))
        plt.scatter(coord[:,1], msz[0]-.5- coord[:,0], s=area.flatten(), alpha=.9,c='None',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
        plt.xlim(0,msz[1])
        plt.ylim(0,msz[0])

        if data != None:
            proj = self.project_data(data)
            msz =  getattr(self, 'mapsize')
            coord_d = self.ind_to_xy(proj)
            a = plt.hist2d(coord_d[:,1], coord_d[:,0], bins=(msz[1],msz[0]),alpha=.0,norm = LogNorm(),cmap=cm.jet)
            # clbar  = plt.colorbar()
            x = np.arange(.5,msz[1]+.5,1)
            y = np.arange(.5,msz[0]+.5,1)
            X, Y = np.meshgrid(x, y)

            area = a[0].T*50

            plt.scatter(coord_d[:,1]+.5, msz[0]-.5-coord_d[:,0], s=area, alpha=0.9,c='None',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
            plt.scatter(coord_d[:,1]+.5, msz[0]-.5-coord_d[:,0], s=area, alpha=0.2,c='b',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
            print 'hi'
#     		plt.scatter(coord[:,1], msz[0]-1-coord[:,0], s=area, alpha=0.2,c='b',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
#     		plt.scatter(X, msz[0]-1-Y, s=area, alpha=0.2,c='b',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')# 
#     		plt.scatter(X, msz[0]-1-Y, s=area, alpha=0.9,c='None',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')    		
            plt.xlim(0,msz[1])
            plt.ylim(0,msz[0])


        plt.show()

    def U_matrix(self,distance=1,row_normalized='Yes'):
        import scipy
        UD2 = self.UD2
        Umatrix = np.zeros((self.nnodes,1))
        if row_normalized=='Yes':
            vector = normalize_by(self.codebook.T, self.codebook.T, method='var').T

        else:
            vector = self.codebook
        for i in range(self.nnodes):
            codebook_i = vector[i][np.newaxis,:]
            neighborbor_ind = UD2[i][0:]<=distance
            neighborbor_codebooks = vector[neighborbor_ind]
            Umatrix[i]  = scipy.spatial.distance_matrix(codebook_i,neighborbor_codebooks).mean()
        return Umatrix.reshape(self.mapsize)

    def view_U_matrix(self,distance2=1,row_normalized='No',show_data='Yes',contooor='Yes',blob = 'No',save='No',save_dir = ''):
        import scipy
        from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
        umat = self.U_matrix(distance=distance2,row_normalized=row_normalized)
        data = getattr(self, 'data_raw')
        proj = self.project_data(data)
        msz =  getattr(self, 'mapsize')
        coord = self.ind_to_xy(proj)
    #     freq = plt.hist2d(coord[:,1], coord[:,0], bins=(msz[1],msz[0]),alpha=1.0,cmap=cm.jet)[0]
    #     plt.close()

    #     fig, ax = plt.figure()
        fig, ax= plt.subplots(1, 1)
        im = imshow(umat,cmap=cm.RdYlBu_r,alpha=1) # drawing the function
        # adding the Contour lines with labels`
        # imshow(freq[0].T,cmap=cm.jet_r,alpha=1)
        if contooor=='Yes':
            mn = np.min(umat.flatten())
            mx = np.max(umat.flatten())
            std = np.std(umat.flatten())
            md = np.median(umat.flatten())
            mx = md + 0*std
#         	mn = md
#         	umat[umat<=mn]=mn
            cset = contour(umat,np.linspace(mn,mx,15),linewidths=0.7,cmap=cm.Blues)

        if show_data=='Yes':
            plt.scatter(coord[:,1], coord[:,0], s=2, alpha=1.,c='Gray',marker='o',cmap='jet',linewidths=3, edgecolor = 'Gray')
            plt.axis('off')

        ratio = float(msz[0])/(msz[0]+msz[1])
        fig.set_size_inches((1-ratio)*15,ratio*15)
        plt.tight_layout()
        plt.subplots_adjust(hspace = .00,wspace=.000)
        sel_points = list()
        if blob=='Yes':
            from skimage.feature import blob_dog, blob_log, blob_doh
            from math import sqrt
            from skimage.color import rgb2gray
            image = 1/umat
            image_gray = rgb2gray(image)

            #'Laplacian of Gaussian'
            blobs = blob_log(image, max_sigma=5, num_sigma=4, threshold=.152)
            blobs[:, 2] = blobs[:, 2] * sqrt(2)
            imshow(umat,cmap=cm.RdYlBu_r,alpha=1)
            sel_points = list()
            for blob in blobs:
                row, col, r = blob
                c = plt.Circle((col, row), r, color='red', linewidth=2, fill=False)
                ax.add_patch(c)
                dist = scipy.spatial.distance_matrix(coord[:,:2],np.array([row,col])[np.newaxis,:])
                sel_point = dist <= r
                plt.plot(coord[:,1][sel_point[:,0]], coord[:,0][sel_point[:,0]],'.r')
                sel_points.append(sel_point[:,0])


        if save=='Yes':
            fig.savefig(save_dir, transparent=False, dpi=400)
        return sel_points,umat





    def hit_map_cluster_number(self,data=None):
        if hasattr(self, 'cluster_labels'):
            codebook = getattr(self, 'cluster_labels')
#     		print 'yesyy'
        else:
            print 'clustering based on default parameters...'
            codebook = self.cluster()
        msz =  getattr(self, 'mapsize')
        fig = plt.figure(figsize=(msz[1]/2.5,msz[0]/2.5))
        ax = fig.add_subplot(111)
#     	ax.xaxis.set_ticklabels([])
#     	ax.yaxis.set_ticklabels([])
#     	ax.grid(True,linestyle='-', linewidth=.5)


        if data == None:
            data_tr = getattr(self, 'data_raw')
            proj = self.project_data(data_tr)
            coord = self.ind_to_xy(proj)
            cents = self.ind_to_xy(np.arange(0,msz[0]*msz[1]))
            for i, txt in enumerate(codebook):
                    ax.annotate(txt, (cents[i,1],cents[i,0]),size=10, va="center")

        if data != None:
            proj = self.project_data(data)
            coord = self.ind_to_xy(proj)
            x = np.arange(.5,msz[1]+.5,1)
            y = np.arange(.5,msz[0]+.5,1)
            cents = self.ind_to_xy(proj)
#     		cents[:,1] = cents[:,1]+.2
#     		print cents.shape
            label = codebook[proj]
            for i, txt in enumerate(label):
                ax.annotate(txt, (cents[i,1],cents[i,0]),size=10, va="center")



        plt.imshow(codebook.reshape(msz[0],msz[1])[::],alpha=.5)
#     	plt.pcolor(codebook.reshape(msz[0],msz[1])[::-1],alpha=.5,cmap='jet')
        plt.show()
        return cents

    def view_map_dot(self,which_dim='all',colormap=None,cols=None,save='No',save_dir='',text_size=8):
        import matplotlib.cm as cm
        if colormap==None:
            colormap = plt.cm.get_cmap('RdYlBu_r')
        else:
            colormap = plt.cm.get_cmap(colormap)
        data = self.data_raw
        msz0, msz1 = getattr(self, 'mapsize')
        proj = self.project_data(data)
        coords = self.ind_to_xy(proj)[:,:2]
        fig = plt.figure()
        if cols==None:
            cols=8
        rows = data.shape[1]/cols+1
        if which_dim == 'all':
            dim = data.shape[0]
            rows = len(which_dim)/cols+1
            no_row_in_plot = dim/cols + 1 #6 is arbitrarily selected
            if no_row_in_plot <=1:
                no_col_in_plot = dim
            else:
                no_col_in_plot = cols
            h = .1
            w= .1
            fig = plt.figure(figsize=(no_col_in_plot*2.5*(1+w),no_row_in_plot*2.5*(1+h)))
            for i in range(data.shape[1]):
                plt.subplot(rows,cols,i+1)

                #this uses the colors uniquely for each record, while in normal views, it is based on the values within each dimensions.
                #This is important when we are dealing with time series. Where we don't want to normalize colors within each time period, rather we like to see th
                #the patterns of each data records in time.
                mn = np.min(data[:,:],axis=1)
                mx = np.max(data[:,:],axis=1)
# 				print mx.shape
# 				print coords.shape
                for j in range(data.shape[0]):
                    sc = plt.scatter(coords[j,1],self.mapsize[0]-1-coords[j,0],c=data[j,which_dim[i]],vmax=mx[j],vmin=mn[j],s=90,marker='.',edgecolor='None', cmap=colormap ,alpha=1)



                mn = data# a[:,i].min()
# 				mx = data[:,i].max()
# 				plt.scatter(coords[:,1],self.mapsize[0]-1-coords[:,0],c=data[:,i],vmax=mx,vmin=mn,s=180,marker='.',edgecolor='None', cmap=colormap ,alpha=1)





                eps = .0075
                plt.xlim(0-eps,self.mapsize[1]-1+eps)
                plt.ylim(0-eps,self.mapsize[0]-1+eps)
                plt.axis('off')
                plt.title(self._component_names[0][i])
                font = {'size'   : text_size}
                plt.rc('font', **font)
                plt.axis('on')
                plt.xticks([])
                plt.yticks([])
        else:
            dim = len(which_dim)
            rows = len(which_dim)/cols+1
            no_row_in_plot = dim/cols + 1 #6 is arbitrarily selected
            if no_row_in_plot <=1:
                no_col_in_plot = dim
            else:
                no_col_in_plot = cols
            h = .1
            w= .1
            fig = plt.figure(figsize=(no_col_in_plot*2.5*(1+w),no_row_in_plot*2.5*(1+h)))
            for i in range(len(which_dim)):
                plt.subplot(rows,cols,i+1)
                mn = np.min(data[:,:],axis=1)
                mx = np.max(data[:,:],axis=1)
# 				print mx.shape
# 				print coords.shape
                for j in range(data.shape[0]):
                    sc = plt.scatter(coords[j,1],self.mapsize[0]-1-coords[j,0],c=data[j,which_dim[i]],vmax=mx[j],vmin=mn[j],s=90,marker='.',edgecolor='None', cmap=colormap ,alpha=1)




# 				mn = data[:,which_dim[i]].min()
# 				mx = data[:,which_dim[i]].max()
# 				plt.scatter(coords[:,1],self.mapsize[0]-1-coords[:,0],c=data[:,which_dim[i]],vmax=mx,vmin=mn,s=180,marker='.',edgecolor='None', cmap=colormap ,alpha=1)




                eps = .0075
                plt.xlim(0-eps,self.mapsize[1]-1+eps)
                plt.ylim(0-eps,self.mapsize[0]-1+eps)
                plt.axis('off')
                plt.title(self._component_names[0][which_dim[i]])
                font = {'size'   : text_size}
                plt.rc('font', **font)
                plt.axis('on')
                plt.xticks([])
                plt.yticks([])

        plt.tight_layout()
        # 		plt.colorbar(sc,ticks=np.round(np.linspace(mn,mx,5),decimals=1),shrink=0.6)
        plt.subplots_adjust(hspace = .16,wspace=.05)
# 		fig.set_size_inches(msz0/2,msz1/2)
# 		fig = plt.figure(figsize=(msz0/2,msz1/2))
        if save=='Yes':
            if save_dir != 'empty':
                fig.savefig(save_dir, transparent=False, dpi=200)
            else:
                add = '/Users/itadmin/Desktop/SOM_dot.png'
                print 'save directory: ', add
                fig.savefig(add, transparent=False, dpi=200)

            plt.close(fig)



    def predict_Probability(self, data, Target, K =5):
        # here it is assumed that Target is the last column in the codebook #and data has dim-1 columns
        codebook = getattr(self, 'codebook')
        data_raw = getattr(self,'data_raw')
        dim = codebook.shape[1]
        ind = np.arange(0,dim)
        indX = ind[ind != Target]
        X = codebook[:,indX]
        Y = codebook[:,Target]
        n_neighborbors = K
        clf = neighborbors.KNeighborsRegressor(n_neighborbors, weights = 'distance')
        clf.fit(X, Y)
        # the codebook values are all normalized
        #we can normalize the input data based on mean and std of original data
        dimdata = data.shape[1]
        if dimdata == dim: 
            data[:,Target] == 0   
            data = normalize_by(data_raw, data, method='var')
            data = data[:,indX]
        elif dimdata == dim -1:          
            data = normalize_by(data_raw[:,indX], data, method='var')       
            #data = normalize(data, method='var')
        weights,ind= clf.kneighborbors(data, n_neighborbors=K, return_distance=True)    
        weights = 1./weights
        sum_ = np.sum(weights,axis=1)
        weights = weights/sum_[:,np.newaxis]
        labels = np.sign(codebook[ind,Target])
        labels[labels>=0]=1

        #for positives
        pos_prob = labels.copy()
        pos_prob[pos_prob<0]=0
        pos_prob = pos_prob*weights
        pos_prob = np.sum(pos_prob,axis=1)[:,np.newaxis]

        #for negatives
        neg_prob = labels.copy()
        neg_prob[neg_prob>0]=0
        neg_prob = neg_prob*weights*-1
        neg_prob = np.sum(neg_prob,axis=1)[:,np.newaxis]

        #Predicted_values = clf.predict(data)
        #Predicted_values = denormalize_by(data_raw[:,Target], Predicted_values)
        return np.concatenate((pos_prob,neg_prob),axis=1)


    def node_Activation(self, data, wt= 'distance',Target = None):
        """
        ‘uniform’
        """

        if Target == None:
            codebook = getattr(self, 'codebook')
            data_raw = getattr(self,'data_raw')
            clf = neighborbors.KNeighborsClassifier(n_neighborbors = getattr(self, 'nnodes'))
            labels = np.arange(0,codebook.shape[0])
            clf.fit(codebook, labels)
            # the codebook values are all normalized
            #we can normalize the input data based on mean and std of original data
            data = normalize_by(data_raw, data, method='var')
            weights,ind= clf.kneighborbors(data)



            ##Softmax function
            weights = 1./weights
#          	S_  = np.sum(np.exp(weights),axis=1)[:,np.newaxis]
#          	weights = np.exp(weights)/S_



        return weights , ind	



        # 



    def view_2d(self, text_size,which_dim='all', what = 'codebook'):
        msz0, msz1 = getattr(self, 'mapsize')
        if what == 'codebook':
            if hasattr(self, 'codebook'):
                codebook = getattr(self, 'codebook')
                data_raw = getattr(self,'data_raw')
                codebook = denormalize_by(data_raw, codebook)
            else:
                print 'first initialize codebook'
            if which_dim == 'all':
                dim = getattr(self, 'dim')
                indtoshow = np.arange(0,dim).T
                ratio = float(dim)/float(dim)
                ratio = np.max((.35,ratio))
                sH, sV = 16,16*ratio*1
                plt.figure(figsize=(sH,sV))
            elif type(which_dim) == int:
                dim = 1
                indtoshow = np.zeros(1)
                indtoshow[0] = int(which_dim)
                sH, sV = 6,6
                plt.figure(figsize=(sH,sV))
            elif type(which_dim) == list:
                max_dim = codebook.shape[1]
                dim = len(which_dim)
                ratio = float(dim)/float(max_dim)
                #print max_dim, dim, ratio
                ratio = np.max((.35,ratio))
                indtoshow = np.asarray(which_dim).T
                sH, sV = 16,16*ratio*1
                plt.figure(figsize=(sH,sV))

            no_row_in_plot = dim/6 + 1 #6 is arbitrarily selected
            if no_row_in_plot <=1:
                no_col_in_plot = dim
            else:
                no_col_in_plot = 6

            axisNum = 0
            compname = getattr(self, 'compname')
            norm = matplotlib.colors.normalize(vmin = np.mean(codebook.flatten())-1*np.std(codebook.flatten()), vmax = np.mean(codebook.flatten())+1*np.std(codebook.flatten()), clip = True)
            while axisNum <dim:
                axisNum += 1
                ax = plt.subplot(no_row_in_plot, no_col_in_plot, axisNum)
                ind = int(indtoshow[axisNum-1])
                mp = codebook[:,ind].reshape(msz0, msz1)
                pl = plt.pcolor(mp[::-1],norm = norm)
    #             pl = plt.imshow(mp[::-1])
                plt.title(compname[0][ind])
                font = {'size'   : text_size*sH/no_col_in_plot}
                plt.rc('font', **font)
                plt.axis('off')
                plt.axis([0, msz0, 0, msz1])
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                plt.colorbar(pl)
            plt.show()


    def view_2d_Pack(self, text_size,which_dim='all', what = 'codebook',save='No', grid='Yes', save_dir = 'empty',text='Yes',CMAP='None',col_sz=None):
        import matplotlib.cm as cm
        msz0, msz1 = getattr(self, 'mapsize')
        if CMAP=='None':
            CMAP= cm.RdYlBu_r
    #     	CMAP = cm.jet
        if what == 'codebook':
            if hasattr(self, 'codebook'):
                codebook = getattr(self, 'codebook')
                data_raw = getattr(self,'data_raw')
                codebook = denormalize_by(data_raw, codebook)
            else:
                print 'first initialize codebook'
            if which_dim == 'all':
                dim = getattr(self, 'dim')
                indtoshow = np.arange(0,dim).T
                ratio = float(dim)/float(dim)
                ratio = np.max((.35,ratio))
                sH, sV = 16,16*ratio*1
    #             plt.figure(figsize=(sH,sV))
            elif type(which_dim) == int:
                dim = 1
                indtoshow = np.zeros(1)
                indtoshow[0] = int(which_dim)
                sH, sV = 6,6
    #             plt.figure(figsize=(sH,sV))
            elif type(which_dim) == list:
                max_dim = codebook.shape[1]
                dim = len(which_dim)
                ratio = float(dim)/float(max_dim)
                #print max_dim, dim, ratio
                ratio = np.max((.35,ratio))
                indtoshow = np.asarray(which_dim).T
                sH, sV = 16,16*ratio*1
    #             plt.figure(figsize=(sH,sV))

    #         plt.figure(figsize=(7,7))
            no_row_in_plot = dim/col_sz + 1 #6 is arbitrarily selected
            if no_row_in_plot <=1:
                no_col_in_plot = dim
            else:
                no_col_in_plot = col_sz

            axisNum = 0
            compname = getattr(self, 'compname')
            h = .1
            w= .1
            fig = plt.figure(figsize=(no_col_in_plot*2.5*(1+w),no_row_in_plot*2.5*(1+h)))
    #         print no_row_in_plot, no_col_in_plot
            norm = matplotlib.colors.Normalize(vmin = np.median(codebook.flatten())-1.5*np.std(codebook.flatten()), vmax = np.median(codebook.flatten())+1.5*np.std(codebook.flatten()), clip = False)

            DD = pd.Series(data = codebook.flatten()).describe(percentiles=[.03,.05,.1,.25,.3,.4,.5,.6,.7,.8,.9,.95,.97])
            norm = matplotlib.colors.Normalize(vmin = DD.ix['3%'], vmax = DD.ix['97%'], clip = False)

            while axisNum <dim:
                axisNum += 1

                ax = fig.add_subplot(no_row_in_plot, no_col_in_plot, axisNum)
                ind = int(indtoshow[axisNum-1])
                mp = codebook[:,ind].reshape(msz0, msz1)

                if grid=='Yes':
                    pl = plt.pcolor(mp[::-1],cmap=CMAP)
                elif grid=='No':
                    plt.imshow(mp[::-1],norm = None,cmap=CMAP)
    #             	plt.pcolor(mp[::-1])
                    plt.axis('off')

                if text=='Yes':
                    plt.title(compname[0][ind])
                    font = {'size'   : text_size}
                    plt.rc('font', **font)
                plt.axis([0, msz0, 0, msz1])
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.xaxis.set_ticks([i for i in range(0,msz1)])
                ax.yaxis.set_ticks([i for i in range(0,msz0)])
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
                ax.grid(True,linestyle='-', linewidth=0.5,color='k')
    #             plt.grid()
    #             plt.colorbar(pl)
    #         plt.tight_layout()
            plt.subplots_adjust(hspace = h,wspace=w)
        if what == 'cluster':
            if hasattr(self, 'cluster_labels'):
                codebook = getattr(self, 'cluster_labels')
            else:
                print 'clustering based on default parameters...'
                codebook = self.cluster()
            h = .2
            w= .001
            fig = plt.figure(figsize=(msz0/2,msz1/2))

            ax = fig.add_subplot(1, 1, 1)
            mp = codebook[:].reshape(msz0, msz1)
            if grid=='Yes':
                plt.imshow(mp[::-1],cmap=CMAP)
    #         pl = plt.pcolor(mp[::-1],cmap=CMAP)
            elif grid=='No':
                plt.imshow(mp[::-1],cmap=CMAP)
    #             plt.pcolor(mp[::-1])
                plt.axis('off')

            if text=='Yes':
                plt.title('clusters')
                font = {'size'   : text_size}
                plt.rc('font', **font)
            plt.axis([0, msz0, 0, msz1])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.xaxis.set_ticks([i for i in range(0,msz1)])
            ax.yaxis.set_ticks([i for i in range(0,msz0)])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.grid(True,linestyle='-', linewidth=0.5,color='k')
            plt.subplots_adjust(hspace = h,wspace=w)


        if save == 'Yes':

                if save_dir != 'empty':
    #         		print save_dir
                    fig.savefig(save_dir,bbox_inches='tight', transparent=False, dpi=200)
                else:
    #         		print save_dir
                    add = '/Users/itadmin/Desktop/SOM.png'
                    fig.savefig(add,bbox_inches='tight', transparent=False, dpi=200)

                plt.close(fig)





    def view_1d(self, text_size, which_dim ='all', what = 'codebook'):
        msz0, msz1 = getattr(self, 'mapsize')
        if what == 'codebook':
            if hasattr(self, 'codebook'):
                codebook = getattr(self, 'codebook')
                data_raw = getattr(self,'data_raw')
                codebook = denormalize_by(data_raw, codebook)
            else:
                print 'first initialize codebook'
            if which_dim == 'all':
                dim = getattr(self, 'dim')
                indtoshow = np.arange(0,dim).T
                ratio = float(dim)/float(dim)
                ratio = np.max((.35,ratio))
                sH, sV = 16,16*ratio*1
                plt.figure(figsize=(sH,sV))
            elif type(which_dim) == int:
                dim = 1
                indtoshow = np.zeros(1)
                indtoshow[0] = int(which_dim)
                sH, sV = 6,6
                plt.figure(figsize=(sH,sV))
            elif type(which_dim) == list:
                max_dim = codebook.shape[1]
                dim = len(which_dim)
                ratio = float(dim)/float(max_dim)
                #print max_dim, dim, ratio
                ratio = np.max((.35,ratio))
                indtoshow = np.asarray(which_dim).T
                sH, sV = 16,16*ratio*1
                plt.figure(figsize=(sH,sV))

            no_row_in_plot = dim/6 + 1 #6 is arbitrarily selected
            if no_row_in_plot <=1:
                no_col_in_plot = dim
            else:
                no_col_in_plot = 6

            axisNum = 0
            compname = getattr(self, 'compname')
            while axisNum < dim:
                axisNum += 1
                ax = plt.subplot(no_row_in_plot, no_col_in_plot, axisNum)
                ind = int(indtoshow[axisNum-1])
                mp = codebook[:,ind]
                plt.plot(mp,'-k',linewidth = 0.8)
                #pl = plt.pcolor(mp[::-1])
                plt.title(compname[0][ind])
                font = {'size'   : text_size*sH/no_col_in_plot}
                plt.rc('font', **font)
                #plt.axis('off')
                #plt.axis([0, msz0, 0, msz1])
                #ax.set_yticklabels([])
                #ax.set_xticklabels([])
                #plt.colorbar(pl)
            plt.show()






    def lininit(self):
        #X = UsigmaWT
        #XTX = Wsigma^2WT
        #T = XW = Usigma #Transformed by W EigenVector, can be calculated by
        #multiplication PC matrix by eigenval too
        #Furthe, we can get lower ranks by using just few of the eigen vevtors
        #T(2) = U(2)sigma(2) = XW(2) ---> 2 is the number of selected eigenvectors
        # This is how we initialize the map, just by using the first two first eigen vals and eigenvectors
        # Further, we create a linear combination of them in the new map by giving values from -1 to 1 in each
        #Direction of SOM map
        # it shoud be noted that here, X is the covariance matrix of original data

        msize =  getattr(self, 'mapsize')
        rows = msize[0]
        cols = msize[1]
        nnodes = getattr(self, 'nnodes')

        if np.min(msize)>1:
            coord = np.zeros((nnodes, 2))
            for i in range(0,nnodes):
                coord[i,0] = int(i/cols) #x
                coord[i,1] = int(i%cols) #y
            mx = np.max(coord, axis = 0)
            mn = np.min(coord, axis = 0)
            coord = (coord - mn)/(mx-mn)
            coord = (coord - .5)*2
            data = getattr(self, 'data')
            me = np.mean(data, 0)
            data = (data - me)
            codebook = np.tile(me, (nnodes,1))
            pca = RandomizedPCA(n_components=2) #Randomized PCA is scalable
            #pca = PCA(n_components=2)
            pca.fit(data)
            eigvec = pca.components_
            eigval = pca.explained_variance_
            norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
            eigvec = ((eigvec.T/norms)*eigval).T; eigvec.shape

            for j in range(nnodes):
                for i in range(eigvec.shape[0]):
                    codebook[j,:] = codebook[j, :] + coord[j,i]*eigvec[i,:]
            return np.around(codebook, decimals = 6)
        elif np.min(msize) == 1:
            coord = np.zeros((nnodes, 1))
            for i in range(0,nnodes):
                #coord[i,0] = int(i/cols) #x
                coord[i,0] = int(i%cols) #y
            mx = np.max(coord, axis = 0)
            mn = np.min(coord, axis = 0)
            #print coord

            coord = (coord - mn)/(mx-mn)
            coord = (coord - .5)*2
            #print coord
            data = getattr(self, 'data')
            me = np.mean(data, 0)
            data = (data - me)
            codebook = np.tile(me, (nnodes,1))
            pca = RandomizedPCA(n_components=1) #Randomized PCA is scalable
            #pca = PCA(n_components=2)
            pca.fit(data)
            eigvec = pca.components_
            eigval = pca.explained_variance_
            norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
            eigvec = ((eigvec.T/norms)*eigval).T; eigvec.shape

            for j in range(nnodes):
                for i in range(eigvec.shape[0]):
                    codebook[j,:] = codebook[j, :] + coord[j,i]*eigvec[i,:]
            return np.around(codebook, decimals = 6)



def normalize(data, method='var'):
    #methods  = ['var','range','log','logistic','histD','histC']
    #status = ['done', 'undone']
    me = np.mean(data, axis = 0)
    st = np.std(data, axis = 0)
    if method == 'var':
        me = np.mean(data, axis = 0)
        st = np.std(data, axis = 0)
        n_data = (data-me)/st
        return n_data

def normalize_by(data_raw, data, method='var'):
    #methods  = ['var','range','log','logistic','histD','histC']
    #status = ['done', 'undone']
    # to have the mean and std of the original data, by which SOM is trained
    me = np.mean(data_raw, axis = 0)
    st = np.std(data_raw, axis = 0)
    if method == 'var':
        n_data = (data-me)/st
        return n_data

def denormalize_by(data_by, n_vect, n_method = 'var'):
    #based on the normalization
    if n_method == 'var':
        me = np.mean(data_by, axis = 0)
        st = np.std(data_by, axis = 0)
        vect = n_vect* st + me
        return vect
    else:
        print 'data is not normalized before'
            return n_vect

def l(a,b):
    c = np.zeros(b.shape)
    c[a-b >=0] = 1
    return c

##Function to show hits
#som_labels = sm.project_data(Tr_data)
#S = pd.dataFrame(data=som_labels,columns= ['label'])
#a = S['label'].value_counts()
#a = a.sort_index()
#a = pd.dataFrame(data=a.values, index=a.index,columns=['label'])
#d = pd.dataFrame(data= range(msz0*msz1),columns=['node_ID'])
#c  = d.join(a,how='outer')
#c.fillna(value=0,inplace=True)
#hits = c.values[:,1]
#hits = hits
#nodeID = np.arange(msz0*msz1)
#c_bmu = nodeID%msz1
#r_bmu = msz0 - nodeID/msz1
#fig, ax = plt.subplots()
#plt.axis([0, msz0, 0, msz1])
#ax.scatter(r_bmu, c_bmu, s=hits/2)
