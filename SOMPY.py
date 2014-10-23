# -*- coding: utf-8 -*-


# Vahid Moosavi 2014 10 23 9:04 pm
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
import tables as tb
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
from sklearn import neighbors
from matplotlib.colors import LogNorm
from matplotlib import cm
import matplotlib
import pandas as pd




class SOM(object):
    
    def __init__(self,name,Data, mapsize = None, norm_method = 'var',initmethod = 'pca'):
        """
        name and data
        """
        self.name = name
        self.data_raw = Data   
        if norm_method == 'var':    
            Data = normalize(Data, method=norm_method)
            self.data = Data
        
        else:
            self.data = Data
        self.dim = Data.shape[1]
        self.dlen = Data.shape[0]
        self.set_topology(mapsize = mapsize)
        self.set_algorithm(initmethod = initmethod)
        self.calc_map_dist()
        
        #Slow for large data sets
        #self.set_data_labels()

        
        
    #set SOM topology
    def set_topology(self, mapsize = None, mapshape = 'planar', lattice = 'rect', mask = None, compname = None):
        """
        all_mapshapes = ['planar','toroid','cylinder']
        all_lattices = ['hexa','rect']
        """
        self.mapshape = mapshape
        self.lattice = lattice
        
        #to set mask
        if mask == None: 
            self.mask = np.ones([1,self.dim])
        else:
            self.mask = mask
        
    
        #to set map size
        if mapsize == None:    
            tmp = int(round(np.sqrt(self.dlen)))
            self.nnodes = tmp
            self.mapsize = [int(3./5*self.nnodes), int(2./5*self.nnodes)]
        else:
            if len(mapsize)==2:
                if np.min(mapsize) == 1:
                    self.mapsize = [1, np.max(mapsize)]
                else:    
                    self.mapsize = mapsize
            elif len(mapsize) == 1:
                #s =  int (mapsize[0]/2)
                self.mapsize = [1 ,mapsize[0]]
                print 'input was considered as node numbers'
                print 'map size is [{0},{1}]'.format(s,s) 
            self.nnodes = self.mapsize[0]*self.mapsize[1]
                
        # to set component names
        if compname == None:    
            try:
                cc = list()
                for i in range(0,self.dim):
                    cc.append ('Variable-'+ str(i+1))
                    self.compname = np.asarray(cc)[np.newaxis,:]
            except:
                pass
                print 'no data yet: plesae first set trainign data to the SOM'
        else:
            try:
                dim =  getattr(self,'dim')
                if  len(compname) == dim:
                    self.compname = np.asarray(compname)[np.newaxis,:]
                else:
                    print 'compname should have the same size'
            except:
                pass
                print 'no data yet: plesae first set trainign data to the SOM'
     
     
    #Set labels of the training data
    # it should be in the format of a list of strings
    def set_data_labels(self, dlabel = None):
        if dlabel == None:    
            try:
                dlen =  (getattr(self,'dlen'))
                cc = list()
                for i in range(0,dlen):
                    cc.append ('dlabel-'+ str(i))
                    self.dlabel = np.asarray(cc)[:, np.newaxis]
            except:
                pass
                print 'no data yet: plesae first set trainign data to the SOM'
        else: 
            try:
                dlen =  (getattr(self,'dlen'))
                if dlabel.shape == (1,dlen):
                    self.dlabel = dlabel.T#[:,np.newaxis]
                elif dlabel.shape == (dlen,1):
                    self.dlabel = dlabel
                elif dlabel.shape == (dlen,):
                    self.dlabel = dlabel[:, np.newaxis]
                else:
                    print 'wrong lable format' 
            except:
                pass
                print 'no data yet: plesae first set trainign data to the SOM'
         
    #calculating the grid distance, which will be called during the training steps
    #currently just works for planar grids
    def calc_map_dist(self):
        cd = getattr(self, 'nnodes')
        UD2 = np.zeros((cd, cd))
        for i in range(cd):
            UD2[i,:] = grid_dist(self, i).reshape(1,cd)
        self.UD2 =  UD2
    
    
    
    def set_algorithm(self, initmethod = 'pca', algtype = 'batch', neighborhoodmethod = 'gaussian', alfatype = 'inv', alfaini = .5, alfafinal = .005):
        """
        initmethod = ['random', 'pca']
        algos = ['seq','batch']
        all_neigh = ['gaussian','manhatan','bubble','cut_gaussian','epanechicov' ]
        alfa_types = ['linear','inv','power']
        
        """
        self.initmethod = initmethod
        self.algtype = algtype
        self.alfaini = alfaini
        self.alfafinal = alfafinal
        self.neigh = neighborhoodmethod
        
    
    ###################################
    #visualize map
    def view_map(self, what = 'codebook', which_dim = 'all', pack= 'Yes', text_size = 2.8,save='No', save_dir = 'empty',grid='No',text='Yes'):
        
        mapsize = getattr(self, 'mapsize')
        if np.min(mapsize) >1:
        	if pack == 'No':
        		view_2d(self, text_size, which_dim = which_dim, what = what)
        	else:
#         		print 'hi' 
        		view_2d_Pack(self, text_size, which_dim = which_dim,what = what,save = save, save_dir = save_dir, grid=grid,text=text)
        
        elif np.min(mapsize) == 1:
             view_1d(self, text_size, which_dim = which_dim, what = what)   

    ################################################################################
    # Initialize map codebook: Weight vectors of SOM
    def init_map(self):
        dim = 0
        n_nod = 0
        if  getattr(self, 'initmethod')=='random':
            #It produces random values in the range of min- max of each dimension based on a uniform distribution
            mn = np.tile(np.min(getattr(self,'data'), axis =0), (getattr(self, 'nnodes'),1))
            mx = np.tile(np.max(getattr(self,'data'), axis =0), (getattr(self, 'nnodes'),1))
            setattr(self, 'codebook', mn + (mx-mn)*(np.random.rand(getattr(self, 'nnodes'), getattr(self, 'dim'))))
        elif getattr(self, 'initmethod') == 'pca':
            codebooktmp = lininit(self) #it is based on two largest eigenvalues of correlation matrix
            setattr(self, 'codebook', codebooktmp)
        else:
            print 'please select a corect initialization method'
            print 'set a correct one in SOM. current SOM.initmethod:  ', getattr(self, 'initmethod')
            print "possible init methods:'random', 'pca'"
     
    
    #Main loop of training
    def train(self, trainlen = None, n_job = 1, shared_memory = 'no',verbose='on'):
        t0 = time()
        data = getattr(self, 'data')
        nnodes = getattr(self, 'nnodes')
        dlen = getattr(self, 'dlen')
        dim = getattr(self, 'dim')
        mapsize = getattr(self, 'mapsize')
        mem = np.log10(dlen*nnodes*dim)
        #print 'data len is %d and data dimension is %d' % (dlen, dim)
        #print 'map size is %d, %d' %(mapsize[0], mapsize[1])
        #print 'array size in log10 scale' , mem 
        #print 'nomber of jobs in parallel: ', n_job 
        #######################################
        #initialization
        if verbose=='on':
            print 
            print 'initialization method = %s, initializing..' %getattr(self, 'initmethod')
            print
            t0 = time()
        self.init_map()
        if verbose=='on':
            print 'initialization done in %f seconds' % round(time()-t0 , 3 )
        
        ########################################
        #rough training
        if verbose=='on':
            print
        batchtrain(self, njob = n_job, phase = 'rough', shared_memory = 'no',verbose=verbose)
        if verbose=='on':
            print
        #######################################
        #Finetuning
        if verbose=='on':
            print
        batchtrain(self, njob = n_job, phase = 'finetune', shared_memory = 'no',verbose=verbose)
        err = np.mean(getattr(self, 'bmu')[1])
        if verbose=='on': 
#         or verbose == 'off':
#             print
            ts = round(time() - t0, 3)
            print
            print "Total time elapsed: %f secodns" %ts
            print "final quantization error: %f" %err 
    
    #to project a data set to a trained SOM and find the index of bmu 
    #It is based on nearest neighborhood search module of scikitlearn, but it is not that fast.
    def project_data(self, data):
        codebook = getattr(self, 'codebook')
        data_raw = getattr(self,'data_raw')
        clf = neighbors.KNeighborsClassifier(n_neighbors = 1)
        labels = np.arange(0,codebook.shape[0])
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
        n_neighbors = K
        clf = neighbors.KNeighborsRegressor(n_neighbors, weights = wt)
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
		n_neighbors = K
		clf = neighbors.KNeighborsRegressor(n_neighbors, weights = wt)
		clf.fit(X_train, Y_train)
		# the codebook values are all normalized
		#we can normalize the input data based on mean and std of original data
		X_test = normalize_by(data_raw[:,:Target], X_test, method='var')
		Predicted_values = clf.predict(X_test)
		Predicted_values = denormalize_by(data_raw[:,Target], Predicted_values)
		return Predicted_values

    
    def find_K_nodes(self, data, K =5):
        from sklearn.neighbors import NearestNeighbors
        # we find the k most similar nodes to the input vector
        codebook = getattr(self, 'codebook')
        neigh = NearestNeighbors(n_neighbors = K)
        neigh.fit(codebook) 
        data_raw = getattr(self,'data_raw')
        # the codebook values are all normalized
        #we can normalize the input data based on mean and std of original data
        data = normalize_by(data_raw, data, method='var')
        return neigh.kneighbors(data)  
        
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
    	km= clust.KMeans(n_clusters=8)
    	labels = km.fit_predict(denormalize_by(self.data_raw, self.codebook, n_method = 'var'))
    	setattr(self,'cluster_labels',labels)
    	return labels
    

    
    
    def hit_map(self,data=None):
    	#First Step: show the hitmap of all the training data
    	
#     	print 'None'
    	data_tr = getattr(self, 'data_raw')
    	proj = self.project_data(data_tr)
    	msz =  getattr(self, 'mapsize')
    	coord = self.ind_to_xy(proj)
    	fig = plt.figure(figsize=(msz[1]/2,msz[0]/2))
    	ax = fig.add_subplot(111)
    	ax.xaxis.set_ticks([i for i in range(0,msz[1])])
    	ax.yaxis.set_ticks([i for i in range(0,msz[0])])
    	ax.xaxis.set_ticklabels([])
    	ax.yaxis.set_ticklabels([])
    	ax.grid(True,linestyle='-', linewidth=.5)
    	a = plt.hist2d(coord[:,1], coord[:,0], bins=(msz[1],msz[0]),alpha=.0,norm = LogNorm(),cmap=cm.jet)
    	# clbar  = plt.colorbar()
    	x = np.arange(.5,msz[1]+.5,1)
    	y = np.arange(.5,msz[0]+.5,1)
    	X, Y = np.meshgrid(x, y)
    	area = a[0].T*12
    	plt.scatter(X, Y, s=area, alpha=0.2,c='b',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
    	plt.scatter(X, Y, s=area, alpha=0.9,c='None',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
    	plt.xlim(0,msz[1])
    	plt.ylim(0,msz[0])

    	if data != None:
    		proj = self.project_data(data)
    		msz =  getattr(self, 'mapsize')
    		coord = self.ind_to_xy(proj)
    		a = plt.hist2d(coord[:,1], coord[:,0], bins=(msz[1],msz[0]),alpha=.0,norm = LogNorm(),cmap=cm.jet)
    		# clbar  = plt.colorbar()
    		x = np.arange(.5,msz[1]+.5,1)
    		y = np.arange(.5,msz[0]+.5,1)
    		X, Y = np.meshgrid(x, y)
    		area = a[0].T*50
    		plt.scatter(X, Y, s=area, alpha=0.2,c='b',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
    		plt.scatter(X, Y, s=area, alpha=0.9,c='None',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')    		
    		plt.xlim(0,msz[1])
    		plt.ylim(0,msz[0])
    	
    	
    	plt.show()
    
    
    
    def hit_map_cluster_number(self,data=None):
    	if hasattr(self, 'cluster_labels'):
    		codebook = getattr(self, 'cluster_labels')
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
    		print cents.shape
    		label = codebook[proj]
    		for i, txt in enumerate(label):
	    		ax.annotate(txt, (cents[i,1],cents[i,0]),size=10, va="center")
    	
    	
    	
    	plt.imshow(codebook.reshape(msz[0],msz[1])[::],alpha=.5)
#     	plt.pcolor(codebook.reshape(msz[0],msz[1])[::-1],alpha=.5,cmap='jet')
    	plt.show()


    
    
    def predict_Probability(self, data, Target, K =5):
        # here it is assumed that Target is the last column in the codebook #and data has dim-1 columns
        codebook = getattr(self, 'codebook')
        data_raw = getattr(self,'data_raw')
        dim = codebook.shape[1]
        ind = np.arange(0,dim)
        indX = ind[ind != Target]
        X = codebook[:,indX]
        Y = codebook[:,Target]
        n_neighbors = K
        clf = neighbors.KNeighborsRegressor(n_neighbors, weights = 'distance')
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
        weights,ind= clf.kneighbors(data, n_neighbors=K, return_distance=True)    
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
        	clf = neighbors.KNeighborsClassifier(n_neighbors = getattr(self, 'nnodes'))
        	labels = np.arange(0,codebook.shape[0])
        	clf.fit(codebook, labels)
	    	# the codebook values are all normalized
    		#we can normalize the input data based on mean and std of original data
        	data = normalize_by(data_raw, data, method='var')
        	weights,ind= clf.kneighbors(data)    
        	
        	weights = 1./weights
        	
        	##Softmax function 
        	S_  = np.sum(np.exp(weights),axis=1)[:,np.newaxis]
        	weights = np.exp(weights)/S_
        	
        	
        	
#         	sum_ = np.sum(weights,axis=1)
#         	weights = weights/sum_[:,np.newaxis]
        	return weights , ind	
        	
        	
        
        # 
    
    
    def para_bmu_find(self, x, y, njb = 1):
        dlen = x.shape[0]
        Y2 = None
        Y2 = np.einsum('ij,ij->i', y, y)
        bmu = None
        b = None
        #here it finds BMUs for chunk of data in parallel
        t_temp  = time()
        b  = Parallel(n_jobs=njb, pre_dispatch='3*n_jobs')(delayed(chunk_based_bmu_find)\
        (self, x[i*dlen // njb:min((i+1)*dlen // njb, dlen)],y, Y2) \
        for i in xrange(njb))
        
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
        nnodes = getattr(self, 'nnodes')
        dlen = getattr(self ,'dlen')
        dim = getattr(self, 'dim')
	New_Codebook = np.empty((nnodes, dim))
        inds = bmu[0].astype(int)
        row = inds
        col = np.arange(dlen)
        val = np.tile(1,dlen)
        P = csr_matrix( (val,(row,col)), shape=(nnodes,dlen) )
        S  = np.empty((nnodes, dim))
        S = P.dot(training_data)
        #assert( S.shape == (nnodes, dim))
        #assert( H.shape == (nnodes, nnodes))
        
        # H has nnodes*nnodes and S has nnodes*dim  ---> Nominator has nnodes*dim
        #print Nom
        Nom = np.empty((nnodes,nnodes))
        Nom =  H.T.dot(S)
        #assert( Nom.shape == (nnodes, dim))
        nV = np.empty((1,nnodes))
        nV = P.sum(axis = 1).reshape(1, nnodes)
        #assert(nV.shape == (1, nnodes))
        Denom = np.empty((nnodes,1))
        Denom = nV.dot(H.T).reshape(nnodes, 1)
        #assert( Denom.shape == (nnodes, 1))
        New_Codebook = np.divide(Nom, Denom)
        Nom = None
        Denom = None
        #assert (New_Codebook.shape == (nnodes,dim))
        #setattr(som, 'codebook', New_Codebook)
        return np.around(New_Codebook, decimals = 6)    
                        
# we will call this function in parallel for different number of jobs
def chunk_based_bmu_find(self, x, y, Y2):
    dim = x.shape[1]
    dlen = x.shape[0]
    nnodes = y.shape[0]
    bmu = np.empty((dlen,2))
    #it seems that smal batches for large dlen is really faster: 
    # that is because of ddata in loops and n_jobs. for large data it slows down due to memory needs in parallel
    blen = min(50,dlen) 
    i0 = 0;
    d = None
    t = time()
    while i0+1<=dlen:
        Low =  (i0)
        High = min(dlen,i0+blen)
        i0 = i0+blen      
        ddata = x[Low:High+1]
        d = np.dot(y, ddata.T)
        d *= -2
        d += Y2.reshape(nnodes,1)
        bmu[Low:High+1,0] = np.argmin(d, axis = 0)
        bmu[Low:High+1,1] = np.min(d, axis = 0) 
        del ddata
        d = None
    return bmu
    
#Batch training which is called for rought training as well as finetuning
def batchtrain(self, njob = 1, phase = None, shared_memory = 'no', verbose='on'):
    t0 = time()
    nnodes = getattr(self, 'nnodes')
    dlen = getattr(self, 'dlen')
    dim = getattr(self, 'dim')
    mapsize = getattr(self, 'mapsize')
    

    #############################################
    # seting the parameters
    initmethod = getattr(self,'initmethod')
    mn = np.min(mapsize)
    if mn == 1:
        mpd = float(nnodes*10)/float(dlen)
    else:
        mpd = float(nnodes)/float(dlen)
    
    ms = max(mapsize[0],mapsize[1])
    if mn == 1:
        ms = ms/5.
    #Based on somtoolbox, Matlab
    #case 'train',    sTrain.trainlen = ceil(50*mpd);
    #case 'rough',    sTrain.trainlen = ceil(10*mpd); 
    #case 'finetune', sTrain.trainlen = ceil(40*mpd);
    if phase == 'rough':
        #training length
        trainlen = int(np.ceil(10*mpd))
        #radius for updating
        if initmethod == 'random':
#         	trainlen = int(np.ceil(15*mpd))
        	radiusin = max(1, np.ceil(ms/2.))
        	radiusfin = max(1, radiusin/8.)
        elif initmethod == 'pca':
            radiusin = max(1, np.ceil(ms/8.))
            radiusfin = max(1, radiusin/4.)
    elif phase == 'finetune':
        #train lening length
        trainlen = int(np.ceil(40*mpd))
        #radius for updating
        if initmethod == 'random':
#             trainlen = int(np.ceil(50*mpd))
            radiusin = max(1, ms/8.) #from radius fin in rough training  
            radiusfin = max(1, radiusin/16.)
        elif initmethod == 'pca':
            radiusin = max(1, np.ceil(ms/8.)/4)
            radiusfin = 1#max(1, ms/128)        
    
    radius = np.linspace(radiusin, radiusfin, trainlen)
    ##################################################    
    
    UD2 = getattr(self, 'UD2')
    New_Codebook_V = np.empty((nnodes, dim))
    New_Codebook_V = getattr(self, 'codebook')
    
    #print 'data is in shared memory?', shared_memory
    if shared_memory == 'yes':
        data = getattr(self, 'data')
        Data_folder = tempfile.mkdtemp()
        data_name = os.path.join(Data_folder, 'data')
        dump(data, data_name)
        data = load(data_name, mmap_mode='r')
    else:
        data = getattr(self, 'data')        
    #X2 is part of euclidean distance (x-y)^2 = x^2 +y^2 - 2xy that we use for each data row in bmu finding.
    #Since it is a fixed value we can skip it during bmu finding for each data point, but later we need it calculate quantification error
    X2 = np.einsum('ij,ij->i', data, data)
    if verbose=='on':
        print '%s training...' %phase
        print 'radius_ini: %f , radius_final: %f, trainlen: %d' %(radiusin, radiusfin, trainlen)
    for i in range(trainlen):
        #in case of Guassian neighborhood
        H = np.exp(-1.0*UD2/(2.0*radius[i]**2)).reshape(nnodes, nnodes)
        t1 = time()
        bmu = None
        bmu = self.para_bmu_find(data, New_Codebook_V, njb = njob)
        if verbose=='on':
            print
        #updating the codebook
        t2 = time()
        New_Codebook_V = self.update_codebook_voronoi(data, bmu, H, radius)
        #print 'updating nodes: ', round (time()- t2, 3)            
        if verbose=='on':
            print "epoch: %d ---> elapsed time:  %f, quantization error: %f " %(i+1, round(time() - t1, 3),np.mean(np.sqrt(bmu[1] + X2)))          
    setattr(self, 'codebook', New_Codebook_V)
    bmu[1] = np.sqrt(bmu[1] + X2)
    setattr(self, 'bmu', bmu)
    

def grid_dist(self,bmu_ind):
    """
    som and bmu_ind
    depending on the lattice "hexa" or "rect" we have different grid distance
    functions.
    bmu_ind is a number between 0 and number of nodes-1. depending on the map size
    bmu_coord will be calculated and then distance matrix in the map will be returned
    """
    try:
        lattice = getattr(self, 'lattice')
    except:
        lattice = 'hexa'
        print 'lattice not found! Lattice as hexa was set'
   
    if lattice == 'rect':
        return rect_dist(self,bmu_ind)
    elif lattice == 'hexa':
        try:
            msize =  getattr(self, 'mapsize')
            rows = msize[0]
            cols = msize[1]
        except:
            rows = 0.
            cols = 0.
            pass 
       
        #needs to be implemented
        print 'to be implemented' , rows , cols
        return np.zeros((rows,cols))

def rect_dist(self,bmu):
    #the way we consider the list of nodes in a planar grid is that node0 is on top left corner,
    #nodemapsz[1]-1 is top right corner and then it goes to the second row. 
    #no. of rows is map_size[0] and no. of cols is map_size[1]
    try:
        msize =  getattr(self, 'mapsize')
        rows = msize[0]
        cols = msize[1]
    except:
        pass 
        
    #bmu should be an integer between 0 to no_nodes
    if 0<=bmu<=(rows*cols):
        c_bmu = int(bmu%cols)
        r_bmu = int(bmu/cols)
    else:
      print 'wrong bmu'  
      
    #calculating the grid distance
    if np.logical_and(rows>0 , cols>0): 
        r,c = np.arange(0, rows, 1)[:,np.newaxis] , np.arange(0,cols, 1)
        dist2 = (r-r_bmu)**2 + (c-c_bmu)**2
        return dist2.ravel()
    else:
        print 'please consider the above mentioned errors'
        return np.zeros((rows,cols)).ravel()
        

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


def view_2d_Pack(self, text_size,which_dim='all', what = 'codebook',save='No', grid='Yes', save_dir = 'empty',text='Yes'):
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
        no_row_in_plot = dim/20 + 1 #6 is arbitrarily selected
        if no_row_in_plot <=1:
            no_col_in_plot = dim
        else:
            no_col_in_plot = 20
        
        axisNum = 0
        compname = getattr(self, 'compname')
        h = .2
        w= .001
        fig = plt.figure(figsize=(no_col_in_plot*1.5*(1+w),no_row_in_plot*1.5*(1+h)))
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
            	pl = plt.pcolor(mp[::-1])
            elif grid=='No':
            	plt.imshow(mp[::-1],norm = None)
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
            pl = plt.pcolor(mp[::-1])
        elif grid=='No':
            plt.imshow(mp[::-1])
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
        
        
##Function to show hits
#som_labels = sm.project_data(Tr_Data)
#S = pd.DataFrame(data=som_labels,columns= ['label'])
#a = S['label'].value_counts()
#a = a.sort_index()
#a = pd.DataFrame(data=a.values, index=a.index,columns=['label'])
#d = pd.DataFrame(data= range(msz0*msz1),columns=['node_ID'])
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
