# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 06:58:33 2015

@author: mario župan
"""


#SOMPYextended is a module. It consist SOMextended, class which extends 
#class SOM, wrote by:

## Vahid Moosavi 2015 03 12 10:04 pm
#sevamoo@gmail.com
#Chair For Computer Aided Architectural Design, ETH  Zurich
# Future Cities Lab
#www.vahidmoosavi.com


import SOMPYextended as SOMmodule
from SOMPYextended import SOMextended as SOM



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
pd.__version__
import sys

#reading csv
from numpy import genfromtxt, savetxt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score




        
### A toy example: two dimensional data, 800rows, four clusters

dlen = 200
Data1 = pd.DataFrame(data= 1*np.random.rand(dlen,2))
Data1.values[:,1] = (Data1.values[:,0][:,np.newaxis] + .42*np.random.rand(dlen,1))[:,0]


Data2 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+1)
Data2.values[:,1] = (-1*Data2.values[:,0][:,np.newaxis] + .62*np.random.rand(dlen,1))[:,0]

Data3 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+2)
Data3.values[:,1] = (.5*Data3.values[:,0][:,np.newaxis] + 1*np.random.rand(dlen,1))[:,0]


Data4 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+3.5)
Data4.values[:,1] = (-.1*Data4.values[:,0][:,np.newaxis] + .5*np.random.rand(dlen,1))[:,0]

Data = np.concatenate((Data1,Data2,Data3,Data4))

fig = plt.figure()
plt.plot(Data[:,0],Data[:,1],'ob',alpha=0.2, markersize=4)
fig.set_size_inches(7,7)

##########################################################################

msz11 = 15
msz10 = 15

som = SOM('som', Data, mapsize = [msz10, msz11],norm_method = 'var',initmethod='pca')
#What you get when you initialize the map with pca
som.init_map()
som.view_map(text_size=7)

#hitmap
som.hit_map()





#What you get when you train the map
som.train(n_job = 1, shared_memory = 'no',verbose='off')
som.view_map(what='codebook', which_dim='all',pack='Yes', text_size=7, save='No', save_dir='empty', grid='No', text='Yes', cmap='None', COL_SiZe=6)
#som.view_map(text_size=7)
#som.view_map(what='codebook', which_dim='all',pack='No', text_size=7, save='No', save_dir='empty', grid='Yes', text='Yes', cmap='None', COL_SiZe=6)


#a = som.cluster_labels
#b = som.cluster()
c = som.data_raw
c.shape



#RESHAPING CODEBOOK DATASET
#reshape 2-dimensional array to 1-dimension
#if you want reshape in 3 dimendions: np.reshape(d, (d.shape[0],3))
d = som.codebook
dim = d.shape
dim
d1 = np.reshape(d, dim[0]*dim[1])
d1.shape # 15x15 matrix i 2 dimensions(features) gives (225,2)



#u-matrix calculation
um = SOMmodule.calculateUMatrixMario('som',d1,15,15,2,mapType='planar')
um.shape


#this gives the cluster label of each node
labels = som.cluster(method='Kmeans', n_clusters=4)
#clusters 0,1,2,3 fore every node. This example have a 225 nodes (15x15)
ClusterNum = som.cluster_labels
print ClusterNum

#if you did clusters before
cents  = som.hit_map_cluster_number()
som.hit_map()



reload(sys.modules['SOMPYextended'])
#%pylab qt
#one way to show u-matrix
som.view_umap(um, what='umatrix', which_dim=0,pack='Yes', text_size=8, save='No', save_dir='empty', grid='Yes', text='Yes', cmap='None', COL_SiZe=6)
#second way to show u-matrix
som.view_umap(um, what='umatrix', which_dim=0,pack='No', text_size=3, save='No', save_dir='empty', grid='Yes', text='Yes', cmap='None', COL_SiZe=6)
som.hit_map()



#AVERAGE SILHOUETTE  - CLUSTER QUALITY INDEX

rowId = som.hit_map_cluster_number(Data[:])
#the first two cols are x,y in SOM and third one is the node id.
print "row ID of the last row:", rowId[799,2]
print 'x y in the SOM:', rowId[799,0],'-',rowId[799,1]


ClustId =  labels[rowId[:,2]]
#ClustId for every row of dataset
print ClustId


# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(Data, ClustId)
print 'The average silhouette_score is :', silhouette_avg

#Pearson correlation between data features (variables). SOM has success in non-linear datasets
# Data array to DataFrame
DataDF = pd.DataFrame(Data)      
#personovo koeficijent korelacije
DataDF.corr(method='pearson')


##########################################################################
##########################################################################
#Plot radar

#
##Another interesting thing is that many times we want to know the charachteristics of the clusters
##Let's assume we have a 3dimensional data
#Data1 = np.concatenate((Data,np.random.rand(Data.shape[0],2)),axis=1)
## a 1d SOM with 10 nodes can be considered as 10 clusters with a major difference to kmeans that similar cluster labels
##have similar context. For example the weight vector of cluster 1 is similar cluster 2 than cluster than
##while in K-means labels are arbitrary. Here these labels are contextual numbers. For more info:http://arxiv.org/abs/1408.0889 
#msz11 = 1
#msz10 = 10
#som = SOM('som', Data1, mapsize = [msz10, msz11],norm_method = 'var',initmethod='pca')
#
#som.train(n_job = 1, shared_memory = 'no',verbose='off')
#
#print 'Done'


#The following function visualizes the pattern of nodes in a 1d som,
#but since the final som is an ordered set, it creates a nice patters
#crtanje u novom prozoru

path = ''
SOMmodule.plot_radar(som,path,save='No',alpha=.4,legend='Yes')



#close all qt windows
plt.close('all')
##################################################################
##################################################################
##################################################################
##################################################################


#STATISTICS ABOUT CODEBOOK. It is not necessary.
##################################################################
##################################################################
##################################################################
##################################################################


DataDF = pd.DataFrame(d)  
#Reading the dataset in a dataframe using Pandas
DataDF.head(10)
#summary
DataDF.describe()



##################################################################
##################################################################
