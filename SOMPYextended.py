# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:01:36 2015

@author: mario
"""

from SOMPY import SOM as SOM
import SOMPY as SOMmodule

import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import pandas as pd
from matplotlib import cm



class SOMextended(SOM):


####################MARIO ŽUPAN CODE########################################
############################################################################
####################MARIO ŽUPAN CODE########################################
############################################################################
###################################
    #visualize u-matrix
    def view_umap(self, um, what = 'umatrix', which_dim =1, pack= 'Yes', text_size = 2.8,save='No', save_dir = 'empty',grid='No',text='Yes',cmap='None',COL_SiZe=6):

        if pack == 'No':
            print 'THIS IS U-MATRIX 1st view' 
            view_umatrix1(self, um, text_size, which_dim = which_dim, what = what)
        else:
            print 'THIS IS U-MATRIX 2nd view' 
            view_umatrix2(self, um, text_size, which_dim = which_dim,what = what,save = save, save_dir = save_dir, grid=grid,text=text,CMAP=cmap,col_sz=COL_SiZe)
 


###########################################################################
#
#
#/** Get weight vector from a codebook using x, y index
# * @param codebook - the codebook to save
# * @param som_y - y coordinate of a node in the map
# * @param som_x - x coordinate of a node in the map
# * @param nSomX - dimensions of SOM map in the x direction
# * @param nSomY - dimensions of SOM map in the y direction
# * @param nDimensions - dimensions of a data instance
# * @return the weight vector
# */



def get_wvec(codebook, som_y, som_x, nSomX, nDimensions):
    wvec = np.empty(nDimensions, dtype=float)
    #print 'empty string', wvec.shape
    #print 'codebook shape', codebook.shape
    for d in range(0,nDimensions):
        wvec[d] = codebook[som_y*nSomX*nDimensions+som_x*nDimensions+d]# CAUTION: (y,x) order
        #print 'vector weight',wvec[d]
    return wvec




#/** Euclidean distance between vec1 and vec2
# * @param vec1
# * @param vec2
# * @param nDimensions
# * @return distance
# */
def get_distance(vec1, vec2, nDimensions):
    distance = 0.0
    x1 = 0.0
    x2 = 0.0
    for d in range(0,nDimensions):
        x1 = min(vec1[d], vec2[d])
        x2 = max(vec1[d], vec2[d])
        distance += abs(x1-x2)*abs(x1-x2)
    return math.sqrt(distance)



def euclideanDistanceOnToroidMap(som_x, som_y, x, y, nSomX, nSomY):
    x1 = min(som_x, x)
    y1 = min(som_y, y)
    x2 = max(som_x, x)
    y2 = max(som_y, y)
    xdist = min(x2-x1, x1+nSomX-x2)
    ydist = min(y2-y1, y1+nSomY-y2)

    return math.sqrt(xdist*xdist+ydist*ydist)


def euclideanDistanceOnPlanarMap(som_x, som_y, x, y):
    x1 = min(som_x, x)
    y1 = min(som_y, y)
    x2 = max(som_x, x)
    y2 = max(som_y, y)
    xdist = x2-x1
    ydist = y2-y1
    
    return math.sqrt(xdist*xdist+ydist*ydist)




#/** Calculate U-matrix
# * @param codebook - the codebook
# * @param nSomX - dimensions of SOM map in the x direction
# * @param nSomY - dimensions of SOM map in the y direction
# * @param nDimensions - dimensions of a data instance
# */

#self means that is used by som1 implementation
#without self you can use it by SOM.

def calculateUMatrixMario(self, codebook, nSomX, nSomY, nDimensions, mapType):
    uMatrix = np.empty(nSomX*nSomY, dtype=float)
    print 'u-matrix size ', uMatrix.shape
    min_dist = 1.5
    #for loop in python goes from 0 to 2 if you wrote in range (0,3)
    for som_y1 in range(0,nSomY):
        for som_x1 in range(0,nSomX):
            dist = 0.0
            nodes_number = 0
            for som_y2 in range(0,nSomY-1):            
                for som_x2 in range(0,nSomX-1):
                    if som_x1 == som_x2 and som_y1 == som_y2:
                        continue
                    tmp = 0.0
                    if mapType == "planar":
                        tmp = euclideanDistanceOnPlanarMap(som_x1, som_y1, som_x2, som_y2)
                    elif mapType == "toroid":
                        tmp = euclideanDistanceOnToroidMap(som_x1, som_y1, som_x2, som_y2, nSomX, nSomY)
                    if tmp <= min_dist:
                        nodes_number += 1
                        vec1 = get_wvec(codebook, som_y1, som_x1, nSomX, nDimensions)
                        vec2 = get_wvec(codebook, som_y2, som_x2, nSomX, nDimensions)
                        dist += get_distance(vec1, vec2, nDimensions)
                        del vec1
                        del vec2
            dist /= nodes_number
            uMatrix[som_y1*nSomX+som_x1] = dist
    return uMatrix




def view_umatrix1(self, um, text_size,which_dim='all', what = 'umatrix'):
    msz0, msz1 = getattr(self, 'mapsize')
    if what == 'umatrix':
        if hasattr(self, 'codebook'):
            umatrix = um
        else:
            print 'first initialize umatrix'
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
            max_dim = umatrix.shape[1]
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
        norm = matplotlib.colors.normalize(vmin = np.mean(umatrix.flatten())-1*np.std(umatrix.flatten()), vmax = np.mean(umatrix.flatten())+1*np.std(umatrix.flatten()), clip = True)
        while axisNum <dim:
            axisNum += 1
            ax = plt.subplot(no_row_in_plot, no_col_in_plot, axisNum)
            ind = int(indtoshow[axisNum-1])
            mp = umatrix[:].reshape(msz0, msz1)
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
            #####################
            #plot data on a u-matrix
            #####################            
            data = getattr(self, 'data_raw')
            proj = self.project_data(data)
            msz =  getattr(self, 'mapsize')
            coord = self.ind_to_xy(proj)

            print 'this is not an appropriate way, but it works'
            coord[:,0] = msz[0]-coord[:,0]

            a = plt.hist2d(coord[:,1], coord[:,0], bins=(msz[1],msz[0]),alpha=.0,norm = LogNorm(),cmap=cm.jet)
            x = np.arange(.5,msz[1]+.5,1)
            y = np.arange(.5,msz[0]+.5,1)
            X, Y = np.meshgrid(x, y)
            area = a[0].T*50
            plt.scatter(X, Y, s=area, alpha=0.2,c='b',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
            plt.scatter(X, Y, s=area, alpha=0.9,c='None',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')            
            plt.xlim(0,msz[1])
            plt.ylim(0,msz[0])
            #####################
            #plot data ona map END
            #####################            
           
        plt.show()    


def view_umatrix2(self, um, text_size,which_dim='all', what = 'umatrix',save='No', grid='Yes', save_dir = 'empty',text='Yes',CMAP='None',col_sz=None):
    import matplotlib.cm as cm
    msz0, msz1 = getattr(self, 'mapsize')
    if CMAP=='None':
        CMAP= cm.RdYlBu_r
    if what == 'umatrix':
        if hasattr(self, 'codebook'):
            umatrix = um
        else:
            print 'first initialize umatrix'
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
            max_dim = umatrix.shape[1]
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
        #5.5 in next row is the size of lattice
        fig = plt.figure(figsize=(no_col_in_plot*5.5*(1+w),no_row_in_plot*5.5*(1+h)))
#         print no_row_in_plot, no_col_in_plot
        norm = matplotlib.colors.Normalize(vmin = np.median(umatrix.flatten())-1.5*np.std(umatrix.flatten()), vmax = np.median(umatrix.flatten())+1.5*np.std(umatrix.flatten()), clip = False)
        
        DD = pd.Series(data = umatrix.flatten()).describe(percentiles=[.03,.05,.1,.25,.3,.4,.5,.6,.7,.8,.9,.95,.97])
        norm = matplotlib.colors.Normalize(vmin = DD.ix['3%'], vmax = DD.ix['97%'], clip = False)

        while axisNum <dim:
            axisNum += 1
            
            ax = fig.add_subplot(no_row_in_plot, no_col_in_plot, axisNum)
            ind = int(indtoshow[axisNum-1])
            mp = umatrix[:].reshape(msz0, msz1)
            
            if grid=='Yes':
                pl = plt.pcolor(mp[::-1])
            elif grid=='No':
                plt.imshow(mp[::-1],norm = None,cmap=CMAP)
#                 plt.pcolor(mp[::-1])
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

        #####################
        #plot data on a u-matrix
        #####################            
        data = getattr(self, 'data_raw')
        proj = self.project_data(data)
        msz =  getattr(self, 'mapsize')
        coord = self.ind_to_xy(proj)

        print 'this is not an appropriate way, but it works'
        coord[:,0] = msz[0]-coord[:,0]

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
        #####################
        #plot data ona map END
        #####################     

        #####################
        #plot blur cluster borders
        #####################     
        
        fig = plt.figure(figsize=(msz[1]/2.5,msz[0]/2.5))
        ax = fig.add_subplot(111)
        plt.imshow(um.reshape(msz[0],msz[1])[::],alpha=.5)
        plt.pcolor(self.codebook.reshape(msz[0],msz[1])[::-1],alpha=.5,cmap='jet')
        plt.show()
        #####################
        #plot blur cluster borders END
        #####################     
        
        

    if what == 'cluster':
        if hasattr(self, 'cluster_labels'):
            umatrix = getattr(self, 'cluster_labels')
        else:
            print 'clustering based on default parameters...'
            umatrix = self.cluster()
        h = .2
        w= .001
        fig = plt.figure(figsize=(msz0/2,msz1/2))
        
        ax = fig.add_subplot(1, 1, 1)
        mp = umatrix[:].reshape(msz0, msz1)
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
#                 print save_dir
                fig.savefig(save_dir,bbox_inches='tight', transparent=False, dpi=200) 
            else:
#                 print save_dir
                add = '/Users/itadmin/Desktop/SOM.png'
                fig.savefig(add,bbox_inches='tight', transparent=False, dpi=200)    
            
            plt.close(fig)



####################MARIO ŽUPAN CODE########################################
############################################################################
####################MARIO ŽUPAN CODE########################################
################################END#########################################


def plot_radar(som,path,save='Yes',legend='No',alpha=.5):
    import matplotlib.pyplot as plt
    titles = som.compname[:][0].copy()
    labels =[]
    denorm_cd= SOMmodule.denormalize_by(som.data_raw,som.codebook)
    mx = denorm_cd.max(axis=0)
    mn = denorm_cd.min(axis=0)
    rng = mx-mn
    denorm_cd = (denorm_cd-mn)/rng
    
    for dim in range(som.codebook.shape[1]):
        titles[dim] = titles[dim].replace("; measures: Value","",1)
        titles[dim]= titles[dim].replace(som.name+': ',"",1)
    
        labels.append(np.around(np.linspace(mn[dim],mx[dim],num=5),decimals=3).tolist())

    titles=titles.tolist()


    fig = plt.figure(figsize=(10, 10))

    
    rect = [0.15, 0.1, .7, .7]

    n = len(titles)
    angles = np.arange(0, 360, 360.0/n)
    axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i) 
                         for i in range(n)]

    ax = axes[0]
    ax.set_thetagrids(angles,labels=titles, fontsize=5, frac=1.15, rotation=0,multialignment='left',fontweight='demi')
    ax.set_theta_offset(10)

    for ax in axes[1:]:
        ax.patch.set_visible(False)
        ax.grid("off")
        ax.xaxis.set_visible(False)
        ax.set_theta_offset(10)

        

    for ax, angle, label in zip(axes, angles, labels):
        ax.set_rgrids(range(1, 6), angle=angle, labels=label)
        ax.spines["polar"].set_visible(False)
        ax.set_ylim(0, 6)

    
    N = denorm_cd.shape[0]
    n_aspect = denorm_cd.shape[1]
    for dim in range(N):
        angle = np.deg2rad(np.r_[angles, angles[0]])
        values = denorm_cd[dim]*5
        values = np.r_[values, values[0]]
        if n_aspect<=2:
            ax.plot(angle,values,"o", markersize=12, color=plt.get_cmap('RdYlBu_r')(dim/float(N)),alpha=0.8)
#         if n_aspect>2:
        ax.fill(angle,values,"-", lw=4, color=plt.get_cmap('RdYlBu_r')(dim/float(N)), alpha=alpha, label=str(dim))
#         radar.plot(denorm_cd[dim]*5,  "-", lw=4, color=plt.get_cmap('RdYlBu_r')(dim/float(N)), alpha=0.4, label='cluster '+str(dim))

    if legend=='Yes':
        
        ax.legend(bbox_to_anchor= (1.21,0.35),labelspacing=.35,fontsize ='x-small', title='Cluster', 
              fancybox = True, handletextpad=.8, borderpad=1.)
    
    font = {'size'   : 8}
    plt.rc('font', **font)
    plt.figtext(0.5, .9, som.name,
                ha='center', color='black', weight='bold', size='large')
    if save =='Yes':
        plt.savefig(path, dpi=400, transparent=False)
        plt.close() 
#####################################################################