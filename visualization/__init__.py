import shutil
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm


###################################
#visualize map
def view_map(self, what = 'codebook', which_dim = 'all', pack= 'Yes', text_size = 2.8,save='No', save_dir = 'empty',grid='No',text='Yes',cmap='None',COL_SiZe=6):

    mapsize = getattr(self, 'mapsize')
    if np.min(mapsize) >1:
        if pack == 'No':
            self.view_2d(self, text_size, which_dim = which_dim, what = what)
        else:
     		#print 'hi'
            self.view_2d_Pack(self, text_size, which_dim = which_dim,what = what,save = save, save_dir = save_dir, grid=grid,text=text,CMAP=cmap,col_sz=COL_SiZe)

    elif np.min(mapsize) == 1:
        self.view_1d(self, text_size, which_dim = which_dim, what = what)

def hit_map(self,data=None):
    #First Step: show the hitmap of all the training data

 	#print 'None'
    data_tr = getattr(self, 'data_raw')
    proj = self.project_data(data_tr)
    msz =  getattr(self, 'mapsize')
    coord = self.bmu_ind_to_xy(proj)

    #this is not an appropriate way, but it works
 	#coord[:,0] = msz[0]-coord[:,0]

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
 	#plt.scatter(coord[:,1]+.5, msz[0]-.5-coord[:,0], s=area, alpha=0.2,c='b',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
    coord = self.bmu_ind_to_xy(np.arange(self.nnodes))
    plt.scatter(coord[:,1], msz[0]-.5- coord[:,0], s=area.flatten(), alpha=.9,c='None',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
    plt.xlim(0,msz[1])
    plt.ylim(0,msz[0])

    if data != None:
        proj = self.project_data(data)
        msz =  getattr(self, 'mapsize')
        coord_d = self.bmu_ind_to_xy(proj)
        a = plt.hist2d(coord_d[:,1], coord_d[:,0], bins=(msz[1],msz[0]),alpha=.0,norm = LogNorm(),cmap=cm.jet)
        # clbar  = plt.colorbar()
        x = np.arange(.5,msz[1]+.5,1)
        y = np.arange(.5,msz[0]+.5,1)
        X, Y = np.meshgrid(x, y)

        area = a[0].T*50

        plt.scatter(coord_d[:,1]+.5, msz[0]-.5-coord_d[:,0], s=area, alpha=0.9,c='None',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
        plt.scatter(coord_d[:,1]+.5, msz[0]-.5-coord_d[:,0], s=area, alpha=0.2,c='b',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
        print 'hi'
 		#plt.scatter(coord[:,1], msz[0]-1-coord[:,0], s=area, alpha=0.2,c='b',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
 		#plt.scatter(X, msz[0]-1-Y, s=area, alpha=0.2,c='b',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')#
 		#plt.scatter(X, msz[0]-1-Y, s=area, alpha=0.9,c='None',marker='o',cmap='jet',linewidths=3, edgecolor = 'r')
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
    coord = self.bmu_ind_to_xy(proj)
     #freq = plt.hist2d(coord[:,1], coord[:,0], bins=(msz[1],msz[0]),alpha=1.0,cmap=cm.jet)[0]
     #plt.close()
     #fig, ax = plt.figure()
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
     	#mn = md
     	#umat[umat<=mn]=mn
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
 		#print 'yesyy'
    else:
        print 'clustering based on default parameters...'
        codebook = self.cluster()
    msz =  getattr(self, 'mapsize')
    fig = plt.figure(figsize=(msz[1]/2.5,msz[0]/2.5))
    ax = fig.add_subplot(111)
 	#ax.xaxis.set_ticklabels([])
 	#ax.yaxis.set_ticklabels([])
 	#ax.grid(True,linestyle='-', linewidth=.5)


    if data == None:
        data_tr = getattr(self, 'data_raw')
        proj = self.project_data(data_tr)
        coord = self.bmu_ind_to_xy(proj)
        cents = self.bmu_ind_to_xy(np.arange(0,msz[0]*msz[1]))
        for i, txt in enumerate(codebook):
                ax.annotate(txt, (cents[i,1],cents[i,0]),size=10, va="center")

    if data != None:
        proj = self.project_data(data)
        coord = self.bmu_ind_to_xy(proj)
        x = np.arange(.5,msz[1]+.5,1)
        y = np.arange(.5,msz[0]+.5,1)
        cents = self.bmu_ind_to_xy(proj)
 		#cents[:,1] = cents[:,1]+.2
 		#print cents.shape
        label = codebook[proj]
        for i, txt in enumerate(label):
            ax.annotate(txt, (cents[i,1],cents[i,0]),size=10, va="center")



    plt.imshow(codebook.reshape(msz[0],msz[1])[::],alpha=.5)
 	#plt.pcolor(codebook.reshape(msz[0],msz[1])[::-1],alpha=.5,cmap='jet')
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
    coords = self.bmu_ind_to_xyp(proj)[:,:2]
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
			#print mx.shape
			#print coords.shape
            for j in range(data.shape[0]):
                sc = plt.scatter(coords[j,1],self.mapsize[0]-1-coords[j,0],c=data[j,which_dim[i]],vmax=mx[j],vmin=mn[j],s=90,marker='.',edgecolor='None', cmap=colormap ,alpha=1)



            mn = data# a[:,i].min()
			#mx = data[:,i].max()
			#plt.scatter(coords[:,1],self.mapsize[0]-1-coords[:,0],c=data[:,i],vmax=mx,vmin=mn,s=180,marker='.',edgecolor='None', cmap=colormap ,alpha=1)





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
			#print mx.shape
			#print coords.shape
            for j in range(data.shape[0]):
                sc = plt.scatter(coords[j,1],self.mapsize[0]-1-coords[j,0],c=data[j,which_dim[i]],vmax=mx[j],vmin=mn[j],s=90,marker='.',edgecolor='None', cmap=colormap ,alpha=1)




			#mn = data[:,which_dim[i]].min()
			#mx = data[:,which_dim[i]].max()
			#plt.scatter(coords[:,1],self.mapsize[0]-1-coords[:,0],c=data[:,which_dim[i]],vmax=mx,vmin=mn,s=180,marker='.',edgecolor='None', cmap=colormap ,alpha=1)




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
	#fig.set_size_inches(msz0/2,msz1/2)
	#fig = plt.figure(figsize=(msz0/2,msz1/2))
    if save=='Yes':
        if save_dir != 'empty':
            fig.savefig(save_dir, transparent=False, dpi=200)
        else:
            add = '/Users/itadmin/Desktop/SOM_dot.png'
            print 'save directory: ', add
            fig.savefig(add, transparent=False, dpi=200)

        plt.close(fig)

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
