import numpy as np
from matplotlib import pyplot as plt


class View(object):

    def __init__(self, width, height, title, show_axis=True, packed=True, text_size=2.8, show_text=True, col_size=6, *args, **kwargs):
        self.width = width
        self.height = height
        self.title = title
        self.show_axis = show_axis
        self.packed = packed
        self.text_size = text_size
        self.show_text = show_text
        self.col_size = col_size

    def prepare(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def show(self, *args, **kwrags):
        raise NotImplementedError()


class MatplotView(View):

    def __init__(self, width, height, title, show_axis=True, packed=True, text_size=2.8, show_text=True, col_size=6, *args, **kwargs):
        super(MatplotView, self).__init__(width, height, title, show_axis, packed, text_size, show_text, col_size, *args, **kwargs)
        self._fig = None

    def __del__(self):
        self._close_fig()

    def _close_fig(self):
        if self._fig:
            plt.close(self._fig)

    def prepare(self, *args, **kwargs):
        self._close_fig()
        self._fig = plt.figure(figsize=(self.width, self.height))
        plt.title(self.title)
        plt.axis('off')
#         plt.axis('on' if self.show_axis else 'off')
        plt.rc('font', **{'size': self.text_size})

    def save(self, filename, transparent=False, bbox_inches='tight', dpi=400):
        self._fig.savefig(filename, transparent=transparent, dpi=dpi, bbox_inches=bbox_inches)

    def show(self, *args, **kwrags):
        raise NotImplementedError()


###################################
#visualize map
#def view_map(self, what = 'codebook', which_dim = 'all', pack= 'Yes', text_size = 2.8,save='No', save_dir = 'empty',grid='No',text='Yes',cmap='None',COL_SiZe=6):
#
#    mapsize = getattr(self, 'mapsize')
#    if np.min(mapsize) >1:
#        if pack == 'No':
#            self.view_2d(self, text_size, which_dim = which_dim, what = what)
#        else:
#     		#print 'hi'
#            self.view_2d_Pack(self, text_size, which_dim = which_dim,what = what,save = save, save_dir = save_dir, grid=grid,text=text,CMAP=cmap,col_sz=COL_SiZe)
#
#    elif np.min(mapsize) == 1:
#        self.view_1d(self, text_size, which_dim = which_dim, what = what)
#
#

