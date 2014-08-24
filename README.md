SOMPY
=====

A Python Library for Self Organizing Map (SOM)

1- Batch training (It has parallel processing similar to sklearn might and it might speed up the training, but it depends on the data size and mainly size of the grid.I couldn't manage the memory problem and therefore, I mainly use single processing, but would be interesting to check it in a cloud)
2- PCA (or RandomPCA from sklearn) initialization or random initialization
3- component plane visualization (different modes)
4- Hitmap
3- 1-d or 2-d SOM with only rectangular, planar grid. (worked well in comparison with hexagonal shape, when I was checking in Matlab)
4- different methods for function approximation and predictions (mostly using Sklearn)
