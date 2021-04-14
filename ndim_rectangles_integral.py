#!/usr/bin/env python

"""Wassim Tenachi, Strabourg Astronomical Observatory"""


import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as collections_OrderedDict

def ndim_rectangles_integral(
                            # main args
                            func,
                            up_limits, 
                            low_limits, 
                            ndim, 
                            nsamples=10000,
                            args_func = {},
                            # demo plot args
                            verbose=False,
                            args_subplots = {'sharex':True, 'sharey':True, 'figsize':(10,10)}, 
                            args_suptitle = {'fontsize':16},
                            args_scatter_mesh = {'marker':"+", 'color':"black", 'label':"rectangular mesh"},
                            args_scatter_func = {'marker':"o", 'label':"computed points"},
                            args_legend = {},
                            dim_labels = None
                            ): 
                            
    """
    Returns the integral of a function in n-dimensions using the textbook rectangle method. 
    Heavy usage of numpy functions to benefit from parallization.
    Tip: To save RAM, divide integration space into sub-spaces and integrate one at a time.
    
    v0.1
    
    Parameters
    ----------
    
    func : function
        A Python function or method to integrate. The function takes an array of coordinates of shape=(ndim) and/or shape=(ndim, nsamples) to be integrated as first argument.
        Other arguments can be passed using the args_func dictionary argument.
    up_limits: array_like
        Upward bounds of integrations. Expected shape = (ndim)
    low_limits: array_like
        Downward bounds of integrations. Expected shape = (ndim)
    nsamples: integer or array_like, optional
        #Samples of integrations in each dimension. Expected shape = (ndim). If an integer is given, #samples are divided between each dimension by nsamples**(1/ndim).
    args_func: dictionary, optional
        Supplementary arguments to pass to func.
        
    verbose: boolean, optional
        Generates a matplotlib (plt) figure of the integration space meshing and samples. This involves the computation of an histogram which is significantly computationaly intensive. Verbose=True should be used for verifications only with a low number of samples.
    args_subplots: dictionary, optional
        Supplementary arguments to pass to the plt.subplot function for pdf sample / space meshing visualisation (for verbose=True).
    args_suptitle: dictionary, optional
        Supplementary arguments to pass to the plt.suptitle function for pdf sample / space meshing visualisation (for verbose=True).
    args_scatter_mesh: dictionary, optional
        Supplementary arguments to pass to the plt.scatter function for space meshing visualisation (for verbose=True).
    args_scatter_func: dictionary, optional
        Supplementary arguments to pass to the plt.scatter function for pdf sample visualisation (for verbose=True).
    args_legend: dictionary, optional
        Supplementary arguments to pass to the plt.legend function for pdf sample / space meshing visualisation (for verbose=True).
    dim_labels = array_like, optional
        Label of each dimension for pdf sample / space meshing visualisation (for verbose=True). Expected shape = (ndim)
        
    Returns
    -------
    result : float
        The result of the integration.
        
    Example
    --------
    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    
    dim_labels = ["x", "y", "z"]
    ndim = len(dim_labels)
    df_func = lambda x:stats.multivariate_normal.pdf(x, mean=np.zeros(ndim), cov=np.eye(ndim))
    integral = ndim_rectangles_integral (func = df_func,
                                    up_limits = np.full(ndim,4),
                                    low_limits = np.full(ndim,-4),
                                    ndim=ndim, 
                                    nsamples = np.full(ndim,11),
                                    verbose = True,
                                    dim_labels = dim_labels,)
    print("integral = %f"%(integral))
    plt.show()
    """
    #---------------------------------------------------------------
    # supporting int as n_samples argument
    if isinstance(nsamples, int):
        nsamples = np.full(ndim,int(nsamples**(1/ndim))) 
    # checking arguments
    if not(len(up_limits)==len(low_limits)==ndim==len(nsamples)):
        raise ValueError("Shapes should be len(up_limits)=len(low_limits)=ndim")
    #---------------------------------------------------------------
    # todo: max_memory argument. automated space division
    #---------------------------------------------------------------
    # hyperrectangles edge size in each dimension
    ndx = np.array([(up_limits[dim] - low_limits[dim])/(nsamples[dim]-1) for dim in range(ndim)])
    # hyperrectangle volume
    vol = np.prod(ndx)
    # hyperrectangles centers: edges
    ncenters = np.array([np.linspace(start=low_limits[dim]+ndx[dim]/2, stop=up_limits[dim]-ndx[dim]/2, num=nsamples[dim]-1) for dim in range(ndim)])
    del ndx
    # hyperrectangles centers: coords
    ncoords_centers = np.array(np.meshgrid(*ncenters))
    del ncenters
    ncoords_centers = ncoords_centers.reshape(ncoords_centers.shape[0],np.prod(ncoords_centers.shape[1:])) # equivalent to ncoords_centers = ncoords_centers.reshape(ndim,np.prod(nsamples-1))
    ncoords_centers = ncoords_centers.transpose()
    #---------------------------------------------------------------
    # integral computation
    try: # if func supports array of coords
        mapped_func = func(ncoords_centers, **args_func)
    except: # if func only supports 1 coord at a time
        mapped_func = np.array([func(ncoords_centers[i], **args_func) for i in range (ncoords_centers.shape[0])])
    # dividing by volume 
    integral = np.sum(mapped_func)*vol
    #---------------------------------------------------------------
    #todo: error computation 
    # # not sure about this...
    # mapped_err = np.abs(mapped_func-np.roll(mapped_func, 1))/2
    # err = np.sum(mapped_err)*vol 
    #---------------------------------------------------------------
    # mesh plot for visualisation purposes
    if verbose==1:
        # meshing edges for display
        nedges = np.array([np.linspace(start=low_limits[dim], stop=up_limits[dim], num=nsamples[dim]) for dim in range(ndim)], dtype=object) # nedges.shape = (ndim, nsamples in dim)
        ncoords_edges = np.array(np.meshgrid(*nedges))
        ncoords_edges = ncoords_edges.reshape(ncoords_edges.shape[0],np.prod(ncoords_edges.shape[1:]))
        # plot
        fig, ax = plt.subplots(ndim ,ndim, **args_subplots)
        #title
        args_suptitle_default = {'t':"Mesh and func samples used. Integral = %f"%(integral)} # default title
        args_suptitle_default.update(args_suptitle)
        fig.suptitle(**args_suptitle_default)
        for i in range(ndim):
            for j in range (ndim):
                # mesh: plot
                ax[i,j].scatter(ncoords_edges[i,:], ncoords_edges[j,:], **args_scatter_mesh)
                # df sample points: cleaning supperposed values, summing prob along other dimensions
                temp_centers_ij = np.append(ncoords_centers[:,[i,j]], mapped_func.reshape(mapped_func.shape[0],1),axis=1)
                temp_centers_ij = temp_centers_ij[np.lexsort((temp_centers_ij[:,0], temp_centers_ij[:,1]))]
                unique_centers = []
                unique_prob = []
                counter = -1 
                for k in range(temp_centers_ij.shape[0]):
                    if np.sum(temp_centers_ij[k,0:2] != temp_centers_ij[k-1,0:2]):
                        unique_prob.append(temp_centers_ij[k,2])
                        unique_centers.append(temp_centers_ij[k,0:2])
                        counter+=1
                    else:
                        unique_prob[counter]+=temp_centers_ij[k,2]
                unique_centers = np.array(unique_centers)
                unique_prob = np.array(unique_prob)
                #todo: use an image instead of points for the sampled pdf
                # df sample points: plot
                df_plot = ax[i,j].scatter(unique_centers[:,0], unique_centers[:,1], c=unique_prob, **args_scatter_func)
                plt.colorbar(df_plot, ax=ax[i,j])
                # labels
                if dim_labels != None:
                    ax[i,j].set_xlabel(dim_labels[i])
                    ax[i,j].set_ylabel(dim_labels[j])
        # legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = collections_OrderedDict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), **args_legend)
    #---------------------------------------------------------------
    return integral
    
## demo
# from scipy import stats
# import numpy as np
# import matplotlib.pyplot as plt
# 
# dim_labels = ["x", "y", "z"]
# ndim = len(dim_labels)
# df_func = lambda x:stats.multivariate_normal.pdf(x, mean=np.zeros(ndim), cov=np.eye(ndim))
# integral = ndim_rectangles_integral (func = df_func,
#                                 up_limits = np.full(ndim,4),
#                                 low_limits = np.full(ndim,-4),
#                                 ndim=ndim, 
#                                 nsamples = np.full(ndim,11),
#                                 verbose = True,
#                                 dim_labels = dim_labels,)
# print("integral = %f"%(integral))
# plt.show()