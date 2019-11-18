import numpy as np
from math import floor, ceil


def upper_tri_indexing(RDM):
    """upper_tri_indexing returns the upper triangular index of an RDM
    
    Args:
        RDM 2Darray: squareform RDM
    
    Returns:
        1D array: upper triangular vector of the RDM
    """
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]


def corr_rdms(X, Y):
    """ corr_rdms useful for correlation of RDMs (e.g. in multimodal RSA fusion)
        where you correlate two ndim RDM stacks.

    Args:
        X [array]: e.g. the fmri searchlight RDMs (ncenters x npairs)
        Y [array]: e.g. the EEG time RDMs (ntimes x npairs)

    Returns:
        [type]: correlations between X and Y, of shape dim0 of X by dim0 of Y
    """
    X = X - X.mean(axis=1, keepdims=True)
    X /= np.sqrt(np.einsum('ij,ij->i', X, X))[:, None]
    Y = Y - Y.mean(axis=1, keepdims=True)
    Y /= np.sqrt(np.einsum('ij,ij->i', Y, Y))[:, None]

    return np.einsum('ik,jk', X, Y)


def chunking(vect, num, chunknum=None):
    """ chunking
    Input:
        <vect> is a array
        <num> is desired length of a chunk
        <chunknum> is chunk number desired (here we use a 1-based
              indexing, i.e. you may want the first chunk, or the second
              chunk, but not the zeroth chunk)
    Returns:
        [numpy array object]:

        return a numpy array object of chunks.  the last vector
        may have fewer than <num> elements.

        also return the beginning indices associated with
        this chunk in <xbegin>.

    Examples:

        a = np.empty((2,), dtype=np.object)
        a[0] = [1, 2, 3]
        a[1] = [4, 5]
        assert(np.all(chunking(list(np.arange(5)+1),3)==a))

        assert(chunking([4, 2, 3], 2, 2)==([3], 3, 3))

    """
    if chunknum is None:
        nchunk = int(np.ceil(len(vect)/num))
        f = []
        for point in range(nchunk):
            f.append(vect[point*num:np.min((len(vect), int((point+1)*num)))])

        return np.asarray(f)
    else:
        f = chunking(vect, num)
        # double check that these behave like in matlab (xbegin)
        xbegin = (chunknum-1)*num+1
        return np.asarray(f[num-1]), xbegin

def isnotfinite(arr):
    """find non-finite entries
    
    Args:
        arr (numpy array): numpy array with finite or infinite entries 
    
    Returns:
        bool: boolean with True when non-finite 
    """
    res = np.isfinite(arr)
    np.bitwise_not(res, out=res)  # in-place
    return res

def makeimagestack(m):
    """ make a simple stack of slices for visualisation.
    
    Args:
        m (3D numpy array): 3 Dimensional volume
    
    Returns:
        [type]: given the number of z-slices, we distribute these
                slices in a n by p
    Example:

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        vol = np.zeros((20, 20, 34))
        vol[5:15, 5:15, 8:28]=1
        fig = plt.figure()
        im = plt.imshow(makeimagestack(vol), cmap=cm.coolwarm)
        plt.show()
        
    """

    bordersize = 1 
    
    # calc
    nrows, ncols, numim = m.shape
    mx = np.nanmax(m.ravel())

    # calculate csize

    rows = floor(np.sqrt(numim))
    cols = ceil(numim/rows)
    csize = [rows, cols]

    rowstop = rows-1

    # calc
    chunksize = csize[0]*csize[1]
    numchunks = ceil(numim/chunksize)

    # total cols and rows for adding border to slices
    tnrows = nrows+bordersize
    tncols = ncols+bordersize

    # make a zero array of chunksize
    # add border
    mchunk = np.zeros((tnrows, tncols, chunksize))
    mchunk[:, :, :numim] = mx
    mchunk[:-1, :-1, :numim] = m

    # combine images

    flatmap = np.zeros((tnrows*rows, tncols*cols))
    ci = 0
    ri = 0
    for plane in range(chunksize):
        flatmap[ri:ri+tnrows, ci:ci+tncols] = mchunk[:, :, plane]
        ri += tnrows
        # if we have filled rows rows, change column
        # and reset r
        if plane != 0 and ri == tnrows*rows:
            ci += tncols
            ri = 0

    return flatmap

# def rankTransformRDM(rdm):
#     """rank transform an RDM

#     Args:
#         rdm ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     scaler = MinMaxScaler()
#     shape = rdm.shape
#     rdm_ranks = rankdata(rdm)
#     return scaler.fit_transform(rdm_ranks.reshape(-1, 1)).reshape(shape)
