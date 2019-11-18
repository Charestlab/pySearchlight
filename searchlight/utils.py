import numpy as np
from math import floor, ceil
from scipy.spatial.distance import cdist, squareform
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import svm
import cv2
import itertools



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
              indexing, i.e. you may want the frist chunk, or the second
              chunk, but not the zeroth chunk)
    Returns:
        [numpy array object]:

        return a numpy array object of chunks.  the last vector
        may have fewer than <num> elements.

        also return the beginning and ending indices associated with
        this chunk in <xbegin> and <xend>.

    Examples:

        a = np.empty((2,), dtype=np.object)
        a[0] = [1, 2, 3]
        a[1] = [4, 5]
        assert(np.all(chunking(list(np.arange(5)+1),3)==a))

        assert(chunking([4, 2, 3], 2, 2)==([3], 3, 3))


        # do in chunks
        chunks = chunking(
            list(range(mflat.shape[1])), int(np.ceil(mflat.shape[1]/numchunks)))

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
        return np.asarray(f[num-1]), xbegin, xend

def isnotfinite(arr):
    res = np.isfinite(arr)
    np.bitwise_not(res, out=res)  # in-place
    return res

def sample_conditions(conditions, n_samples, replace=False):
    
    unique_conditions = np.unique(conditions)
    
    choices = np.random.choice(unique_conditions, n_samples, replace=replace)

    conditions_bool = np.any(np.array([conditions == v for v in choices]), axis=0)

    return conditions_bool

def average_over_conditions(data, conditions, conditions_to_avg):
    
    lookup =  np.unique(conditions_to_avg)
    n_conds = lookup.shape[0]
    n_dims = data.ndim

    if n_dims==2:
        n_voxels, _ = data.shape
        avg_data = np.empty((n_voxels, n_conds))
    else:
        x, y, z, _ = data.shape 
        avg_data = np.empty((x,y,z, n_conds))

    for j, x in enumerate(lookup):

        conditions_bool = conditions==x 
        if n_dims ==2:
            avg_data[:,j] = data[:, conditions_bool].mean(axis=1)
        else:
            avg_data[:,:,:,j] = data[:, :, :, conditions_bool].mean(axis=3)

    return avg_data


def makeimagestack(m):

    """
    def makeimagestack(m)

    <m> is a 3D matrix.  if more than 3D, we reshape to be 3D.
    we automatically convert to double format for the purposes of this method.
    try to make as square as possible
    (e.g. for 16 images, we would use [4 4]).
    find the minimum possible to fit all the images in.
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

    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    fig, ax = plt.subplots()
    im = ax.imshow(flatmap, cmap=cm.coolwarm)
    plt.show()
    """

def resizeImages(imageArray, resx, resy):
    nimages, x, y, rgb = imageArray.shape
    res = []
    for image in range(nimages):
        res.append(cv2.resize(imageArray[image, :, :, :], dsize=(
            resx, resy), interpolation=cv2.INTER_CUBIC))
    return np.asarray(res)


def rankTransformRDM(rdm):
    """[summary]

    Args:
        rdm ([type]): [description]

    Returns:
        [type]: [description]
    """
    scaler = MinMaxScaler()
    shape = rdm.shape
    rdm_ranks = rankdata(rdm)
    return scaler.fit_transform(rdm_ranks.reshape(-1, 1)).reshape(shape)


def upper_tri_indexing(A):
    # returns the upper triangle
    m = A.shape[0]
    r, c = np.triu_indices(m, 1)
    return A[r, c]