#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_distances
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np 
from numpy.testing import assert_array_almost_equal
from scipy.spatial.distance import cdist
from searchlight.searchlight import get_rdm, upper_tri_indexing
from searchlight.utils import corr_rdms


def get_rdm_cdist(data, ind):
    """get_distance_cdist returns the correlation distance 
       across condition patterns in X
       get_distance_cdist uses scipy's cdist method
    
    Args:
        data (array): conditions x all channels.
        ind (vector): subspace of voxels for that sphere.
    
    Returns:
        UTV: pairwise distances between condition patterns in X 
             (in upper triangular vector form)
    """
    ind = np.array(ind)
    X = data[ind,:].T
    return upper_tri_indexing(cdist(X, X, 'correlation'))

class TestDistances(unittest.TestCase): 

    def test_get_rdm(self):
        """ test_get_distance 
            here we simulate a fMRI measurement with
            100 voxels and 36 conditions.
            We also simulate 5 pseudo center spheres.
            We use get_distances to return an RDM for 
            one of the so-called center spheres. 
            and assess its shape.
        """
        data = np.random.rand(100, 36)
        centers = [np.random.choice(range(100), size=10) for x in range(5)]

        d = get_rdm(data, centers[0])
        d2 = get_rdm_cdist(data, centers[0])

        assert_array_almost_equal(d, d2)

    def test_corr_rdms(self):
        """ Here we simulate 1000 centers with a 36x36 vectorised RDM (1, 630) in each.
            We also simulate a model RDM utv.
            We use corr_rdms to return the pearson correlation 
            of the model to all center RDMs 
            we do the same with 1 - cdist to test.
        """
        X = np.random.rand(1000,630)
        Y = np.random.rand(1,630)
        d = corr_rdms(X, Y)

        d2 = 1 - cdist(X, Y,
                    metric='correlation')
        
        assert_array_almost_equal(d, d2)





  
