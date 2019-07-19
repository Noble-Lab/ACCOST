import numpy as np
import csv
import gzip
import logging
import copy
import scipy
from sets import Set
from scipy import sparse
from scipy.sparse import coo_matrix



class matrixSet:
    """
    Class to represent a set of contact count matrices.
    This could be a set of replicates, or pseudo-replicates.
    
    The set of bins and bad/filtered/low-mappability bins must be
    the same for the entire set of matrices.
    
    Args:
       allBins: dictionary of chromosome + midpoints to bin index
       allBins_reversed: list of indices, each pointing to a (chr,mid) tuple
       badBins: list of indices of bad (low mappability) bins

    """

    def __init__(self,allBins,allBins_reversed,badBins):
        self.allBins = allBins
        self.allBins_reversed = allBins_reversed
        self.badBins = badBins
        self.matrices = Set()
        self.n_matrices = 0

    def add_matrix(self,counts):
        """
        Add a contact count matrix object to the set of matrices.
        """
        self.matrices.add(counts)
        self.n_matrices = self.n_matrices + 1
    
