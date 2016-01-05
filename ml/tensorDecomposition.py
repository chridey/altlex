import collections

import numpy as np

import sktensor
import ncp

def build(iterator,
          verbose=False):

    cooccurrenceCounts = collections.defaultdict(float)
    
    for feature in iterator:
        weight = iterator.weighting(feature)
        if weight == 0:
            continue
        
        cooccurrenceCounts[feature] += weight

    return cooccurrenceCounts

def decompose(cooccurrenceCounts,
              shape,
              rank=25,
              maxIterations=100,
              verbose=False,
              method='als'):

    keys = tuple(list(i) for i in zip(*cooccurrenceCounts.iterkeys()))
    tensor = sktensor.sptensor(keys,
                               list(cooccurrenceCounts.itervalues()),
                               shape=shape)

    if verbose:
        print(tensor.shape)
        print(tensor.nnz())

    #initialize using higher order SVD
    #https://en.wikipedia.org/wiki/Higher-order_singular_value_decomposition
    Uinit = [None for _ in range(tensor.ndim)]
    if verbose:
        print('initializing...')
    for n in range(tensor.ndim):
        if verbose:
            print(n)
        Uinit[n] = np.array(sktensor.core.nvecs(tensor, n, rank),
                            dtype=np.float)
    #do non-negative tensor factorization using alternating least squares
    if verbose:
        print('decomposing...')
    decomposition = ncp.nonnegative_tensor_factorization(tensor,
                                                         rank,
                                                         max_iter=maxIterations,
                                                         init=Uinit,
                                                         verbose=verbose)
    return decomposition

def save(filename, decomposition, rank):
    filenames = {}
    filename = 'lambda_{}'.format(rank)
    filenames[filename] = decomposition.lmbda
    for i in range(len(decomposition.U)):
        filename = 'U{}_{}'.format(i, rank)
        filenames[filename] = decomposition.U[i]

    np.savez_compressed('{}.npz'.format(filename), **filenames)
