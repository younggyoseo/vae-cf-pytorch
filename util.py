import numpy as np
from scipy.sparse import csr_matrix, issparse


def sparse_info(m):
    print("{} of {}".format(type(m), m.dtype))
    print("shape = {}, nnz = {}".format(m.shape, m.nnz))
    print("density = {:.3}".format(m.nnz / np.prod(m.shape)))


def prune_global(x, target_density=0.005):

    target_nnz = int(target_density * np.prod(x.shape))
    if issparse(x):
        x_sp = x.copy().tocsr()
        x_sp.eliminate_zeros()
        if x_sp.nnz <= target_nnz:
            return x_sp

        thr = np.partition(np.abs(x_sp.data), kth=-target_nnz)[-target_nnz]
        x_sp.data[np.abs(x_sp.data) < thr] = 0.0
    else:
        thr = np.quantile(np.abs(x), 1.0-target_density)
        x[np.abs(x) < thr] = 0.0
        x_sp = csr_matrix(x)
    x_sp.eliminate_zeros()

    return x_sp
