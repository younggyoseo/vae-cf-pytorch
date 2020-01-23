import numpy as np


def gramm_matrix(x):

    return x.T.dot(x)


def closed_form_slim(x, l2_reg=500):

    gramm = gramm_matrix(x).toarray()
    diag_indices = np.diag_indices(gramm.shape[0])
    gramm[diag_indices] += l2_reg
    inv_gramm = np.linalg.inv(gramm)
    b = inv_gramm / (-np.diag(inv_gramm))
    b[diag_indices] = 0.0

    return b
