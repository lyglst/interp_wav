import interp_wav as iw
import numpy as np

nkx = nky = nkz = 4

k_vec = [[[[(i-1)/nkx, (j-1)/nky, (k-1)/nkz] for k in np.arange(1, nkz+1)] for j in np.arange(1, nky+1)] for i in np.arange(1, nkx+1)]
k_vec = np.reshape(k_vec, (-1, 3))

eig, eigv = iw.interpolate_ham_from_hr(k_vec)
