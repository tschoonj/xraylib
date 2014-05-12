cimport xraylib_np_c as xrl
import numpy as np
cimport numpy as np
cimport cython


K_SHELL = xrl.K_SHELL
L1_SHELL = xrl.L1_SHELL

@cython.boundscheck(False)
@cython.wraparound(False)
def AtomicWeight(np.ndarray[np.int_t, ndim=1] Z not None):
	#cdef np.ndarray[double] Zcopy = np.reshape(Z, Z.size, order='C')
	cdef np.ndarray[double, ndim=1, mode='c'] AW = np.empty((Z.shape[0]))
	for i in range(Z.shape[0]):
		AW[i] = xrl.AtomicWeight(Z[i])
	return AW

