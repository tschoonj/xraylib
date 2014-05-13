cimport xraylib_np_c as xrl
import numpy as np
cimport numpy as np
cimport cython


K_SHELL = xrl.K_SHELL
L1_SHELL = xrl.L1_SHELL

def AtomicWeight(np.ndarray[np.int_t, ndim=1] Z not None):
	#cdef np.ndarray[double] Zcopy = np.reshape(Z, Z.size, order='C')
	cdef np.ndarray[double, ndim=1, mode='c'] AW = np.empty((Z.shape[0]))
	for i in range(Z.shape[0]):
		AW[i] = xrl.AtomicWeight(Z[i])
	return AW



def XRL_1I(fun_wrap):
	def fun(np.ndarray[np.int_t, ndim=1] arg1 not None):
		cdef np.ndarray[double, ndim=1, mode='c'] rv = np.empty((arg1.shape[0]))
		for i in range(arg1.shape[0]):
			rv[i] = fun_wrap(arg1[i])
		return rv
	return fun

def _ElementDensity(np.int_t arg1): 
	return xrl.ElementDensity(arg1)

ElementDensity = XRL_1I(_ElementDensity)
