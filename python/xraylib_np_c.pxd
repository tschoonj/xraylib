cdef extern from "xraylib.h":
	double AtomicWeight(int Z)
	double ElementDensity(int Z)
	int K_SHELL "K_SHELL"
	int L1_SHELL "L1_SHELL"
