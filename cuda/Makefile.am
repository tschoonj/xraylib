
if ENABLE_CUDA
lib_LTLIBRARIES=libxrlcuda.la
endif
libxrlcuda_la_SOURCES = xraylib-cuda.cu xraylib-cuda.h
libxrlcuda_la_CFLAGS = -I$(top_srcdir)/src -I$(top_srcdir)/include
libxrlcuda_la_LIBADD = $(CUDA_LDFLAGS) -lcudart



.cu.o:
	$(NVCC) -m64 -o $@ -c $<

.cu.lo:
	$(srcdir)/cudalt.py $@ $(NVCC) \
	--compiler-options=\"$(CFLAGS) \
	$(DEFAULT_INCLUDES) $(INCLUDES) \
	$(AM_CPPFLAGS) $(CPPFLAGS) \
	$(libxrlcuda_la_CFLAGS)\" -m64 -c $<
	


EXTRA_DIST = cudalt.py
