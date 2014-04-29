xraylibincludedir = ${includedir}/xraylib


if ENABLE_CUDA
lib_LIBRARIES=libxrlcuda.a
xraylibinclude_HEADERS = xraylib-cuda.h
libxrlcuda_a_SOURCES = xraylib-cuda.cu xraylib-cuda.h
libxrlcuda_a_CFLAGS = $(CUDA_CFLAGS) -I$(top_srcdir)/src -I$(top_srcdir)/include
#libxrlcuda_a_LIBADD = $(CUDA_LDFLAGS) -lcudart
libxrlcuda_a_AR = $(NVCC) -arch=sm_20 -lib -o


.cu.o:
	$(NVCC) $(libxrlcuda_a_CFLAGS) -arch=sm_20 -dc -o $@ $<

endif
#.cu.lo:
#	$(srcdir)/cudalt.py $@ $(NVCC) -arch=sm_20 -dc -m64 \
#	--compiler-options=\"$(CFLAGS) \
#       	$(DEFAULT_INCLUDES) $(INCLUDES) \
#       	$(AM_CPPFLAGS) $(CPPFLAGS) \
#       	$(libxrlcuda_la_CFLAGS)\" $(CUDA_CFLAGS) -c $<


#EXTRA_DIST = cudalt.py
