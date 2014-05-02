#Copyright (c) 2014, Tom Schoonjans and Antonio Brunetti
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans and Antonio Brunetti ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans and Antonio Brunetti BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

xraylibincludedir = ${includedir}/xraylib


if ENABLE_CUDA
lib_LIBRARIES=libxrlcuda.a
xraylibinclude_HEADERS = xraylib-cuda.h
libxrlcuda_a_SOURCES = xraylib-cuda.cu xraylib-cuda.h \
		       xraylib-cuda-private.h \
		       atomiclevelwidth.cu \
		       atomicweight.cu \
		       auger_trans.cu \
		       comptonprofiles.cu \
		       coskron.cu \
		       cross_sections.cu \
		       densities.cu \
		       edges.cu \
		       fi.cu \
		       fii.cu \
		       fluor_yield.cu \
		       jump.cu \
		       radrate.cu \
		       scattering.cu \
		       kissel_pe.cu \
		       xrf_cross_sections_aux.cu \
		       cs_line.cu \
		       splint.cu \
		       fluor_lines.cu
libxrlcuda_a_CFLAGS = $(CUDA_CFLAGS) -I$(top_srcdir)/src -I$(top_srcdir)/include
libxrlcuda_a_AR = $(NVCC) -arch=sm_20 -lib -o


.cu.o: xraylib-cuda.h xraylib-cuda-private.h
	$(NVCC) $(libxrlcuda_a_CFLAGS) -arch=sm_20 -dc -o $@ $<

endif
