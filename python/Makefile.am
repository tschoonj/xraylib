
#Copyright (c) 2009, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



#python bindings for xraylib

pyexec_LTLIBRARIES = 

if ENABLE_PYTHON
pyexec_LTLIBRARIES += _xraylib.la
_xraylib_la_CFLAGS = $(PYTHON_CFLAGS) -I$(top_srcdir)/include $(PYTHON_CPPFLAGS) 
_xraylib_la_LIBADD = ../src/libxrl.la
nodist__xraylib_la_SOURCES = xraylib_wrap.c
_xraylib_la_LDFLAGS = -avoid-version -module -shared -export-dynamic

dist_bin_SCRIPTS = xraylib
nodist_python_PYTHON = xraylib.py xrayhelp.py xraymessages.py
endif

if ENABLE_PYTHON_NUMPY
pyexec_LTLIBRARIES += xraylib_np.la 
xraylib_np_la_CFLAGS = $(OPENMP_CFLAGS) $(PYTHON_CFLAGS) -I$(top_srcdir)/include $(PYTHON_CPPFLAGS) $(NUMPY_HEADERS)
xraylib_np_la_LIBADD = ../src/libxrl.la
nodist_xraylib_np_la_SOURCES = xraylib_np.c
xraylib_np_la_LDFLAGS = -avoid-version -module -shared -export-dynamic $(OPENMP_CFLAGS)
endif

xraylib_np.c: ${srcdir}/xraylib_np.pyx ${srcdir}/xraylib_np_c.pxd
	$(CYTHON) -X boundscheck=False,wraparound=False,cdivision=True -I${top_srcdir}/include -o ${builddir}/xraylib_np.c ${srcdir}/xraylib_np.pyx

xraylib_wrap.c: ${top_srcdir}/src/xraylib.i ${top_srcdir}/include/xraylib*h
	$(SWIG) -I${top_srcdir}/include -includeall -o xraylib_wrap.c -python ${top_srcdir}/src/xraylib.i

xraylib.py: xraylib_wrap.c

clean-local:
	rm -rf xraylib_wrap.c xraylib.py xraylib.pyc xraylib_np.c

distclean-local:
	rm -rf __pycache__

EXTRA_DIST = xraylib_np_c.pxd xraylib_np.pyx
