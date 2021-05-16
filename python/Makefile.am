
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
_xraylib_la_CFLAGS = $(PYTHON_CFLAGS) -I$(top_srcdir)/include $(PYTHON_CPPFLAGS) $(NUMPY_HEADERS) -I$(top_srcdir)/src $(WSTRICT_CFLAGS) $(CYTHON_ERROR_CFLAGS)
_xraylib_la_LIBADD = ../src/libxrl.la
nodist__xraylib_la_SOURCES = xraylib_wrap.c
_xraylib_la_LDFLAGS = $(LDFLAGS_PYTHON) -avoid-version -module -shared -export-dynamic -shrext $(PYTHON_EXT)
if OS_WINDOWS
_xraylib_la_LDFLAGS += $(PYTHON_LDFLAGS)
#_xraylib_la_CPPFLAGS = -mdll
endif

nodist_python_PYTHON = xraylib.py
endif

if ENABLE_PYTHON_NUMPY
pyexec_LTLIBRARIES += xraylib_np.la 
xraylib_np_la_CFLAGS = $(OPENMP_CFLAGS) $(PYTHON_CFLAGS) -I$(top_srcdir)/include $(PYTHON_CPPFLAGS) $(NUMPY_HEADERS) -I$(top_srcdir)/src $(WSTRICT_CFLAGS) $(CYTHON_ERROR_CFLAGS)
xraylib_np_la_LIBADD = ../src/libxrl.la
nodist_xraylib_np_la_SOURCES = xraylib_np.c
xraylib_np_la_LDFLAGS = $(LDFLAGS_PYTHON) -avoid-version -module -shared -export-dynamic $(OPENMP_CFLAGS) -shrext $(PYTHON_EXT)
if OS_WINDOWS
xraylib_np_la_LDFLAGS += $(PYTHON_LDFLAGS)
#xraylib_np_la_CPPFLAGS = -mdll
endif
endif

xraylib_np.c: ${srcdir}/xraylib_np.pyx ${srcdir}/xraylib_np_c.pxd
	$(AM_V_GEN) $(CYTHON) -X language_level=3,boundscheck=False,wraparound=False,cdivision=True -I${top_srcdir}/include -o ${builddir}/xraylib_np.c ${srcdir}/xraylib_np.pyx

xraylib_wrap.c: ${top_srcdir}/src/xraylib.i ${top_srcdir}/include/xraylib*h
	$(AM_V_GEN) $(SWIG) -DVERSION="'$(VERSION)'" -I$(top_srcdir)/src -I${top_srcdir}/include -includeall -I$(srcdir) -I${top_builddir} -o xraylib_wrap.c -python -py3 ${top_srcdir}/src/xraylib.i

xraylib.py: xraylib_wrap.c

clean-local:
	rm -rf xraylib_wrap.c xraylib.py xraylib.pyc xraylib_np.c

distclean-local:
	rm -rf __pycache__

if ENABLE_PYTHON
SUBDIRS = tests
endif

EXTRA_DIST = xraylib_np_c.pxd xraylib_np.pyx pyfragments.swg meson.build
