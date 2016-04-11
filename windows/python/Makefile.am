#Copyright (c) 2010-2016, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

python:
	$(SWIG) -I${top_srcdir}/include -includeall -o xraylib_wrap.c -python ${top_srcdir}/src/xraylib.i
	cython27 -X boundscheck=False,wraparound=False -I$(top_srcdir)/include -o xraylib_np.c $(top_srcdir)/python/xraylib_np.pyx
	python27 xrlsetup.py build_ext --inplace --compiler=mingw32
	$(MKDIR_P) python2.7
	mv *.pyd python2.7/
	cython34 -X boundscheck=False,wraparound=False -I$(top_srcdir)/include -o xraylib_np.c $(top_srcdir)/python/xraylib_np.pyx
	python34 xrlsetup.py build_ext --inplace --compiler=mingw32
	$(MKDIR_P) python3.4
	mv *.pyd python3.4/
	cython35 -X boundscheck=False,wraparound=False -I$(top_srcdir)/include -o xraylib_np.c $(top_srcdir)/python/xraylib_np.pyx
	python35 xrlsetup.py build_ext --inplace --compiler=mingw32
	$(MKDIR_P) python3.5
	mv *.pyd python3.5/


SUBDIRS = cli cli_py2exe 


EXTRA_DIST = xrlsetup.py.in
