#Copyright (c) 2012, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


xraylibincludedir=${includedir}/xraylib


if ENABLE_FORTRAN
lib_LTLIBRARIES=libxrlf03.la
libxrlf03_la_SOURCES = xraylib_wrap.F90
libxrlf03_la_LIBADD = ../src/libxrl.la
libxrlf03_la_FCFLAGS = $(ARCHFLAGS)
libxrlf03_la_LDFLAGS=$(LDFLAGS_FORTRAN) -version-info @LIB_CURRENT@:@LIB_REVISION@:@LIB_AGE@
nodist_xraylibinclude_HEADERS = xraylib.mod

xraylib.mod: libxrlf03.la

xraylib_wrap.F90: xraylib_wrap_generated.F90

clean-local:
	rm -f xraylib.mod

SUBDIRS = tests

endif

EXTRA_DIST = xraylib_wrap_generated.F90 meson.build install_fortran_mod.py
