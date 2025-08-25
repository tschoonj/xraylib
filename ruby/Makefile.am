#Copyright (c) 2009, 2010, 2011, 2012, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if ENABLE_RUBY

plugindir = $(RUBYDIR)
plugin_LTLIBRARIES = xraylib.la
xraylib_la_CPPFLAGS = $(RUBY_EXT_CPPFLAGS) -DSWIGRUBY_TRICK -I${top_builddir} -I${top_srcdir}/include $(RUBY_EXT_INC) -I$(top_srcdir)/src
xraylib_la_CFLAGS = $(WSTRICT_CFLAGS) -Wno-error=deprecated-declarations
xraylib_la_LIBADD = ../src/libxrl.la
nodist_xraylib_la_SOURCES = xraylib_wrap.c
xraylib_la_LDFLAGS = -avoid-version -module -shared -export-dynamic -shrext $(RUBY_EXT_DLEXT)


xraylib_wrap.c: ${top_srcdir}/src/xraylib.i ${top_srcdir}/include/xraylib*h
	$(AM_V_GEN) $(SWIG) -includeall -I$(top_srcdir)/src -I${top_srcdir}/include -o xraylib_wrap_temp.c -ruby ${top_srcdir}/src/xraylib.i ; \
	$(SED) -e 's/^[[:space:]]*\/\/.\{1,\}//' xraylib_wrap_temp.c > xraylib_wrap.c ; \
	rm xraylib_wrap_temp.c





clean-local:
	rm -rf xraylib_wrap.c

SUBDIRS = tests
endif
