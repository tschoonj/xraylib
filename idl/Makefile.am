
#Copyright (c) 2009, 2010, 2011, 2012, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

xraylibincludedir=${includedir}/xraylib

AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include -I.. -I${top_srcdir}/src

if ENABLE_IDL

dlmdir = $(prefix)/dlm
dist_dlm_DATA = libxrlidl.dlm
prodir = $(prefix)/pro
dist_pro_DATA = xraylib_nist_compounds.pro  xraylib_auger.pro xraylib_shells.pro  xraylib_help.pro xraylib_lines.pro xraylib.pro xraylib_radionuclides.pro

plugindir = $(dlmdir)
plugin_LTLIBRARIES = libxrlidl.la
libxrlidl_la_CFLAGS = -I@RSIIDL_INCDIR@ $(WSTRICT_CFLAGS)
libxrlidl_la_LIBADD = ../src/libxrl.la @IDL_LIBADD@
libxrlidl_la_SOURCES = xraylib_idl.c
libxrlidl_la_LDFLAGS = -avoid-version -module -shared -export-dynamic $(LDFLAGS_IDL)

endif



