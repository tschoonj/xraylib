
#Copyright (c) 2009, 2010, 2011, Tom Schoonjans, David Sagan
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

xraylibincludedir=${includedir}/xraylib

AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include -I..


lib_LTLIBRARIES=libxrl.la
noinst_PROGRAMS=prdata
prdata_SOURCES = pr_data.c xrayvars.c cross_sections.c \
scattering.c atomicweight.c edges.c fluor_lines.c fluor_yield.c jump.c \
coskron.c radrate.c cs_line.c polarized.c splint.c cs_barns.c fi.c fii.c \
splint.h xrayglob.h xrayvars.h xrayglob.c xrayfiles.c comptonprofiles.c \
atomiclevelwidth.c auger_trans.c crystal_diffraction.c
prdata_CFLAGS = $(AM_CFLAGS) $(ARCHFLAGS)



if ENABLE_FORTRAN
  lib_LTLIBRARIES+=libxrlf03.la
endif
libxrl_la_SOURCES = xrayvars.c cross_sections.c \
scattering.c atomicweight.c edges.c fluor_lines.c fluor_yield.c jump.c \
coskron.c radrate.c cs_line.c polarized.c splint.c cs_barns.c fi.c fii.c \
kissel_pe.c xrayfiles_inline.c splint.h xrayglob.h xrayvars.h \
xraylib-aux.c xraylib-aux.h xraylib-parser.c cs_cp.c refractive_indices.c comptonprofiles.c \
atomiclevelwidth.c auger_trans.c xrf_cross_sections_aux.h xrf_cross_sections_aux.c \
crystal_diffraction.c
libxrl_la_CFLAGS = $(ARCHFLAGS)
if ENABLE_FORTRAN
 libxrlf03_la_SOURCES = xraylib_wrap.f90
 libxrlf03_la_LIBADD = libxrl.la
 libxrlf03_la_FCFLAGS = $(ARCHFLAGS)
 libxrlf03_la_LDFLAGS=-version-info @LIB_CURRENT@:@LIB_REVISION@:@LIB_AGE@
 nodist_xraylibinclude_HEADERS = xraylib.mod
endif

nodist_libxrl_la_SOURCES = xrayglob_inline.c

libxrl_la_LDFLAGS=-version-info @LIB_CURRENT@:@LIB_REVISION@:@LIB_AGE@ $(LDFLAGS_LIBXRL)

EXTRA_DIST = xraylib_idl.c libxrlidl.dlm xraylib.i



all: 
if ENABLE_IDL
	$(MAKE) idlxrl
endif

#$(EXEEXT) is necessary to allow compilation in cygwin
#wine is necessary when cross-compiling
xrayglob_inline.c: prdata$(EXEEXT)
	XRAYLIB_DIR=${top_srcdir} $(WINE) ./prdata$(EXEEXT)


#I'm not sure how portable this .libs link location is...
#Maybe this could be rewritten using libtool...
idlxrl:	xraylib_idl.c libxrl.la
	$(CC) -c $(AM_CPPFLAGS) -I@RSIIDL_DIR@/external/include $(CFLAGS) @IDL_CFLAGS@ ${top_srcdir}/src/xraylib_idl.c
	gcc @IDL_LD_FLAGS@ -o libxrlidl.so xraylib_idl.o -L${top_builddir}/src/.libs -lxrl



install-exec-local:
if ENABLE_IDL
	$(MKDIR_P) $(DESTDIR)$(prefix)/dlm
	$(INSTALL) libxrlidl.so ${srcdir}/libxrlidl.dlm $(DESTDIR)$(prefix)/dlm
endif



uninstall-local:
if ENABLE_IDL
	rm -f $(DESTDIR)$(prefix)/dlm/libxrlidl.so $(DESTDIR)$(prefix)/dlm/libxrlidl.dlm
endif

clean-local:
	rm -f xrayglob_inline.c xraylib.mod 
	rm -f libxrlidl.so 
	rm -f xraylib_idl.o 
	rm -rf libxrlidl.so.dSYM
