#Copyright (c) 2009, 2010, 2011, 2012, Tom Schoonjans, David Sagan
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

xraylibincludedir=${includedir}/xraylib

AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include -I..

NULL=

if LIBXRL_CONVENIENCE_BUILD
noinst_LTLIBRARIES = libxrl.la libprdata.la
else
lib_LTLIBRARIES=libxrl.la
noinst_LTLIBRARIES = libprdata.la
endif

noinst_PROGRAMS=prdata
prdata_SOURCES = pr_data.c
prdata_LDADD = libprdata.la
libprdata_la_CFLAGS = $(AM_CFLAGS) $(ARCHFLAGS) $(WSTRICT_CFLAGS)
libprdata_la_SOURCES = \
		 xrayvars.c \
		 xrayglob.h \
		 xrayvars.h \
		 xrayglob.c \
		 xrayfiles.c \
		 fluor_yield.c \
		 coskron.c \
		 crystal_diffraction.c \
		 scattering.c \
		 fi.c \
		 fii.c \
		 splint.c \
		 splint.h \
		 atomicweight.c \
		 xraylib-error.c \
		 xraylib-error-private.h \
		 xrf_cross_sections_aux-private.h \
		 xrf_cross_sections_aux-private.c \
		 radrate.c \
		 auger_trans.c \
		 kissel_pe.c \
	     cross_sections.c \
		 xraylib-aux.c

libprdata_la_LIBADD = $(LIBM)

prdata_CFLAGS = $(AM_CFLAGS) $(ARCHFLAGS) $(WSTRICT_CFLAGS)



libxrl_la_SOURCES = xrayvars.c \
	            cross_sections.c \
		    scattering.c \
		    atomicweight.c \
		    edges.c \
		    fluor_lines.c \
		    fluor_yield.c \
		    jump.c \
		    coskron.c \
		    radrate.c \
		    cs_line.c \
		    polarized.c \
		    splint.c \
		    cs_barns.c \
		    fi.c \
		    fii.c \
		    kissel_pe.c \
		    xrayfiles_inline.c \
		    splint.h \
		    xrayglob.h \
		    xrayvars.h \
		    xraylib-aux.c \
		    xraylib-parser.c \
		    cs_cp.c \
		    refractive_indices.c \
		    comptonprofiles.c \
		    atomiclevelwidth.c \
		    auger_trans.c \
		    xrf_cross_sections_aux.h \
		    xrf_cross_sections_aux.c \
		    crystal_diffraction.c \
		    xraylib-nist-compounds.c \
		    xraylib-nist-compounds-internal.h \
		    densities.c \
		    xraylib-radionuclides-internal.h \
		    xraylib-radionuclides.c \
		    xraylib-error.c \
		    xraylib-error-private.h \
		    xraylib-deprecated-private.h \
		    $(NULL)

libxrl_la_CFLAGS = $(ARCHFLAGS) $(HIDDEN_VISIBILITY_CFLAGS) $(WSTRICT_CFLAGS)

nodist_libxrl_la_SOURCES = xrayglob_inline.c

libxrl_la_LDFLAGS=-version-info @LIB_CURRENT@:@LIB_REVISION@:@LIB_AGE@ $(LDFLAGS_LIBXRL)
libxrl_la_LIBADD = $(LIBM)

EXTRA_DIST = xraylib.i meson.build

all:

#$(EXEEXT) is necessary to allow compilation in cygwin
#wine is necessary when cross-compiling
xrayglob_inline.c: prdata$(EXEEXT)
	$(AM_V_GEN) $(WINE) ./prdata$(EXEEXT) ${top_srcdir} xrayglob_inline.c

clean-local:
	rm -rf xrayglob_inline.c xraylib.mod prdata.dSYM libxrl-$(LIB_CURRENT_MINUS_AGE).def
