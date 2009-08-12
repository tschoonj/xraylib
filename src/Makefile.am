SUBDIRS = prdata
xraylibincludedir=${includedir}/xraylib

AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include

lib_LTLIBRARIES=libxrl.la
if ENABLE_FORTRAN
  lib_LTLIBRARIES+=libxrlf03.la
endif
libxrl_la_SOURCES = xrayvars.c cross_sections.c \
scattering.c atomicweight.c edges.c fluor_lines.c fluor_yield.c jump.c \
coskron.c radrate.c cs_line.c polarized.c splint.c cs_barns.c fi.c fii.c \
kissel_pe.c xrayfiles_inline.c splint.h xrayglob.h xrayvars.h
if ENABLE_FORTRAN
 libxrlf03_la_SOURCES = xraylib_wrap.f90
 libxrlf03_la_LIBADD = libxrl.la
 xraylibinclude_HEADERS = xraylib.mod
endif

nodist_libxrl_la_SOURCES = xrayglob_inline.c

libxrl_la_LDFLAGS=-version-info @LIB_CURRENT@:@LIB_REVISION@:@LIB_AGE@






all:
	echo "dummy line"
clean-local:
	rm -f xrayglob_inline.c xraylib.mod

	
