SUBDIRS = prdata .
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
 libxrlf03_la_LDFLAGS=-version-info @LIB_CURRENT@:@LIB_REVISION@:@LIB_AGE@
 xraylibinclude_HEADERS = xraylib.mod
endif

nodist_libxrl_la_SOURCES = xrayglob_inline.c

libxrl_la_LDFLAGS=-version-info @LIB_CURRENT@:@LIB_REVISION@:@LIB_AGE@

EXTRA_DIST = xraylib_idl.c libxrlidl.dlm


if ENABLE_IDL

all: 
	make idlxrl

#I'm not sure how portable this .libs link location is...
#Maybe this could be rewritten using libtool...
idlxrl:	xraylib_idl.c libxrl.la
	$(CC) @IDL_LD_FLAGS@ -I@RSIIDL_DIR@/external/include $(AM_CPPFLAGS) -o libxrlidl.so $(CFLAGS)  @IDL_CFLAGS@ ${top_srcdir}/src/xraylib_idl.c -L${top_builddir}/src/.libs -lxrl
	cp ${top_srcdir}/src/libxrlidl.dlm .

install-exec-local: idlxrl
	$(MKDIR_P) $(prefix)/dlm
	$(INSTALL) libxrlidl.so libxrlidl.dlm $(prefix)/dlm
uninstall-local:
	rm -f $(prefix)/dlm/libxrlidl.so $(prefix)/dlm/libxrlidl.dlm

endif
clean-local:
	rm -f xrayglob_inline.c xraylib.mod

	
