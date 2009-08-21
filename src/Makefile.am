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
 libxrlf03_la_LDFLAGS=-version-info @LIB_CURRENT@:@LIB_REVISION@:@LIB_AGE@
 xraylibinclude_HEADERS = xraylib.mod
endif

nodist_libxrl_la_SOURCES = xrayglob_inline.c

libxrl_la_LDFLAGS=-version-info @LIB_CURRENT@:@LIB_REVISION@:@LIB_AGE@

EXTRA_DIST = xraylib_idl.c libxrlidl.dlm perl/Makefile.PL perl/Makefile_perl xraylib.i



all: 
if ENABLE_IDL
	$(MAKE) idlxrl
endif
if ENABLE_PERL
	$(MAKE) perlxrl
endif




#I'm not sure how portable this .libs link location is...
#Maybe this could be rewritten using libtool...
idlxrl:	xraylib_idl.c libxrl.la
	$(CC) @IDL_LD_FLAGS@ -I@RSIIDL_DIR@/external/include $(AM_CPPFLAGS) -o libxrlidl.so $(CFLAGS)  @IDL_CFLAGS@ ${top_srcdir}/src/xraylib_idl.c -L${top_builddir}/src/.libs -lxrl
	cp ${top_srcdir}/src/libxrlidl.dlm .


perlxrl: libxrl.la xraylib.i
	cp -R ${top_srcdir}/src/perl .
	chmod a+w perl
	$(SWIG) -o perl/xraylib_wrap.c -outdir perl -perl xraylib.i
#	rm -f perl/xraylib.pm perl/xraylib_wrap.c
#	mv xraylib.pm perl
#	mv xraylib_wrap.c perl
#	echo "prefix: $(prefix)"
	$(MAKE) $(AM_MAKEFLAGS)	MAKE=$(MAKE) PERL=$(PERL) INSTALL_BASE=$(DESTDIR)$(prefix) VERSION=$(VERSION) -C perl -f Makefile_perl

install-exec-local:
if ENABLE_IDL
	$(MKDIR_P) $(DESTDIR)$(prefix)/dlm
	$(INSTALL) libxrlidl.so libxrlidl.dlm $(DESTDIR)$(prefix)/dlm
endif
if ENABLE_PERL
	$(MAKE) $(AM_MAKEFLAGS) -C perl install
endif



uninstall-local:
if ENABLE_IDL
	rm -f $(DESTDIR)$(prefix)/dlm/libxrlidl.so $(DESTDIR)$(prefix)/dlm/libxrlidl.dlm
endif
if ENABLE_PERL
#	$(MAKE) $(AM_MAKEFLAGS) -C perl uninstall
#	perl uninstall doesn't work... it's unsafe according to them.
#	So we'll have to do it ourselves
#	let's use the .packlist file for this.
	find $(DESTDIR)$(prefix) -name '.packlist' | xargs cat | xargs rm -vf
	find $(DESTDIR)$(prefix) -name '.packlist' | xargs rm -fv
	find $(DESTDIR)$(prefix) -name 'perllocal.pod' | xargs rm -fv
endif

clean-local:
	rm -f xrayglob_inline.c xraylib.mod perl/xraylib_wrap.* 
	rm -f perl/pm_to_blib
	rm -f perl/xraylib.bs
	rm -f perl/xraylib.pm
	rm -rf perl/blib
	rm -f perl/Makefile perl/Makefile_perl perl/Makefile.PL

	
