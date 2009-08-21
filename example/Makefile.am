AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include


#since the C lib will always be built, always check xrlexample1
check_PROGRAMS = xrlexample1
if ENABLE_FORTRAN
  check_PROGRAMS += xrlexample3
endif
check_SCRIPTS=
if ENABLE_PERL
  check_SCRIPTS += xrlexample2.sh
endif
if ENABLE_IDL
  check_SCRIPTS += xrlexample4.sh
endif
if ENABLE_PYTHON
  check_SCRIPTS += xrlexample5.sh
endif


xrlexample1_SOURCES = xrlexample1.c
xrlexample1_LDADD = ../src/libxrl.la
xrlexample1_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include

if ENABLE_FORTRAN
  xrlexample3_SOURCES = xrlexample3.f90
  xrlexample3_LDADD = ../src/libxrlf03.la
  xrlexample3_CPPFLAGS = -I../src
  xrlexample3_FFLAGS = ${xrlexample3_CPPFLAGS}
  xrlexample3_FCFLAGS = ${xrlexample3_CPPFLAGS}
endif

TESTS = $(check_PROGRAMS) $(check_SCRIPTS)

EXTRA_DIST = xrlexample4.pro xrlexample2.pl xrlexample5.py


#test the idl bindings using this script
xrlexample4.sh: 
	echo "Testing IDL bindings"
	echo "unset IDL_STARTUP" > xrlexample4.sh
	echo "LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" IDL_PATH=\"${top_srcdir}/example:${top_srcdir}/idl:<IDL_DEFAULT>\" IDL_DLM_PATH=\"../src:<IDL_DEFAULT>\" @RSIIDL_BINDIR@/idl xrlexample4" >> xrlexample4.sh
	chmod +x xrlexample4.sh	

#test the perl bindings
xrlexample2.sh:
	echo "Testing perl bindings"
#	echo "unset PERL5LIB" > xrlexample2.sh
#	echo "unset PERLLIB" >> xrlexample2.sh
	echo "LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" PERL5LIB=\"${top_builddir}/src/perl/blib/lib:${top_builddir}/src/perl/blib/arch\" $(PERL) ${top_srcdir}/example/xrlexample2.pl" >> xrlexample2.sh
	chmod +x xrlexample2.sh

#test the python bindings
xrlexample5.sh:
	echo "Testing python bindings"
	echo "LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" PYTHONPATH=\"${top_builddir}/python\" $(PYTHON) ${top_srcdir}/example/xrlexample5.py" >> xrlexample5.sh
	chmod +x xrlexample5.sh




clean-local:
	rm -rf xrlexample4.sh xrlexample2.sh xrlexample5.sh
