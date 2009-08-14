AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include


#since the C lib will always be built, always check xrlexample1
check_PROGRAMS = xrlexample1
if ENABLE_FORTRAN
  check_PROGRAMS += xrlexample3
endif

if ENABLE_IDL
  check_SCRIPTS = xrlexample4.sh
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

TESTS = $(check_PROGRAMS)
EXTRA_DIST = xrlexample4.pro
if ENABLE_IDL
  TESTS += xrlexample4.sh
endif


xrlexample4.sh: 
	echo "unset IDL_STARTUP" > xrlexample4.sh
	echo "LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" IDL_PATH=\"${top_srcdir}/example:${top_srcdir}/idl:<IDL_DEFAULT>\" IDL_DLM_PATH=\"../src:<IDL_DEFAULT>\" @RSIIDL_BINDIR@/idl xrlexample4" >> xrlexample4.sh
	chmod +x xrlexample4.sh	


clean-local:
	rm -rf xrlexample4.sh
