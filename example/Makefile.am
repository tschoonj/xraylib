AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include


#since the C lib will always be built, always check xrlexample1
check_PROGRAMS = xrlexample1
if ENABLE_FORTRAN
  check_PROGRAMS += xrlexample3
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
