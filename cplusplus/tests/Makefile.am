
AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include -I${top_srcdir}/src -I${top_srcdir}/cplusplus

NULL=

check_PROGRAMS = \
	test-atomicweight \
    $(NULL)

TESTS = $(check_PROGRAMS)


test_atomicweight_SOURCES = test-atomicweight.cpp
test_atomicweight_LDADD = ../../src/libxrl.la