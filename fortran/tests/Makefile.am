NULL=

if ENABLE_FORTRAN
check_PROGRAMS = \
	test-atomiclevelwidth \
	test-atomicweight \
	test-auger \
	test-compoundparser \
	$(NULL)

check_LTLIBRARIES = libtest.la

TESTS = $(check_PROGRAMS) $(check_SCRIPTS)
endif

libtest_la_SOURCES = libtest.f90
libtest_la_LIBADD = ../libxrlf03.la ../../src/libxrl.la

dist_test_atomicweight_SOURCES = test-atomicweight.f90
nodist_test_atomicweight_SOURCES = libtest.mod
test_atomicweight_LDADD = libtest.la
test_atomicweight_CPPFLAGS = -I.. -I.
test_atomicweight_FCFLAGS = ${test_atomicweight_CPPFLAGS}

dist_test_atomiclevelwidth_SOURCES = test-atomiclevelwidth.f90
nodist_test_atomiclevelwidth_SOURCES = libtest.mod
test_atomiclevelwidth_LDADD = libtest.la
test_atomiclevelwidth_CPPFLAGS = -I.. -I.
test_atomiclevelwidth_FCFLAGS = ${test_atomiclevelwidth_CPPFLAGS}

dist_test_auger_SOURCES = test-auger.f90
nodist_test_auger_SOURCES = libtest.mod
test_auger_LDADD = libtest.la
test_auger_CPPFLAGS = -I.. -I.
test_auger_FCFLAGS = ${test_auger_CPPFLAGS}

dist_test_compoundparser_SOURCES = test-compoundparser.f90
nodist_test_compoundparser_SOURCES = libtest.mod
test_compoundparser_LDADD = libtest.la
test_compoundparser_CPPFLAGS = -I.. -I.
test_compoundparser_FCFLAGS = ${test_compoundparser_CPPFLAGS}

BUILT_SOURCES = libtest.mod

libtest.mod: libtest.la

clean-local:
	rm -f libtest.mod
