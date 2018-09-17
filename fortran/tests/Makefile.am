NULL=

check_PROGRAMS = \
	test-atomiclevelwidth \
	test-atomicweight \
	test-auger \
	test-compoundparser \
	$(NULL)

check_LTLIBRARIES = libtest.la

TESTS = $(check_PROGRAMS) $(check_SCRIPTS)

libtest_la_SOURCES = libtest.f90
libtest_la_LIBADD = ../libxrlf03.la ../../src/libxrl.la

test_atomicweight_SOURCES = test-atomicweight.f90 libtest.mod
test_atomicweight_LDADD = libtest.la
test_atomicweight_CPPFLAGS = -I.. -I.
test_atomicweight_FCFLAGS = ${test_atomicweight_CPPFLAGS}

test_atomiclevelwidth_SOURCES = test-atomiclevelwidth.f90 libtest.mod
test_atomiclevelwidth_LDADD = libtest.la
test_atomiclevelwidth_CPPFLAGS = -I.. -I.
test_atomiclevelwidth_FCFLAGS = ${test_atomiclevelwidth_CPPFLAGS}

test_auger_SOURCES = test-auger.f90 libtest.mod
test_auger_LDADD = libtest.la
test_auger_CPPFLAGS = -I.. -I.
test_auger_FCFLAGS = ${test_auger_CPPFLAGS}

test_compoundparser_SOURCES = test-compoundparser.f90 libtest.mod
test_compoundparser_LDADD = libtest.la
test_compoundparser_CPPFLAGS = -I.. -I.
test_compoundparser_FCFLAGS = ${test_compoundparser_CPPFLAGS}

BUILT_SOURCES = libtest.mod

libtest.mod: libtest.la

clean-local:
	rm -f libtest.mod
