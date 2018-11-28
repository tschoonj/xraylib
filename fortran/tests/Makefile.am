NULL=

if ENABLE_FORTRAN
check_PROGRAMS = \
	test-atomiclevelwidth \
	test-atomicweight \
	test-auger \
	test-compoundparser \
	test-comptonprofiles \
	test-refractive_indices \
	test-nist-compounds \
	$(NULL)

check_LTLIBRARIES = libtest.la

endif

TESTS = $(check_PROGRAMS) $(check_SCRIPTS)

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

dist_test_comptonprofiles_SOURCES = test-comptonprofiles.f90
nodist_test_comptonprofiles_SOURCES = libtest.mod
test_comptonprofiles_LDADD = libtest.la
test_comptonprofiles_CPPFLAGS = -I.. -I.
test_comptonprofiles_FCFLAGS = ${test_comptonprofiles_CPPFLAGS}

dist_test_refractive_indices_SOURCES = test-refractive_indices.f90
nodist_test_refractive_indices_SOURCES = libtest.mod
test_refractive_indices_LDADD = libtest.la
test_refractive_indices_CPPFLAGS = -I.. -I.
test_refractive_indices_FCFLAGS = ${test_refractive_indices_CPPFLAGS}

dist_test_nist_compounds_SOURCES = test-nist-compounds.f90
nodist_test_nist_compounds_SOURCES = libtest.mod
test_nist_compounds_LDADD = libtest.la
test_nist_compounds_CPPFLAGS = -I.. -I.
test_nist_compounds_FCFLAGS = ${test_nist_compounds_CPPFLAGS}

BUILT_SOURCES = libtest.mod

libtest.mod: libtest.la

clean-local:
	rm -f libtest.mod
