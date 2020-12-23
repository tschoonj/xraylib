
AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include -I${top_srcdir}/src -I${top_srcdir}/cplusplus

NULL=

check_PROGRAMS = \
	test-atomicweight \
	test-atomiclevelwidth \
	test-refractive_indices \
    test-compoundparser \
    test-nist-compounds \
    test-radionuclides \
    $(NULL)

TESTS = $(check_PROGRAMS)


test_atomicweight_SOURCES = test-atomicweight.cpp
test_atomicweight_LDADD = ../../src/libxrl.la

test_atomiclevelwidth_SOURCES = test-atomiclevelwidth.cpp
test_atomiclevelwidth_LDADD = ../../src/libxrl.la

test_refractive_indices_SOURCES = test-refractive_indices.cpp
test_refractive_indices_LDADD = ../../src/libxrl.la

test_compoundparser_SOURCES = test-compoundparser.cpp
test_compoundparser_LDADD = ../../src/libxrl.la

test_radionuclides_SOURCES = test-radionuclides.cpp
test_radionuclides_LDADD = ../../src/libxrl.la

test_nist_compounds_SOURCES = test-nist-compounds.cpp
test_nist_compounds_LDADD = ../../src/libxrl.la