NULL=

if ENABLE_FORTRAN
check_PROGRAMS = \
	test-atomiclevelwidth \
	test-atomicweight \
	test-auger \
	test-compoundparser \
	test-comptonprofiles \
	test-crystal-diffraction \
	test-refractive_indices \
	test-nist-compounds \
	test-radionuclides \
	$(NULL)

TESTS = $(check_PROGRAMS) $(check_SCRIPTS)

LDADD = ../libxrlf03.la ../../src/libxrl.la

test_atomicweight_SOURCES = libtest.f90 test-atomicweight.f90
test_atomicweight_LDADD = $(LDADD)
test_atomicweight_CPPFLAGS = -I.. -I.
test_atomicweight_FCFLAGS = ${test_atomicweight_CPPFLAGS}

test_atomiclevelwidth_SOURCES = libtest.f90 test-atomiclevelwidth.f90
test_atomiclevelwidth_LDADD = $(LDADD)
test_atomiclevelwidth_CPPFLAGS = -I.. -I.
test_atomiclevelwidth_FCFLAGS = ${test_atomiclevelwidth_CPPFLAGS}

test_auger_SOURCES = libtest.f90 test-auger.f90
test_auger_LDADD = $(LDADD)
test_auger_CPPFLAGS = -I.. -I.
test_auger_FCFLAGS = ${test_auger_CPPFLAGS}

test_compoundparser_SOURCES = libtest.f90 test-compoundparser.f90
test_compoundparser_LDADD = $(LDADD)
test_compoundparser_CPPFLAGS = -I.. -I.
test_compoundparser_FCFLAGS = ${test_compoundparser_CPPFLAGS}

test_comptonprofiles_SOURCES = libtest.f90 test-comptonprofiles.f90
test_comptonprofiles_LDADD = $(LDADD)
test_comptonprofiles_CPPFLAGS = -I.. -I.
test_comptonprofiles_FCFLAGS = ${test_comptonprofiles_CPPFLAGS}

test_crystal_diffraction_SOURCES = libtest.f90 test-crystal_diffraction.f90
test_crystal_diffraction_LDADD = $(LDADD)
test_crystal_diffraction_CPPFLAGS = -I.. -I.
test_crystal_diffraction_FCFLAGS = ${test_crystal_diffraction_CPPFLAGS}

test_refractive_indices_SOURCES = libtest.f90 test-refractive_indices.f90
test_refractive_indices_LDADD = $(LDADD)
test_refractive_indices_CPPFLAGS = -I.. -I.
test_refractive_indices_FCFLAGS = ${test_refractive_indices_CPPFLAGS}

test_nist_compounds_SOURCES = libtest.f90 test-nist-compounds.f90
test_nist_compounds_LDADD = $(LDADD)
test_nist_compounds_CPPFLAGS = -I.. -I.
test_nist_compounds_FCFLAGS = ${test_nist_compounds_CPPFLAGS}

test_radionuclides_SOURCES = libtest.f90 test-radionuclides.f90
test_radionuclides_LDADD = $(LDADD)
test_radionuclides_CPPFLAGS = -I.. -I.
test_radionuclides_FCFLAGS = ${test_radionuclides_CPPFLAGS}

clean-local:
	rm -f libtest.mod

endif

EXTRA_DIST = meson.build