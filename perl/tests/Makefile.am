
NULL=
TEST_EXTENSIONS = .pl
PL_LOG_COMPILER = $(PERL)
AM_PL_LOG_FLAGS =
AM_TESTS_ENVIRONMENT = LD_LIBRARY_PATH='${top_builddir}/src/.libs' DYLD_LIBRARY_PATH='${top_builddir}/src/.libs' PERL5LIB='${srcdir}:${top_builddir}/perl:${top_builddir}/perl/.libs' ; export LD_LIBRARY_PATH DYLD_LIBRARY_PATH PERL5LIB ;

if ENABLE_PERL
dist_check_SCRIPTS = \
	test-atomiclevelwidth.pl \
	test-atomicweight.pl \
	test-auger.pl \
	test-compoundparser.pl \
	test-comptonprofiles.pl \
	test-nist-compounds.pl \
	test-radionuclides.pl \
	$(NULL)
endif

TESTS = $(dist_check_SCRIPTS)

EXTRA_DIST = xrltest.pm

