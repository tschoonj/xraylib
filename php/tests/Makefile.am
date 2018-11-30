
NULL=
TEST_EXTENSIONS = .php
PHP_LOG_COMPILER = $(PHP)
AM_PHP_LOG_FLAGS = -d extension=${top_builddir}/php/.libs/xraylib.so -d include_path=${top_builddir}/php/:${srcdir}
AM_TESTS_ENVIRONMENT = LD_LIBRARY_PATH='${top_builddir}/src/.libs' DYLD_LIBRARY_PATH='${top_builddir}/src/.libs' ; export LD_LIBRARY_PATH DYLD_LIBRARY_PATH ;

if ENABLE_PHP
dist_check_SCRIPTS = \
	test-atomiclevelwidth.php \
	test-atomicweight.php \
	test-auger.php \
	test-compoundparser.php \
	test-comptonprofiles.php \
	test-nist-compounds.php \
	test-radionuclides.php \
	$(NULL)
endif

TESTS = $(dist_check_SCRIPTS)

EXTRA_DIST = xrltest.php

