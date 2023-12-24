
NULL=
TEST_EXTENSIONS = .py
PY_LOG_COMPILER = $(PYTHON)
AM_PY_LOG_FLAGS =
AM_TESTS_ENVIRONMENT = EXTRA_DLL_SEARCH_PATHS='${abs_top_builddir}/src/.libs' LD_LIBRARY_PATH='${top_builddir}/src/.libs' DYLD_LIBRARY_PATH='${top_builddir}/src/.libs' PATH='${top_builddir}/src/.libs' PYTHONPATH='${top_builddir}/python:${top_builddir}/python/.libs' ; export LD_LIBRARY_PATH DYLD_LIBRARY_PATH PATH PYTHONPATH EXTRA_DLL_SEARCH_PATHS ;

if ENABLE_PYTHON
dist_check_SCRIPTS = \
	test-atomiclevelwidth.py \
	test-atomicweight.py \
	test-auger.py \
	test-comptonprofiles.py \
	test-compoundparser.py \
	test-crystal_diffraction.py \
	test-nist-compounds.py \
	test-radionuclides.py \
	test-numpy.py \
	$(NULL)
endif

TESTS = $(dist_check_SCRIPTS)
EXTRA_DIST = meson.build
