
NULL=
check_SCRIPTS = $(NULL)
TESTS = $(check_SCRIPTS)
TEST_EXTENSIONS = .py
PY_LOG_COMPILER = $(PYTHON)
AM_PY_LOG_FLAGS =
AM_TESTS_ENVIRONMENT = LD_LIBRARY_PATH='${top_builddir}/src/.libs' DYLD_LIBRARY_PATH='${top_builddir}/src/.libs' PATH='${top_builddir}/src/.libs' PYTHONPATH='${top_builddir}/python:${top_builddir}/python/.libs' ; export LD_LIBRARY_PATH DYLD_LIBRARY_PATH PATH PYTHONPATH ;

if ENABLE_PYTHON
check_SCRIPTS += \
	test-atomiclevelwidth.py \
	test-atomicweight.py \
	test-auger.py \
	test-compoundparser.py \
	$(NULL)

endif
