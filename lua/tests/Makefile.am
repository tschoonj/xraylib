
NULL=
TEST_EXTENSIONS = .lua
LUA_LOG_COMPILER = $(LUA)
AM_LUA_LOG_FLAGS =
AM_TESTS_ENVIRONMENT = LUA_PATH='${srcdir}/?.lua;;' LUA_CPATH='${top_builddir}/lua/.libs/?.so' LD_LIBRARY_PATH='${top_builddir}/src/.libs' DYLD_LIBRARY_PATH='${top_builddir}/src/.libs' PATH='${top_builddir}/src/.libs' PYTHONPATH='${top_builddir}/python:${top_builddir}/python/.libs' ; export LD_LIBRARY_PATH DYLD_LIBRARY_PATH PATH PYTHONPATH LUA_CPATH LUA_PATH ;

if ENABLE_LUA
dist_check_SCRIPTS = \
	test-atomiclevelwidth.lua \
	test-atomicweight.lua \
	test-auger.lua \
	test-compoundparser.lua \
	test-comptonprofiles.lua \
	test-nist-compounds.lua \
	test-radionuclides.lua \
	$(NULL)

endif

TESTS = $(dist_check_SCRIPTS)

EXTRA_DIST = luaunit.lua
