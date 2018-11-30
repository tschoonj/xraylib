
NULL=
TEST_EXTENSIONS = .rb
RB_LOG_COMPILER = $(RUBY)
AM_RB_LOG_FLAGS =
AM_TESTS_ENVIRONMENT = LD_LIBRARY_PATH='${top_builddir}/src/.libs' DYLD_LIBRARY_PATH='${top_builddir}/src/.libs' RUBYLIB='${top_builddir}/ruby/.libs' ; export LD_LIBRARY_PATH DYLD_LIBRARY_PATH RUBYLIB ;

if ENABLE_RUBY
dist_check_SCRIPTS = \
	test-atomiclevelwidth.rb \
	test-atomicweight.rb \
	test-auger.rb \
	test-compoundparser.rb \
	test-comptonprofiles.rb \
	test-nist-compounds.rb \
	test-radionuclides.rb \
	$(NULL)
endif

TESTS = $(dist_check_SCRIPTS)
