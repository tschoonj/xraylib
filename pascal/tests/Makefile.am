if ENABLE_PASCAL
NULL=

check_SCRIPTS = \
	test-atomiclevelwidth.sh \
	test-atomicweight.sh \
	test-auger.sh \
	test-compoundparser.sh \
	test-comptonprofiles.sh \
	test-refractive_indices.sh \
	$(NULL)

TESTS = $(check_SCRIPTS)
endif

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL) 

test-atomiclevelwidth.sh: test-atomiclevelwidth.pas
	@echo "$(FPC) -Mdelphi -FE${builddir} -Fu${top_builddir}/pascal -Fl${top_builddir}/src/.libs ${srcdir}/test-atomiclevelwidth.pas || exit 1" > test-atomiclevelwidth.sh
	@echo "PATH=\"${top_builddir}/src/.libs\" LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" ./test-atomiclevelwidth$(EXEEXT) || exit 1" >> test-atomiclevelwidth.sh
	@echo "rm -f test-atomiclevelwidth$(EXEEXT) test-atomiclevelwidth.o" >> test-atomiclevelwidth.sh

test-atomicweight.sh: test-atomicweight.pas
	@echo "$(FPC) -Mdelphi -FE${builddir} -Fu${top_builddir}/pascal -Fl${top_builddir}/src/.libs ${srcdir}/test-atomicweight.pas || exit 1" > test-atomicweight.sh
	@echo "PATH=\"${top_builddir}/src/.libs\" LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" ./test-atomicweight$(EXEEXT) || exit 1" >> test-atomicweight.sh
	@echo "rm -f test-atomicweight$(EXEEXT) test-atomicweight.o" >> test-atomicweight.sh

test-auger.sh: test-auger.pas
	@echo "$(FPC) -Mdelphi -FE${builddir} -Fu${top_builddir}/pascal -Fl${top_builddir}/src/.libs ${srcdir}/test-auger.pas || exit 1" > test-auger.sh
	@echo "PATH=\"${top_builddir}/src/.libs\" LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" ./test-auger$(EXEEXT) || exit 1" >> test-auger.sh
	@echo "rm -f test-auger$(EXEEXT) test-auger.o" >> test-auger.sh

test-compoundparser.sh: test-compoundparser.pas
	@echo "$(FPC) -Mdelphi -FE${builddir} -Fu${top_builddir}/pascal -Fl${top_builddir}/src/.libs ${srcdir}/test-compoundparser.pas || exit 1" > test-compoundparser.sh
	@echo "PATH=\"${top_builddir}/src/.libs\" LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" ./test-compoundparser$(EXEEXT) || exit 1" >> test-compoundparser.sh
	@echo "rm -f test-compoundparser$(EXEEXT) test-compoundparser.o" >> test-compoundparser.sh

test-comptonprofiles.sh: test-comptonprofiles.pas
	@echo "$(FPC) -Mdelphi -FE${builddir} -Fu${top_builddir}/pascal -Fl${top_builddir}/src/.libs ${srcdir}/test-comptonprofiles.pas || exit 1" > test-comptonprofiles.sh
	@echo "PATH=\"${top_builddir}/src/.libs\" LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" ./test-comptonprofiles$(EXEEXT) || exit 1" >> test-comptonprofiles.sh
	@echo "rm -f test-comptonprofiles$(EXEEXT) test-comptonprofiles.o" >> test-comptonprofiles.sh

test-refractive_indices.sh: test-refractive_indices.pas
	@echo "$(FPC) -Mdelphi -FE${builddir} -Fu${top_builddir}/pascal -Fl${top_builddir}/src/.libs ${srcdir}/test-refractive_indices.pas || exit 1" > test-refractive_indices.sh
	@echo "PATH=\"${top_builddir}/src/.libs\" LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" ./test-refractive_indices$(EXEEXT) || exit 1" >> test-refractive_indices.sh
	@echo "rm -f test-refractive_indices$(EXEEXT) test-refractive_indices.o" >> test-refractive_indices.sh

clean-local:
	rm -f $(check_SCRIPTS) xrltest.o  xrltest.ppu
EXTRA_DIST = xrltest.pas test-atomicweight.pas test-atomiclevelwidth.pas test-auger.pas test-compoundparser.pas test-comptonprofiles.pas test-refractive_indices.pas
