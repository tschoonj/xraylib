#Copyright (c) 2017, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include -I${top_srcdir}/src
AM_CFLAGS = $(WSTRICT_CFLAGS)

NULL=

check_PROGRAMS = \
	test-atomiclevelwidth \
	test-atomicweight \
	test-auger \
	test-compoundparser \
	test-comptonprofiles \
	test-coskron \
	test-cross_sections \
	test-crystal_diffraction \
	test-cs_barns \
	test-cs_cp \
	test-cs_line \
	test-densities \
	test-edges \
	test-fi \
	test-fii \
	test-fluor_lines \
	test-fluor_yield \
	test-jump \
	test-kissel_pe \
	test-polarized \
	test-radrate \
	test-refractive_indices \
	test-scattering \
	test-nist-compounds \
	test-radionuclides \
	test-error \
	test-version \
	$(NULL)

TESTS = $(check_PROGRAMS)

test_compoundparser_SOURCES = test-compoundparser.c
test_compoundparser_LDADD = ../src/libxrl.la

test_error_SOURCES = test-error.c
test_error_LDADD = ../src/libxrl.la

test_atomiclevelwidth_SOURCES = test-atomiclevelwidth.c
test_atomiclevelwidth_LDADD = ../src/libxrl.la

test_atomicweight_SOURCES = test-atomicweight.c
test_atomicweight_LDADD = ../src/libxrl.la

test_auger_SOURCES = test-auger.c
test_auger_LDADD = ../src/libxrl.la

test_comptonprofiles_SOURCES = test-comptonprofiles.c
test_comptonprofiles_LDADD = ../src/libxrl.la

test_coskron_SOURCES = test-coskron.c
test_coskron_LDADD = ../src/libxrl.la

test_cross_sections_SOURCES = test-cross_sections.c
test_cross_sections_LDADD = ../src/libxrl.la

test_crystal_diffraction_SOURCES = test-crystal_diffraction.c
test_crystal_diffraction_LDADD = ../src/libxrl.la $(LIBM)

test_cs_barns_SOURCES = test-cs_barns.c
test_cs_barns_LDADD = ../src/libxrl.la

test_cs_cp_SOURCES = test-cs_cp.c
test_cs_cp_LDADD = ../src/libxrl.la

test_cs_line_SOURCES = test-cs_line.c
test_cs_line_LDADD = ../src/libxrl.la

test_densities_SOURCES = test-densities.c
test_densities_LDADD = ../src/libxrl.la

test_edges_SOURCES = test-edges.c
test_edges_LDADD = ../src/libxrl.la

test_version_SOURCES = test-version.c
test_version_LDADD = ../src/libxrl.la

test_fi_SOURCES = test-fi.c
test_fi_LDADD = ../src/libxrl.la

test_fii_SOURCES = test-fii.c
test_fii_LDADD = ../src/libxrl.la

test_fluor_lines_SOURCES = test-fluor_lines.c
test_fluor_lines_LDADD = ../src/libxrl.la

test_fluor_yield_SOURCES = test-fluor_yield.c
test_fluor_yield_LDADD = ../src/libxrl.la

test_jump_SOURCES = test-jump.c
test_jump_LDADD = ../src/libxrl.la

test_kissel_pe_SOURCES = test-kissel_pe.c
test_kissel_pe_LDADD = ../src/libxrl.la

test_polarized_SOURCES = test-polarized.c
test_polarized_LDADD = ../src/libxrl.la

test_radrate_SOURCES = test-radrate.c
test_radrate_LDADD = ../src/libxrl.la

test_refractive_indices_SOURCES = test-refractive_indices.c
test_refractive_indices_LDADD = ../src/libxrl.la

test_scattering_SOURCES = test-scattering.c
test_scattering_LDADD = ../src/libxrl.la

test_nist_compounds_SOURCES = test-nist-compounds.c
test_nist_compounds_LDADD = ../src/libxrl.la

test_radionuclides_SOURCES = test-radionuclides.c
test_radionuclides_LDADD = ../src/libxrl.la

EXTRA_DIST = meson.build

clean-local:
	rm -rf test-error.dSYM

