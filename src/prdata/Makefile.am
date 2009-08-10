AM_CPPFLAGS= -I${top_builddir}/include -I${top_srcdir}/include -I${top_builddir}/src -I${top_srcdir}/src



noinst_PROGRAMS = prdata
prdata_SOURCES = pr_data.c ../xrayvars.c ../cross_sections.c \
../scattering.c ../atomicweight.c ../edges.c ../fluor_lines.c ../fluor_yield.c ../jump.c \
../coskron.c ../radrate.c ../cs_line.c ../polarized.c ../splint.c ../cs_barns.c ../fi.c ../fii.c \
../kissel_pe.c ../splint.h ../xrayglob.h ../xrayvars.h ../xrayglob.c ../xrayfiles.c

all: prdata 
	XRAYLIB_DIR=${top_srcdir} ./prdata
clean-local:
	rm -f prdata
