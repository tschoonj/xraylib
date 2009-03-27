#!/usr/bin/perl -w

use ExtUtils::MakeMaker;




  if ($ENV{MK_TYPE} eq "inline") {
	WriteMakefile(
	'NAME'=> 'xraylib',
	'LIBS'=> ['-lm'],
	'OBJECT'=> 'xraylib_wrap.o xrayglob_inline.o xrayfiles_inline.o
	xrayvars.o cross_sections.o scattering.o atomicweight.o edges.o
	fluor_lines.o fluor_yield.o jump.o coskron.o radrate.o 
	cs_line.o polarized.o splint.o cs_barns.o fi.o fii.o kissel_pe.o',
	'INC' => '-I../'
);
  }
  elsif ($ENV{MK_TYPE} eq "external") {
	WriteMakefile(
	'NAME'=> 'xraylib',
	'LIBS'=> ['-lm'],
	'OBJECT'=> 'xraylib_wrap.o xrayglob.o xrayfiles.o
	xrayvars.o cross_sections.o scattering.o atomicweight.o edges.o
	fluor_lines.o fluor_yield.o jump.o coskron.o radrate.o 
	cs_line.o polarized.o splint.o cs_barns.o fi.o fii.o kissel_pe.o',
	'INC' => '-I../'
);
  }
