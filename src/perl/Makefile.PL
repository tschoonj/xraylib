#!/usr/bin/perl -w

use ExtUtils::MakeMaker 6.31;




WriteMakefile(
	'NAME'=> 'xraylib',
	'OBJECT'=> 'xraylib_wrap.o',
);
