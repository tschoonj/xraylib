#!/usr/bin/perl 



#use strict;
use xraylib;



xraylib::XRayInit();

printf "Example of perl program using xraylib\n";
printf("Calcium K-alpha Fluorescence Line Energy: %f\n",
	 xraylib::LineEnergy(20,0));

exit 0;
