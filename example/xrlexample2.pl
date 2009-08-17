#!/usr/bin/perl 



#use strict;
use xraylib;



xraylib::XRayInit();
xraylib::SetHardExit(1);



printf "Example of perl program using xraylib\n";
printf("Ca K-alpha Fluorescence Line Energy: %f\n",
	 xraylib::LineEnergy(20,$xraylib::KA_LINE ));
printf("Fe partial photoionization cs of L3 at 6.0 keV: %f\n",xraylib::CS_Photo_Partial(26,$xraylib::L3_SHELL,6.0));
printf("Zr L1 edge energy: %f\n",xraylib::EdgeEnergy(40,$xraylib::L1_SHELL));
printf("Pb Lalpha XRF production cs at 20.0 keV (jump approx): %f\n",xraylib::CS_FluorLine(82,$xraylib::LA_LINE,20.0));
printf("Pb Lalpha XRF production cs at 20.0 keV (Kissel): %f\n",xraylib::CS_FluorLine_Kissel(82,$xraylib::LA_LINE,20.0));


exit 0;
