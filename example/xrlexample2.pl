#!/usr/bin/perl 

#Copyright (c) 2009, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#use strict;
use xraylib;



xraylib::XRayInit();
#xraylib::SetHardExit(1);



printf "Example of perl program using xraylib\n";
printf("Ca K-alpha Fluorescence Line Energy: %f\n",
	 xraylib::LineEnergy(20,$xraylib::KA_LINE ));
printf("Fe partial photoionization cs of L3 at 6.0 keV: %f\n",xraylib::CS_Photo_Partial(26,$xraylib::L3_SHELL,6.0));
printf("Zr L1 edge energy: %f\n",xraylib::EdgeEnergy(40,$xraylib::L1_SHELL));
printf("Pb Lalpha XRF production cs at 20.0 keV (jump approx): %f\n",xraylib::CS_FluorLine(82,$xraylib::LA_LINE,20.0));
printf("Pb Lalpha XRF production cs at 20.0 keV (Kissel): %f\n",xraylib::CS_FluorLine_Kissel(82,$xraylib::LA_LINE,20.0));
printf("Bi M1N2 radiative rate: %f\n",xraylib::RadRate(83,$xraylib::M1N2_LINE));
printf("U M3O3 Fluorescence Line Energy: %f\n",xraylib::LineEnergy(92,$xraylib::M3O3_LINE));
printf("Ca(HCO3)2 Rayleigh cs at 10.0 keV: %f\n",xraylib::CS_Rayl_CP("Ca(HCO3)2",10.0) );
printf("CS2 Refractive Index at 10.0 keV : %g - %g i\n",xraylib::Refractive_Index_Re("CS2",10.0,1.261),xraylib::Refractive_Index_Im("CS2",10.0,1.261));
printf("C16H14O3 Refractive Index at 1 keV : %g - %g i\n",xraylib::Refractive_Index_Re("C16H14O3",1.0,1.2),xraylib::Refractive_Index_Im("C16H14O3",1.0,1.2));
printf("SiO2 Refractive Index at 5 keV : %g - %g i\n",xraylib::Refractive_Index_Re("SiO2",5.0,2.65),xraylib::Refractive_Index_Im("SiO2",5.0,2.65));
printf("Compton profile for Fe at pz = 1.1 : %g\n",xraylib::ComptonProfile(26,1.1));
printf("M5 Compton profile for Fe at pz = 1.1 : %g\n",xraylib::ComptonProfile_Partial(26,$xraylib::M5_SHELL,1.1));


exit 0;
