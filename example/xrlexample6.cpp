/*
Copyright (c) 2010, 2011 Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstdio>
#include <cstdlib>
#include "xraylib.h"

int main()
{
  struct compoundData cdtest;
  int i;
  XRayInit();
  //if something goes wrong, the test will end with EXIT_FAILURE
  //SetHardExit(1);

  std::printf("Example of C++ program using xraylib\n");
  std::printf("Ca K-alpha Fluorescence Line Energy: %f\n",
	 LineEnergy(20,KA_LINE));
  std::printf("Fe partial photoionization cs of L3 at 6.0 keV: %f\n",CS_Photo_Partial(26,L3_SHELL,6.0));
  std::printf("Zr L1 edge energy: %f\n",EdgeEnergy(40,L1_SHELL));
  std::printf("Pb Lalpha XRF production cs at 20.0 keV (jump approx): %f\n",CS_FluorLine(82,LA_LINE,20.0));
  std::printf("Pb Lalpha XRF production cs at 20.0 keV (Kissel): %f\n",CS_FluorLine_Kissel(82,LA_LINE,20.0));
  std::printf("Bi M1N2 radiative rate: %f\n",RadRate(83,M1N2_LINE));
  std::printf("U M3O3 Fluorescence Line Energy: %f\n",LineEnergy(92,M3O3_LINE));
  //parser test for Ca(HCO3)2 (calcium bicarbonate)
  if (CompoundParser("Ca(HCO3)2",&cdtest) == 0)
	return 1;
  std::printf("Ca(HCO3)2 contains %i atoms and %i elements\n",cdtest.nAtomsAll,cdtest.nElements);
  for (i = 0 ; i < cdtest.nElements ; i++)
    std::printf("Element %i: %lf %%\n",cdtest.Elements[i],cdtest.massFractions[i]*100.0);

  FREE_COMPOUND_DATA(cdtest)

  //parser test for SiO2 (quartz)
  if (CompoundParser("SiO2",&cdtest) == 0)
	return 1;

  std::printf("SiO2 contains %i atoms and %i elements\n",cdtest.nAtomsAll,cdtest.nElements);
  for (i = 0 ; i < cdtest.nElements ; i++)
    std::printf("Element %i: %lf %%\n",cdtest.Elements[i],cdtest.massFractions[i]*100.0);

  FREE_COMPOUND_DATA(cdtest)

  std::printf("Ca(HCO3)2 Rayleigh cs at 10.0 keV: %f\n",CS_Rayl_CP("Ca(HCO3)2",10.0f) );

  std::printf("CS2 Refractive Index at 10.0 keV : %f - %f i\n",Refractive_Index_Re("CS2",10.0f,1.261f),Refractive_Index_Im("CS2",10.0f,1.261f));
  std::printf("C16H14O3 Refractive Index at 1 keV : %f - %f i\n",Refractive_Index_Re("C16H14O3",1.0f,1.2f),Refractive_Index_Im("C16H14O3",1.0f,1.2f));
  std::printf("SiO2 Refractive Index at 5 keV : %f - %f i\n",Refractive_Index_Re("SiO2",5.0f,2.65f),Refractive_Index_Im("SiO2",5.0f,2.65f));
  std::printf("Compton profile for Fe at pz = 1.1 : %f\n",ComptonProfile(26,1.1f));
  std::printf("M5 Compton profile for Fe at pz = 1.1 : %f\n",ComptonProfile_Partial(26,M5_SHELL,1.1f));
  std::printf("K atomic level width for Fe: %f\n", AtomicLevelWidth(26,K_SHELL));

  std::printf("Bi L2-M5M5 Auger non-radiative rate: %f\n",AugerRate(86,L2_M5M5_AUGER));

  return 0;
}

