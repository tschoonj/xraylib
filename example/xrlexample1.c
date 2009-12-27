/*
Copyright (c) 2009, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include "xraylib.h"

int main()
{
  XRayInit();
  //if something goes wrong, the test will end with EXIT_FAILURE
  SetHardExit(1);

  printf("Example of C program using xraylib\n");
  printf("Ca K-alpha Fluorescence Line Energy: %f\n",
	 LineEnergy(20,KA_LINE));
  printf("Fe partial photoionization cs of L3 at 6.0 keV: %f\n",CS_Photo_Partial(26,L3_SHELL,6.0));
  printf("Zr L1 edge energy: %f\n",EdgeEnergy(40,L1_SHELL));
  printf("Pb Lalpha XRF production cs at 20.0 keV (jump approx): %f\n",CS_FluorLine(82,LA_LINE,20.0));
  printf("Pb Lalpha XRF production cs at 20.0 keV (Kissel): %f\n",CS_FluorLine_Kissel(82,LA_LINE,20.0));
  printf("Bi M1N2 radiative rate: %f\n",RadRate(83,M1N2_LINE));

  return 0;
}
