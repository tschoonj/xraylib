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

  return 0;
}
