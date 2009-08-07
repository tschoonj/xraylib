#include <stdio.h>
#include "xraylib.h"

int main()
{
  XRayInit();
  //if something goes wrong, the test will end with EXIT_FAILURE
  SetHardExit(1);

  printf("Example of C program using xraylib\n");
  printf("Calcium K-alpha Fluorescence Line Energy: %f\n",
	 LineEnergy(20,KA_LINE));

  return 0;
}
