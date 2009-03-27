#include <stdio.h>
#include "xraylib.h"
#include "xrayglob.h"


void XRayInit(void);
int main()
{
  int i, Z;
  float E, q;

  XRayInit();
  for (i=0; i<28; i++) {
    printf("%s\n", ShellName[i]);
  }

  for(;;) {
    printf("Z ? : ");
    scanf("%d", &Z);
    /*    
    printf("\nEnergy ? : ");
    scanf("%f", &E);
    
    printf("\nCS Photo : %e\n", CS_Photo(Z, E));
    printf("CS Rayl : %e\n", CS_Rayl(Z, E));
    printf("CS Compt : %e\n", CS_Compt(Z, E));
    printf("CS Total : %e\n", CS_Total(Z, E));
    */
    printf("\nq ? : ");
    scanf("%f", &q);
    
    printf("FF Rayl : %e\n", FF_Rayl(Z, q));
    printf("SF Compt : %e\n", SF_Compt(Z, q));
  }
  return 0;
}















