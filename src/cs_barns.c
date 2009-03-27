#include "xrayglob.h"
#include "xraylib.h"

//////////////////////////////////////////////////////////////////////
//                                                                  //
//                  Cross sections in barn/atom                     //
//  (see cross_sections.c, scattering.c and polarized.c)            //
//                                                                  //
//////////////////////////////////////////////////////////////////////

float CSb_Total(int Z, float E)
{
  return CS_Total(Z, E)*AtomicWeight_arr[Z]/AVOGNUM;
}

float CSb_Photo(int Z, float E)
{
  return CS_Photo(Z, E)*AtomicWeight_arr[Z]/AVOGNUM;
}

float CSb_Rayl(int Z, float E)
{
  return CS_Rayl(Z, E)*AtomicWeight_arr[Z]/AVOGNUM;
}

float CSb_Compt(int Z, float E)
{
  return CS_Compt(Z, E)*AtomicWeight_arr[Z]/AVOGNUM;
}

float CSb_FluorLine(int Z, int line, float E)
{
  return CS_FluorLine(Z, line, E)*AtomicWeight_arr[Z]/AVOGNUM;
}

float DCSb_Rayl(int Z, float E, float theta)
{
  return DCS_Rayl(Z, E, theta)*AtomicWeight_arr[Z]/AVOGNUM;
}

float DCSb_Compt(int Z, float E, float theta)
{
  return DCS_Compt(Z, E, theta)*AtomicWeight_arr[Z]/AVOGNUM;
}

float DCSPb_Rayl(int Z, float E, float theta, float phi)
{
  return DCSP_Rayl(Z, E, theta, phi)*AtomicWeight_arr[Z]/AVOGNUM;
}

float DCSPb_Compt(int Z, float E, float theta, float phi)
{
  return DCSP_Compt(Z, E, theta, phi)*AtomicWeight_arr[Z]/AVOGNUM;
}

