/*
Copyright (c) 2014, Tom Schoonjans and Antonio Brunetti
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans and Antonio Brunetti ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans and Antonio Brunetti BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "xraylib-cuda.h"
#include "xraylib-cuda-private.h"
#include "xraylib-defs.h"
#include "xraylib.h"

__device__ int NE_Photo_Total_Kissel_d[ZMAX+1];
__device__ double E_Photo_Total_Kissel_d[(ZMAX+1)*NE_PHOTO_TOTAL_KISSEL_MAX];
__device__ double Photo_Total_Kissel_d[(ZMAX+1)*NE_PHOTO_TOTAL_KISSEL_MAX];
__device__ double Photo_Total_Kissel2_d[(ZMAX+1)*NE_PHOTO_TOTAL_KISSEL_MAX];

__device__ double Electron_Config_Kissel_d[(ZMAX+1)*SHELLNUM_K];
__device__ double EdgeEnergy_Kissel_d[(ZMAX+1)*SHELLNUM_K];

__device__ int NE_Photo_Partial_Kissel_d[(ZMAX+1)*SHELLNUM_K];
__device__ double E_Photo_Partial_Kissel_d[(ZMAX+1)*SHELLNUM_K*NE_PHOTO_PARTIAL_KISSEL_MAX];
__device__ double Photo_Partial_Kissel_d[(ZMAX+1)*SHELLNUM_K*NE_PHOTO_PARTIAL_KISSEL_MAX];
__device__ double Photo_Partial_Kissel2_d[(ZMAX+1)*SHELLNUM_K*NE_PHOTO_PARTIAL_KISSEL_MAX];


__device__ double CSb_Photo_Total_cu(int Z, double E) {
  int shell;
  double rv = 0.0;

  if (Z<1 || Z>ZMAX || NE_Photo_Total_Kissel_d[Z] < 0) {
    return 0.0;
  }
  if (E <= 0.) {
	  return 0.0;
  }
  for (shell = K_SHELL ; shell <= Q3_SHELL ; shell++) {
    if (Electron_Config_Kissel_d[Z*SHELLNUM_K+shell] > 1.0E-06 && E >= EdgeEnergy_arr_d[Z*SHELLNUM+shell] ) {
  	rv += CSb_Photo_Partial_cu(Z,shell,E)*Electron_Config_Kissel_d[Z*SHELLNUM_K+shell];
    }
  }
  return rv;
}

__device__ double CS_Photo_Total_cu(int Z, double E) {
  return CSb_Photo_Total_cu(Z, E)*AVOGNUM/AtomicWeight_arr_d[Z];
}

__device__ double CSb_Photo_Partial_cu(int Z, int shell, double E) {
  double ln_E, ln_sigma, sigma;
  double x0, x1, y0, y1;
  double m;

  if (Z < 1 || Z > ZMAX) {
    return 0.0;
  }
  if (shell < 0 || shell >= SHELLNUM_K) {
    return 0.0;
  }
  if (E <= 0.0) {
    return 0.0;
  }
  if (Electron_Config_Kissel_d[Z*SHELLNUM_K+shell] < 1.0E-06){
    return 0.0;
  } 
  
  if (EdgeEnergy_arr_d[Z*SHELLNUM+shell] > E) {
    return 0.0;
  } 
  else {
    ln_E = log(E);
    if (EdgeEnergy_Kissel_d[Z*SHELLNUM_K+shell] > EdgeEnergy_arr_d[Z*SHELLNUM+shell] && E < EdgeEnergy_Kissel_d[Z*SHELLNUM_K+shell]) {
   	/*
	 * use log-log extrapolation 
	 */
	x0 = E_Photo_Partial_Kissel_d[Z*SHELLNUM_K*NE_PHOTO_PARTIAL_KISSEL_MAX+NE_PHOTO_PARTIAL_KISSEL_MAX*shell+0];
	x1 = E_Photo_Partial_Kissel_d[Z*SHELLNUM_K*NE_PHOTO_PARTIAL_KISSEL_MAX+NE_PHOTO_PARTIAL_KISSEL_MAX*shell+1];
	y0 = Photo_Partial_Kissel_d[Z*SHELLNUM_K*NE_PHOTO_PARTIAL_KISSEL_MAX+NE_PHOTO_PARTIAL_KISSEL_MAX*shell+0];
	y1 = Photo_Partial_Kissel_d[Z*SHELLNUM_K*NE_PHOTO_PARTIAL_KISSEL_MAX+NE_PHOTO_PARTIAL_KISSEL_MAX*shell+1];
	/*
	 * do not allow "extreme" slopes... force them to be within -1;1
	 */
	m = (y1-y0)/(x1-x0);
	if (m > 1.0)
		m=1.0;
	else if (m < -1.0)
		m=-1.0;
	ln_sigma = y0+m*(ln_E-x0);
    }
    else {
        int offset = NE_PHOTO_PARTIAL_KISSEL_MAX*(Z*SHELLNUM_K+shell);
    	ln_sigma = splint_cu(E_Photo_Partial_Kissel_d+offset+-1, Photo_Partial_Kissel_d+offset-1, Photo_Partial_Kissel2_d+offset-1,NE_Photo_Partial_Kissel_d[Z*SHELLNUM_K+shell], ln_E);
   }
   sigma = exp(ln_sigma);

   return sigma; 

  }
}

__device__ double CS_Photo_Partial_cu(int Z, int shell, double E) {
  return CSb_Photo_Partial_cu(Z, shell, E)*Electron_Config_Kissel_d[Z*SHELLNUM_K+shell]*AVOGNUM/AtomicWeight_arr_d[Z];
}

__device__ double CS_FluorLine_Kissel_cu(int Z, int line, double E) {
	return CS_FluorLine_Kissel_Cascade_cu(Z, line, E);
}

__device__ double CSb_FluorLine_Kissel_cu(int Z, int line, double E) {
  return CS_FluorLine_Kissel_Cascade_cu(Z, line, E)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double CS_Total_Kissel_cu(int Z, double E) { 

  if (Z<1 || Z>ZMAX || NE_Photo_Total_Kissel_d[Z]<0 || NE_Rayl_d[Z]<0 || NE_Compt_d[Z]<0) {
    return 0.0;
  }

  if (E <= 0.) {
    return 0.0;
  }

  return CS_Photo_Total_cu(Z, E) + CS_Rayl_cu(Z, E) + CS_Compt_cu(Z, E);

}

__device__ double CSb_Total_Kissel_cu(int Z, double E) {

  return CS_Total_Kissel_cu(Z,E)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double ElectronConfig_cu(int Z, int shell) {

  if (Z<1 || Z>ZMAX  ) {
    return 0.0;
  }

  if (shell < 0 || shell >= SHELLNUM_K ) {
    return 0.0;
  }

  return Electron_Config_Kissel_d[Z*SHELLNUM_K+shell]; 

}

__device__ double CS_FluorLine_Kissel_no_Cascade_cu(int Z, int line, double E) {
  double PL1, PL2, PM1, PM2, PM3, PM4;

  PL1 = PL2 = PM1 = PM2 = PM3 = PM4 = 0.0;


  if (Z<1 || Z>ZMAX) {
    return 0.0;
  }

  if (E <= 0.) {
    return 0.0;
  }

  if (line>=KN5_LINE && line<=KB_LINE) {
    /*
     * K lines -> never cascade effect!
     */
    return CS_Photo_Partial_cu(Z, K_SHELL, E)*FluorYield_cu(Z, K_SHELL)*RadRate_cu(Z,line);
  }
  else if (line>=L1P5_LINE && line<=L1M1_LINE) {
    /*
     * L1 lines
     */
    return PL1_pure_kissel_cu(Z,E)*FluorYield_cu(Z, L1_SHELL)*RadRate_cu(Z,line);
  }
  else if (line>=L2Q1_LINE && line<=L2M1_LINE) {
    /*
     * L2 lines
     */
    PL1 = PL1_pure_kissel_cu(Z,E);
    return (FluorYield_cu(Z, L2_SHELL)*RadRate_cu(Z,line))*
		PL2_pure_kissel_cu(Z, E, PL1);
  }
  else if (line>=L3Q1_LINE && line<=L3M1_LINE) {
    /*
     * L3 lines
     */
    PL1 = PL1_pure_kissel_cu(Z,E);
    PL2 = PL2_pure_kissel_cu(Z, E, PL1);
    return (FluorYield_cu(Z, L3_SHELL)*RadRate_cu(Z,line))*PL3_pure_kissel_cu(Z, E, PL1, PL2);
  }
  /*else if (line == LA_LINE) {
    return (CS_FluorLine_Kissel_no_Cascade_cu(Z,L3M4_LINE,E)+CS_FluorLine_Kissel_no_Cascade_cu(Z,L3M5_LINE,E)); 
  }*/
  /*else if (line == LB_LINE) {
    return (CS_FluorLine_Kissel_no_Cascade_cu(Z,L2M4_LINE,E)+
    	CS_FluorLine_Kissel_no_Cascade_cu(Z,L2M3_LINE,E)+
        CS_FluorLine_Kissel_no_Cascade_cu(Z,L3N5_LINE,E)+
        CS_FluorLine_Kissel_no_Cascade_cu(Z,L3O4_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade_cu(Z,L3O5_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade_cu(Z,L3O45_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade_cu(Z,L3N1_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade_cu(Z,L3O1_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade_cu(Z,L3N6_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade_cu(Z,L3N7_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade_cu(Z,L3N4_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade_cu(Z,L1M3_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade_cu(Z,L1M2_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade_cu(Z,L1M5_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade_cu(Z,L1M4_LINE,E)
    );
  }*/
  else if (line>=M1P5_LINE && line<=M1N1_LINE) {
    /*
     * M1 lines
     */
    return PM1_pure_kissel_cu(Z, E)*FluorYield_cu(Z, M1_SHELL)*RadRate_cu(Z,line);
  }
  else if (line>=M2P5_LINE && line<=M2N1_LINE) {
    /*
     * M2 lines
     */
    PM1 = PM1_pure_kissel_cu(Z, E);
    return (FluorYield_cu(Z, M2_SHELL)*RadRate_cu(Z,line))*
		PM2_pure_kissel_cu(Z, E, PM1);
  }
  else if (line>=M3Q1_LINE && line<=M3N1_LINE) {
    /*
     * M3 lines
     */
    PM1 = PM1_pure_kissel_cu(Z, E);
    PM2 = PM2_pure_kissel_cu(Z, E, PM1);
    return (FluorYield_cu(Z, M3_SHELL)*RadRate_cu(Z,line))*
		PM3_pure_kissel_cu(Z, E, PM1, PM2);
  }
  else if (line>=M4P5_LINE && line<=M4N1_LINE) {
    /*
     * M4 lines
     */
    PM1 = PM1_pure_kissel_cu(Z, E);
    PM2 = PM2_pure_kissel_cu(Z, E, PM1);
    PM3 = PM3_pure_kissel_cu(Z, E, PM1, PM2);
    return (FluorYield_cu(Z, M4_SHELL)*RadRate_cu(Z,line))*
		PM4_pure_kissel_cu(Z, E, PM1, PM2, PM3);
  }
  else if (line>=M5P5_LINE && line<=M5N1_LINE) {
    /*
     * M5 lines
     */
    PM1 = PM1_pure_kissel_cu(Z, E);
    PM2 = PM2_pure_kissel_cu(Z, E, PM1);
    PM3 = PM3_pure_kissel_cu(Z, E, PM1, PM2);
    PM4 = PM4_pure_kissel_cu(Z, E, PM1, PM2, PM3);
    return (FluorYield_cu(Z, M5_SHELL)*RadRate_cu(Z,line))*
		PM5_pure_kissel_cu(Z, E, PM1, PM2, PM3, PM4);

  }
  else {
    return 0.0;
  }  
}

__device__ double CS_FluorLine_Kissel_Radiative_Cascade_cu(int Z, int line, double E) {
  double PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4;

  PK = PL1 = PL2 = PL3 = PM1 = PM2 = PM3 = PM4 = 0.0;


  if (Z<1 || Z>ZMAX) {
    return 0.0;
  }

  if (E <= 0.) {
    return 0.0;
  }

  if (line>=KN5_LINE && line<=KB_LINE) {
    /*
     * K lines -> never cascade effect!
     */
    return CS_Photo_Partial_cu(Z, K_SHELL, E)*FluorYield_cu(Z, K_SHELL)*RadRate_cu(Z,line);
  }
  else if (line>=L1P5_LINE && line<=L1M1_LINE) {
    /*
     * L1 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    return PL1_rad_cascade_kissel_cu(Z, E, PK)*FluorYield_cu(Z, L1_SHELL)*RadRate_cu(Z,line);
  }
  else if (line>=L2Q1_LINE && line<=L2M1_LINE) {
    /*
     * L2 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel_cu(Z,E, PK);
    return (FluorYield_cu(Z, L2_SHELL)*RadRate_cu(Z,line))*
		PL2_rad_cascade_kissel_cu(Z, E, PK, PL1);
  }
  else if (line>=L3Q1_LINE && line<=L3M1_LINE) {
    /*
     * L3 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_rad_cascade_kissel_cu(Z, E, PK, PL1);
    return (FluorYield_cu(Z, L3_SHELL)*RadRate_cu(Z,line))*PL3_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2);
  }
  /*else if (line == LA_LINE) {
    return (CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L3M4_LINE,E)+CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L3M5_LINE,E)); 
  }*/
  /*else if (line == LB_LINE) {
    return (CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L2M4_LINE,E)+
    	CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L2M3_LINE,E)+
        CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L3N5_LINE,E)+
        CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L3O4_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L3O5_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L3O45_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L3N1_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L3O1_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L3N6_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L3N7_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L3N4_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L1M3_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L1M2_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L1M5_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade_cu(Z,L1M4_LINE,E)
    );
  }*/
  else if (line>=M1P5_LINE && line<=M1N1_LINE) {
    /*
     * M1 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_rad_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    return PM1_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3)*FluorYield_cu(Z, M1_SHELL)*RadRate_cu(Z,line);
  }
  else if (line>=M2P5_LINE && line<=M2N1_LINE) {
    /*
     * M2 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_rad_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    PM1 = PM1_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3);
    return (FluorYield_cu(Z, M2_SHELL)*RadRate_cu(Z,line))*
		PM2_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1);
  }
  else if (line>=M3Q1_LINE && line<=M3N1_LINE) {
    /*
     * M3 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_rad_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    PM1 = PM1_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1);
    return (FluorYield_cu(Z, M3_SHELL)*RadRate_cu(Z,line))*
		PM3_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
  }
  else if (line>=M4P5_LINE && line<=M4N1_LINE) {
    /*
     * M4 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_rad_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    PM1 = PM1_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1);
    PM3 = PM3_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    return (FluorYield_cu(Z, M4_SHELL)*RadRate_cu(Z,line))*
		PM4_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
  }
  else if (line>=M5P5_LINE && line<=M5N1_LINE) {
    /*
     * M5 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_rad_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    PM1 = PM1_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1);
    PM3 = PM3_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    PM4 = PM4_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
    return (FluorYield_cu(Z, M5_SHELL)*RadRate_cu(Z,line))*
		PM5_rad_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4);
  }
  else {
    return 0.0;
  }  
}

__device__ double CS_FluorLine_Kissel_Nonradiative_Cascade_cu(int Z, int line, double E) {
  double PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4;

  PK = PL1 = PL2 = PL3 = PM1 = PM2 = PM3 = PM4 = 0.0;


  if (Z<1 || Z>ZMAX) {
    return 0.0;
  }

  if (E <= 0.) {
    return 0.0;
  }

  if (line>=KN5_LINE && line<=KB_LINE) {
    /*
     * K lines -> never cascade effect!
     */
    return CS_Photo_Partial_cu(Z, K_SHELL, E)*FluorYield_cu(Z, K_SHELL)*RadRate_cu(Z,line);
  }
  else if (line>=L1P5_LINE && line<=L1M1_LINE) {
    /*
     * L1 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    return PL1_auger_cascade_kissel_cu(Z, E, PK)*FluorYield_cu(Z, L1_SHELL)*RadRate_cu(Z,line);
  }
  else if (line>=L2Q1_LINE && line<=L2M1_LINE) {
    /*
     * L2 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel_cu(Z,E, PK);
    return (FluorYield_cu(Z, L2_SHELL)*RadRate_cu(Z,line))*
		PL2_auger_cascade_kissel_cu(Z, E, PK, PL1);
  }
  else if (line>=L3Q1_LINE && line<=L3M1_LINE) {
    /*
     * L3 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_auger_cascade_kissel_cu(Z, E, PK, PL1);
    return (FluorYield_cu(Z, L3_SHELL)*RadRate_cu(Z,line))*PL3_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2);
  }
  /*else if (line == LA_LINE) {
    return (CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L3M4_LINE,E)+CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L3M5_LINE,E)); 
  }*/
  /*else if (line == LB_LINE) {
    return (CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L2M4_LINE,E)+
    	CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L2M3_LINE,E)+
        CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L3N5_LINE,E)+
        CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L3O4_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L3O5_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L3O45_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L3N1_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L3O1_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L3N6_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L3N7_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L3N4_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L1M3_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L1M2_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L1M5_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z,L1M4_LINE,E)
    );
  }*/
  else if (line>=M1P5_LINE && line<=M1N1_LINE) {
    /*
     * M1 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_auger_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    return PM1_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3)*FluorYield_cu(Z, M1_SHELL)*RadRate_cu(Z,line);
  }
  else if (line>=M2P5_LINE && line<=M2N1_LINE) {
    /*
     * M2 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_auger_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    PM1 = PM1_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3);
    return (FluorYield_cu(Z, M2_SHELL)*RadRate_cu(Z,line))*
		PM2_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1);
  }
  else if (line>=M3Q1_LINE && line<=M3N1_LINE) {
    /*
     * M3 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_auger_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    PM1 = PM1_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1);
    return (FluorYield_cu(Z, M3_SHELL)*RadRate_cu(Z,line))*
		PM3_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
  }
  else if (line>=M4P5_LINE && line<=M4N1_LINE) {
    /*
     * M4 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_auger_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    PM1 = PM1_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1);
    PM3 = PM3_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    return (FluorYield_cu(Z, M4_SHELL)*RadRate_cu(Z,line))*
		PM4_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
  }
  else if (line>=M5P5_LINE && line<=M5N1_LINE) {
    /*
     * M5 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_auger_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    PM1 = PM1_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1);
    PM3 = PM3_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    PM4 = PM4_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
    return (FluorYield_cu(Z, M5_SHELL)*RadRate_cu(Z,line))*
		PM5_auger_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4);
  }
  else {
    return 0.0;
  }  
}

__device__ double CS_FluorLine_Kissel_Cascade_cu(int Z, int line, double E) {
  double PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4;

  PK = PL1 = PL2 = PL3 = PM1 = PM2 = PM3 = PM4 = 0.0;


  if (Z<1 || Z>ZMAX) {
    return 0.0;
  }

  if (E <= 0.) {
    return 0.0;
  }

  if (line>=KN5_LINE && line<=KB_LINE) {
    /*
     * K lines -> never cascade effect!
     */
    return CS_Photo_Partial_cu(Z, K_SHELL, E)*FluorYield_cu(Z, K_SHELL)*RadRate_cu(Z,line);
  }
  else if (line>=L1P5_LINE && line<=L1M1_LINE) {
    /*
     * L1 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    return PL1_full_cascade_kissel_cu(Z, E, PK)*FluorYield_cu(Z, L1_SHELL)*RadRate_cu(Z,line);
  }
  else if (line>=L2Q1_LINE && line<=L2M1_LINE) {
    /*
     * L2 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel_cu(Z,E, PK);
    return (FluorYield_cu(Z, L2_SHELL)*RadRate_cu(Z,line))*
		PL2_full_cascade_kissel_cu(Z, E, PK, PL1);
  }
  else if (line>=L3Q1_LINE && line<=L3M1_LINE) {
    /*
     * L3 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_full_cascade_kissel_cu(Z, E, PK, PL1);
    return (FluorYield_cu(Z, L3_SHELL)*RadRate_cu(Z,line))*PL3_full_cascade_kissel_cu(Z, E, PK, PL1, PL2);
  }
  /*else if (line == LA_LINE) {
    return (CS_FluorLine_Kissel_Cascade_cu(Z,L3M4_LINE,E)+CS_FluorLine_Kissel_Cascade_cu(Z,L3M5_LINE,E)); 
  }*/
  /*else if (line == LB_LINE) {
    return (CS_FluorLine_Kissel_Cascade_cu(Z,L2M4_LINE,E)+
    	CS_FluorLine_Kissel_Cascade_cu(Z,L2M3_LINE,E)+
        CS_FluorLine_Kissel_Cascade_cu(Z,L3N5_LINE,E)+
        CS_FluorLine_Kissel_Cascade_cu(Z,L3O4_LINE,E)+
	CS_FluorLine_Kissel_Cascade_cu(Z,L3O5_LINE,E)+
	CS_FluorLine_Kissel_Cascade_cu(Z,L3O45_LINE,E)+
	CS_FluorLine_Kissel_Cascade_cu(Z,L3N1_LINE,E)+
	CS_FluorLine_Kissel_Cascade_cu(Z,L3O1_LINE,E)+
	CS_FluorLine_Kissel_Cascade_cu(Z,L3N6_LINE,E)+
	CS_FluorLine_Kissel_Cascade_cu(Z,L3N7_LINE,E)+
	CS_FluorLine_Kissel_Cascade_cu(Z,L3N4_LINE,E)+
	CS_FluorLine_Kissel_Cascade_cu(Z,L1M3_LINE,E)+
	CS_FluorLine_Kissel_Cascade_cu(Z,L1M2_LINE,E)+
	CS_FluorLine_Kissel_Cascade_cu(Z,L1M5_LINE,E)+
	CS_FluorLine_Kissel_Cascade_cu(Z,L1M4_LINE,E)
    );
  }*/
  else if (line>=M1P5_LINE && line<=M1N1_LINE) {
    /*
     * M1 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_full_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_full_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    return PM1_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3)*FluorYield_cu(Z, M1_SHELL)*RadRate_cu(Z,line);
  }
  else if (line>=M2P5_LINE && line<=M2N1_LINE) {
    /*
     * M2 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_full_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_full_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    PM1 = PM1_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3);
    return (FluorYield_cu(Z, M2_SHELL)*RadRate_cu(Z,line))*
		PM2_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1);
  }
  else if (line>=M3Q1_LINE && line<=M3N1_LINE) {
    /*
     * M3 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_full_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_full_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    PM1 = PM1_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1);
    return (FluorYield_cu(Z, M3_SHELL)*RadRate_cu(Z,line))*
		PM3_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
  }
  else if (line>=M4P5_LINE && line<=M4N1_LINE) {
    /*
     * M4 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_full_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_full_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    PM1 = PM1_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1);
    PM3 = PM3_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    return (FluorYield_cu(Z, M4_SHELL)*RadRate_cu(Z,line))*
		PM4_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
  }
  else if (line>=M5P5_LINE && line<=M5N1_LINE) {
    /*
     * M5 lines
     */
    PK = CS_Photo_Partial_cu(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel_cu(Z, E, PK);
    PL2 = PL2_full_cascade_kissel_cu(Z, E, PK, PL1);
    PL3 = PL3_full_cascade_kissel_cu(Z, E, PK, PL1, PL2);
    PM1 = PM1_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1);
    PM3 = PM3_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    PM4 = PM4_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
    return (FluorYield_cu(Z, M5_SHELL)*RadRate_cu(Z,line))*
		PM5_full_cascade_kissel_cu(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4);
  }
  else {
    return 0.0;
  }  
}

__device__ double CSb_FluorLine_Kissel_Cascade_cu(int Z, int line, double E) {
  return CS_FluorLine_Kissel_Cascade_cu(Z, line, E)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double CSb_FluorLine_Kissel_Nonradiative_Cascade_cu(int Z, int line, double E) {
  return CS_FluorLine_Kissel_Nonradiative_Cascade_cu(Z, line, E)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double CSb_FluorLine_Kissel_Radiative_Cascade_cu(int Z, int line, double E) {
  return CS_FluorLine_Kissel_Radiative_Cascade_cu(Z, line, E)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double CSb_FluorLine_Kissel_no_Cascade_cu(int Z, int line, double E) {
  return CS_FluorLine_Kissel_no_Cascade_cu(Z, line, E)*AtomicWeight_arr_d[Z]/AVOGNUM;
}
