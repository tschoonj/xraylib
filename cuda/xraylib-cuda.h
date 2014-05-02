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

#ifndef XRAYLIB_CUDA_H
#define XRAYLIB_CUDA_H

#include <stdio.h>



#ifdef __cplusplus
extern "C" {
#endif




//host functions

/*
 *
 * Initializes cuda xraylib
 * Copies all the relevant datasets to the GPU device memory
 *
 */
int CudaXRayInit();


/*
 * device functions
 */
__device__ double AtomicLevelWidth_cu(int Z, int shell);
__device__ double AtomicWeight_cu(int Z);
__device__ double AugerRate_cu(int Z, int auger_trans);
__device__ double AugerYield_cu(int Z, int shell);
__device__ double ComptonProfile_cu(int Z, double pz);
__device__ double ComptonProfile_Partial_cu(int Z, int shell, double pz);
__device__ double CosKronTransProb_cu(int Z, int trans);
__device__ double CS_Total_cu(int Z, double E);
__device__ double CS_Photo_cu(int Z, double E);
__device__ double CS_Rayl_cu(int Z, double E);
__device__ double CS_Compt_cu(int Z, double E);
__device__ double CS_Energy_cu(int Z, double E);
__device__ double ElementDensity_cu(int Z);
__device__ double EdgeEnergy_cu(int Z, int shell);
__device__ double Fi_cu(int Z, double E);
__device__ double Fii_cu(int Z, double E);
__device__ double FluorYield_cu(int Z, int shell);
__device__ double JumpFactor_cu(int Z, int shell);
__device__ double RadRate_cu(int Z, int line);
__device__ double FF_Rayl_cu(int Z, double q);
__device__ double SF_Compt_cu(int Z, double q);
__device__ double DCS_Thoms_cu(double theta);
__device__ double DCS_KN_cu(double E, double theta); 
__device__ double DCS_Rayl_cu(int Z, double E, double theta);
__device__ double DCS_Compt_cu(int Z, double E, double theta);
__device__ double MomentTransf_cu(double E, double theta);
__device__ double CS_KN_cu(double E);
__device__ double ComptonEnergy_cu(double E0, double theta);
__device__ double CSb_Photo_Total_cu(int Z, double E);
__device__ double CS_Photo_Total_cu(int Z, double E);
__device__ double CSb_Photo_Partial_cu(int Z, int shell, double E);
__device__ double CS_Photo_Partial_cu(int Z, int shell, double E);
__device__ double CS_FluorLine_Kissel_cu(int Z, int line, double E);
__device__ double CSb_FluorLine_Kissel_cu(int Z, int line, double E);
__device__ double CS_Total_Kissel_cu(int Z, double E);
__device__ double CSb_Total_Kissel_cu(int Z, double E);
__device__ double ElectronConfig_cu(int Z, int shell);
__device__ double CS_FluorLine_Kissel_no_Cascade_cu(int Z, int line, double E);
__device__ double CS_FluorLine_Kissel_Radiative_Cascade_cu(int Z, int line, double E);
__device__ double CS_FluorLine_Kissel_Nonradiative_Cascade_cu(int Z, int line, double E);
__device__ double CS_FluorLine_Kissel_Cascade_cu(int Z, int line, double E);
__device__ double CSb_FluorLine_Kissel_Cascade_cu(int Z, int line, double E);
__device__ double CSb_FluorLine_Kissel_Nonradiative_Cascade_cu(int Z, int line, double E);
__device__ double CSb_FluorLine_Kissel_Radiative_Cascade_cu(int Z, int line, double E);
__device__ double CSb_FluorLine_Kissel_no_Cascade_cu(int Z, int line, double E);
__device__ double CS_FluorLine_cu(int Z, int line, double E);
__device__ double LineEnergy_cu(int Z, int line);
__device__ double splint_cu(double *xa, double *ya, double *y2a, int n, double x);
__device__ double DCSP_Rayl_cu(int Z, double E, double theta, double phi);
__device__ double DCSP_Compt_cu(int Z, double E, double theta, double phi);
__device__ double DCSP_KN_cu(double E, double theta, double phi);
__device__ double DCSP_Thoms_cu(double theta, double phi);
__device__ double CSb_Total_cu(int Z, double E);
__device__ double CSb_Photo_cu(int Z, double E);
__device__ double CSb_Rayl_cu(int Z, double E);
__device__ double CSb_Compt_cu(int Z, double E);
__device__ double CSb_FluorLine_cu(int Z, int line, double E);
__device__ double DCSb_Rayl_cu(int Z, double E, double theta);
__device__ double DCSb_Compt_cu(int Z, double E, double theta);
__device__ double DCSPb_Rayl_cu(int Z, double E, double theta, double phi);
__device__ double DCSPb_Compt_cu(int Z, double E, double theta, double phi);


__device__ double PL1_pure_kissel_cu(int Z, double E);
__device__ double PL1_rad_cascade_kissel_cu(int Z, double E, double PK);
__device__ double PL1_auger_cascade_kissel_cu(int Z, double E, double PK);
__device__ double PL1_full_cascade_kissel_cu(int Z, double E, double PK);
__device__ double PL2_pure_kissel_cu(int Z, double E, double PL1);
__device__ double PL2_rad_cascade_kissel_cu(int Z, double E, double PK, double PL1);
__device__ double PL2_auger_cascade_kissel_cu(int Z, double E, double PK, double PL1);
__device__ double PL2_full_cascade_kissel_cu(int Z, double E, double PK, double PL1);
__device__ double PL3_pure_kissel_cu(int Z, double E, double PL1, double PL2);
__device__ double PL3_rad_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2);
__device__ double PL3_auger_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2);
__device__ double PL3_full_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2);
__device__ double PM1_pure_kissel_cu(int Z, double E);
__device__ double PM1_rad_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3);
__device__ double PM1_auger_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3);
__device__ double PM1_full_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3);
__device__ double PM2_pure_kissel_cu(int Z, double E, double PM1);
__device__ double PM2_rad_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1);
__device__ double PM2_auger_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1);
__device__ double PM2_full_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1);
__device__ double PM3_pure_kissel_cu(int Z, double E, double PM1, double PM2);
__device__ double PM3_rad_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2);
__device__ double PM3_auger_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2);
__device__ double PM3_full_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2);
__device__ double PM4_pure_kissel_cu(int Z, double E, double PM1, double PM2, double PM3);
__device__ double PM4_rad_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3);
__device__ double PM4_auger_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3);
__device__ double PM4_full_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3);
__device__ double PM5_pure_kissel_cu(int Z, double E, double PM1, double PM2, double PM3, double PM4);
__device__ double PM5_rad_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4);
__device__ double PM5_auger_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4);
__device__ double PM5_full_cascade_kissel_cu(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4);








#ifdef __cplusplus
}
#endif
#endif
