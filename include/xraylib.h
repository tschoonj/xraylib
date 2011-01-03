/*
Copyright (c) 2009, 2010, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef _XRAYLIB_H
#define _XRAYLIB_H


#ifdef __cplusplus
extern "C" {
#endif



#define XRAYLIB_MAJOR 2
#define XRAYLIB_MINOR 14


//#define ZMAX 120
#ifndef PI
#define PI  3.14159265359
#endif

//values taken from physics.nist.gov
#define AVOGNUM 0.602214179        // Avogadro number (mol-1 * barn-1 * cm2) 
#define KEV2ANGST 12.39841875   // keV to angstrom-1 conversion factor 
#define MEC2 510.998910         // electron rest mass (keV) 
#define RE2 0.079407877        // square of classical electron radius (barn)

#include "shells.h"
#include "lines.h"
#include "xraylib-parser.h"

//Siegbahn notation
//according to Table VIII.2 from Nomenclature system for X-ray spectroscopy
//Linegroups -> usage is discouraged
#define KA_LINE 0
#define KB_LINE 1
#define LA_LINE 2
#define LB_LINE 3

//single lines
#define KA1_LINE KL3_LINE
#define KA2_LINE KL2_LINE
#define KB1_LINE KM3_LINE
#define KB2_LINE KN3_LINE
#define KB3_LINE KM2_LINE
#define KB4_LINE KN5_LINE
#define KB5_LINE KM5_LINE

#define LA1_LINE L3M5_LINE
#define LA2_LINE L3M4_LINE
#define LB1_LINE L2M4_LINE
#define LB2_LINE L3N5_LINE
#define LB3_LINE L1M3_LINE
#define LB4_LINE L1M2_LINE
#define LB5_LINE L3O45_LINE
#define LB6_LINE L3N1_LINE
#define LB7_LINE L3O1_LINE
#define LB9_LINE L1M5_LINE
#define LB10_LINE L1M4_LINE
#define LB15_LINE L3N4_LINE
#define LB17_LINE L2M3_LINE
#define LG1_LINE L2N4_LINE
#define LG2_LINE L1N2_LINE
#define LG3_LINE L1N3_LINE
#define LG4_LINE L1O3_LINE
#define LG5_LINE L2N1_LINE
#define LG6_LINE L2O4_LINE
#define LG8_LINE L2O1_LINE
#define LE_LINE L2M1_LINE
#define LL_LINE L3M1_LINE
#define LS_LINE L3M3_LINE
#define LT_LINE L3M2_LINE
#define LU_LINE L3N6_LINE
#define LV_LINE L2N6_LINE

#define MA1_LINE M5N7_LINE
#define MA2_LINE M5N6_LINE
#define MB_LINE M4N6_LINE
#define MG_LINE M3N5_LINE

#define F1_TRANS   0    
#define F12_TRANS  1     
#define F13_TRANS  2    
#define FP13_TRANS 3     
#define F23_TRANS  4    

// Initialization
void XRayInit(void);

// Error Handling
void SetHardExit(int hard_exit);
void SetExitStatus(int exit_status);
int GetExitStatus(void);
	
// Atomic weights
float AtomicWeight(int Z);
                                  
// Cross sections (cm2/g)
float CS_Total(int Z, float E);
float CS_Photo(int Z, float E);
float CS_Rayl(int Z, float E);
float CS_Compt(int Z, float E); 
// barn/atom
float CSb_Total(int Z, float E);
float CSb_Photo(int Z, float E);
float CSb_Rayl(int Z, float E);
float CSb_Compt(int Z, float E); 
float CS_KN(float E);

// Unpolarized differential scattering cross sections
float DCS_Thoms(float theta);
float DCS_KN(float E, float theta);
float DCS_Rayl(int Z, float E, float theta);
float DCS_Compt(int Z, float E, float theta);
float DCSb_Rayl(int Z, float E, float theta);
float DCSb_Compt(int Z, float E, float theta);
 
// Polarized differential scattering cross sections
float DCSP_Thoms(float theta, float phi);
float DCSP_KN(float E, float theta, float phi);
float DCSP_Rayl(int Z, float E, float theta, float phi);
float DCSP_Compt(int Z, float E, float theta, float phi);
float DCSPb_Rayl(int Z, float E, float theta, float phi);
float DCSPb_Compt(int Z, float E, float theta, float phi);
 
// Scattering factors
float  FF_Rayl(int Z, float q);
float  SF_Compt(int Z, float q);
float  MomentTransf(float E, float theta);

// X-ray fluorescent line energy
float LineEnergy(int Z, int line);

// Fluorescence yield 
float  FluorYield(int Z, int shell);

// Coster-Kronig transition Probability
float  CosKronTransProb(int Z, int trans);

// Absorption-edge energies     
float EdgeEnergy(int Z, int shell);

// Jump ratio
float  JumpFactor(int Z, int shell);

// Fluorescent-lines cross sections
float CS_FluorLine(int Z, int line, float E);
float CSb_FluorLine(int Z, int line, float E);

// Fractional radiative rate
float  RadRate(int Z, int line);

// Photon energy after Compton scattering
float ComptonEnergy(float E0, float theta);

// Anomalous Scattering Factors
float Fi(int Z, float E);
float Fii(int Z, float E);

// Kissel Photoelectric cross sections
float CS_Photo_Total(int Z, float E);
float CSb_Photo_Total(int Z, float E);
float CS_Photo_Partial(int Z, int shell, float E);
float CSb_Photo_Partial(int Z, int shell, float E);

// XRF cross sections using Kissel partial photoelectric cross sections
float CS_FluorLine_Kissel(int Z, int line, float E); 
float CSb_FluorLine_Kissel(int Z, int line, float E); 

// Total cross sections (photoionization+Rayleigh+Compton) using Kissel Total photoelectric cross sections
float CS_Total_Kissel(int Z, float E); 
float CSb_Total_Kissel(int Z, float E); 

//Electron configuration (according to Kissel)
float ElectronConfig(int Z, int shell);


//Cross Section functions using the compound parser
float CS_Total_CP(const char compound[], float E);
float CS_Photo_CP(const char compound[], float E);
float CS_Rayl_CP(const char compound[], float E);
float CS_Compt_CP(const char compound[], float E); 
float CSb_Total_CP(const char compound[], float E);
float CSb_Photo_CP(const char compound[], float E);
float CSb_Rayl_CP(const char compound[], float E);
float CSb_Compt_CP(const char compound[], float E); 
float DCS_Rayl_CP(const char compound[], float E, float theta);
float DCS_Compt_CP(const char compound[], float E, float theta);
float DCSb_Rayl_CP(const char compound[], float E, float theta);
float DCSb_Compt_CP(const char compound[], float E, float theta);
float DCSP_Rayl_CP(const char compound[], float E, float theta, float phi);
float DCSP_Compt_CP(const char compound[], float E, float theta, float phi);
float DCSPb_Rayl_CP(const char compound[], float E, float theta, float phi);
float DCSPb_Compt_CP(const char compound[], float E, float theta, float phi);
float CS_Photo_Total_CP(const char compound[], float E);
float CSb_Photo_Total_CP(const char compound[], float E);
float CS_Total_Kissel_CP(const char compound[], float E); 
float CSb_Total_Kissel_CP(const char compound[], float E); 

//Refractive indices functions
float Refractive_Index_Re(const char compound[], float E, float density);
float Refractive_Index_Im(const char compound[], float E, float density);

//ComptonProfiles
float ComptonProfile(int Z, float pz);
float ComptonProfile_Partial(int Z, int shell, float pz);

//Atomic level widths
float AtomicLevelWidth(int Z, int shell);


#ifdef __cplusplus
}
#endif

#endif






