#ifndef _XRAYLIB_H
#define _XRAYLIB_H

#define XRAYLIB_MAJOR 2
#define XRAYLIB_MINOR 11


//#define ZMAX 120
#ifndef PI
#define PI  3.14159265359
#endif

#define AVOGNUM 0.602252        // Avogadro number (mol-1 * barn-1 * cm2) 
#define KEV2ANGST 12.398520   // keV to angstrom-1 conversion factor 
#define MEC2 511.0034         // electron rest mass (keV) 
#define RE2 0.07940775        // square of classical electron radius (barn)

#include "shells.h"
#include "lines.h"

#define KA_LINE 0
#define KB_LINE 1
#define LA_LINE 2
#define LB_LINE 3

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

#endif






