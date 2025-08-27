/*
Copyright (c) 2009-2021, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef XRAYLIB_H
#define XRAYLIB_H


#ifdef __cplusplus
extern "C" {
#endif



#define XRAYLIB_MAJOR 4
#define XRAYLIB_MINOR 2
#define XRAYLIB_MICRO 0


#ifndef PI
#define PI  3.1415926535897932384626433832795
#endif

#ifndef TWOPI
#define TWOPI     (2 * PI)
#endif

#define RADEG     ( 180.0 / PI )
#define DEGRAD    ( PI / 180.0 )

#ifndef XRL_EXTERN
#define XRL_EXTERN extern
#endif

/*
 *
 * values taken from physics.nist.gov
 *
 */
#define AVOGNUM 0.602214129        /* Avogadro number (mol-1 * barn-1 * cm2) */
#define KEV2ANGST 12.39841930      /* keV to angstrom-1 conversion factor */
#define MEC2 510.998928            /* electron rest mass (keV) */
#define RE2 0.079407877            /* square of classical electron radius (barn) */
#define R_E 2.8179403267e-15       /* Classical electron radius (m) */

#include "xraylib-error.h"
#include "xraylib-shells.h"
#include "xraylib-lines.h"
#include "xraylib-parser.h"
#include "xraylib-auger.h"
#include "xraylib-defs.h"
#include "xraylib-crystal-diffraction.h"
#include "xraylib-nist-compounds.h"
#include "xraylib-radionuclides.h"
#include "xraylib-deprecated.h"
#include "xraylib-aux.h"

/*
 * Siegbahn notation
 * according to Table VIII.2 from Nomenclature system for X-ray spectroscopy
 * Linegroups -> usage is discouraged
 *
 */
#define KA_LINE 0
#define KB_LINE 1
#define LA_LINE 2
#define LB_LINE 3

/* single lines */
#define KA1_LINE KL3_LINE
#define KA2_LINE KL2_LINE
#define KA3_LINE KL1_LINE
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
#define LH_LINE L2M1_LINE
#define LL_LINE L3M1_LINE
#define LS_LINE L3M3_LINE
#define LT_LINE L3M2_LINE
#define LU_LINE L3N6_LINE
#define LV_LINE L2N6_LINE

#define MA1_LINE M5N7_LINE
#define MA2_LINE M5N6_LINE
#define MB_LINE M4N6_LINE
#define MG_LINE M3N5_LINE

#define FL12_TRANS 1
#define FL13_TRANS 2
#define FLP13_TRANS 3
#define FL23_TRANS 4
#define FM12_TRANS 5
#define FM13_TRANS 6
#define FM14_TRANS 7
#define FM15_TRANS 8
#define FM23_TRANS 9
#define FM24_TRANS 10
#define FM25_TRANS 11
#define FM34_TRANS 12
#define FM35_TRANS 13
#define FM45_TRANS 14


/* Initialization */
XRL_EXTERN
void XRayInit(void);

/* Atomic weights */
XRL_EXTERN
double AtomicWeight(int Z, xrl_error **error);

/* Density of pure atomic element */
XRL_EXTERN
double ElementDensity(int Z, xrl_error **error);

/* Cross sections (cm2/g) */
XRL_EXTERN
double CS_Total(int Z, double E, xrl_error **error);
XRL_EXTERN
double CS_Photo(int Z, double E, xrl_error **error);
XRL_EXTERN
double CS_Rayl(int Z, double E, xrl_error **error);
XRL_EXTERN
double CS_Compt(int Z, double E, xrl_error **error);
XRL_EXTERN
double CS_KN(double E, xrl_error **error);
XRL_EXTERN
double CS_Energy(int Z, double E, xrl_error **error);

/* barn/atom */
XRL_EXTERN
double CSb_Total(int Z, double E, xrl_error **error);
XRL_EXTERN
double CSb_Photo(int Z, double E, xrl_error **error);
XRL_EXTERN
double CSb_Rayl(int Z, double E, xrl_error **error);
XRL_EXTERN
double CSb_Compt(int Z, double E, xrl_error **error);


/* Unpolarized differential scattering cross sections */
XRL_EXTERN
double DCS_Thoms(double theta, xrl_error **error);
XRL_EXTERN
double DCS_KN(double E, double theta, xrl_error **error);
XRL_EXTERN
double DCS_Rayl(int Z, double E, double theta, xrl_error **error);
XRL_EXTERN
double DCS_Compt(int Z, double E, double theta, xrl_error **error);
XRL_EXTERN
double DCSb_Rayl(int Z, double E, double theta, xrl_error **error);
XRL_EXTERN
double DCSb_Compt(int Z, double E, double theta, xrl_error **error);

/* Polarized differential scattering cross sections */
XRL_EXTERN
double DCSP_Thoms(double theta, double phi, xrl_error **error);
XRL_EXTERN
double DCSP_KN(double E, double theta, double phi, xrl_error **error);
XRL_EXTERN
double DCSP_Rayl(int Z, double E, double theta, double phi, xrl_error **error);
XRL_EXTERN
double DCSP_Compt(int Z, double E, double theta, double phi, xrl_error **error);
XRL_EXTERN
double DCSPb_Rayl(int Z, double E, double theta, double phi, xrl_error **error);
XRL_EXTERN
double DCSPb_Compt(int Z, double E, double theta, double phi, xrl_error **error);

/* Scattering factors */
XRL_EXTERN
double FF_Rayl(int Z, double q, xrl_error **error);
XRL_EXTERN
double SF_Compt(int Z, double q, xrl_error **error);
XRL_EXTERN
double MomentTransf(double E, double theta, xrl_error **error);

/* X-ray fluorescent line energy */
XRL_EXTERN
double LineEnergy(int Z, int line, xrl_error **error);

/* Fluorescence yield */
XRL_EXTERN
double FluorYield(int Z, int shell, xrl_error **error);

/* Coster-Kronig transition Probability */
XRL_EXTERN
double CosKronTransProb(int Z, int trans, xrl_error **error);

/* Absorption-edge energies */
XRL_EXTERN
double EdgeEnergy(int Z, int shell, xrl_error **error);

/* Jump ratio */
XRL_EXTERN
double JumpFactor(int Z, int shell, xrl_error **error);

/* Fluorescent-lines cross sections */
XRL_EXTERN
double CS_FluorLine(int Z, int line, double E, xrl_error **error);
XRL_EXTERN
double CSb_FluorLine(int Z, int line, double E, xrl_error **error);

/* Fluorescence cross sections for an entire shell */
XRL_EXTERN
double CS_FluorShell(int Z, int shell, double E, xrl_error **error);
XRL_EXTERN
double CSb_FluorShell(int Z, int shell, double E, xrl_error **error);

/* Fractional radiative rate */
XRL_EXTERN
double RadRate(int Z, int line, xrl_error **error);

/* Photon energy after Compton scattering */
XRL_EXTERN
double ComptonEnergy(double E0, double theta, xrl_error **error);

/* Anomalous Scattering Factors */
XRL_EXTERN
double Fi(int Z, double E, xrl_error **error);
XRL_EXTERN
double Fii(int Z, double E, xrl_error **error);

/* Kissel Photoelectric cross sections */
XRL_EXTERN
double CS_Photo_Total(int Z, double E, xrl_error **error);
XRL_EXTERN
double CSb_Photo_Total(int Z, double E, xrl_error **error);
XRL_EXTERN
double CS_Photo_Partial(int Z, int shell, double E, xrl_error **error);
XRL_EXTERN
double CSb_Photo_Partial(int Z, int shell, double E, xrl_error **error);

/* XRF cross sections using Kissel partial photoelectric cross sections */
XRL_EXTERN
double CS_FluorLine_Kissel(int Z, int line, double E, xrl_error **error);
XRL_EXTERN
double CSb_FluorLine_Kissel(int Z, int line, double E, xrl_error **error);
XRL_EXTERN
double CS_FluorLine_Kissel_Cascade(int Z, int line, double E, xrl_error **error);
XRL_EXTERN
double CSb_FluorLine_Kissel_Cascade(int Z, int line, double E, xrl_error **error);
XRL_EXTERN
double CS_FluorLine_Kissel_Nonradiative_Cascade(int Z, int line, double E, xrl_error **error);
XRL_EXTERN
double CSb_FluorLine_Kissel_Nonradiative_Cascade(int Z, int line, double E, xrl_error **error);
XRL_EXTERN
double CS_FluorLine_Kissel_Radiative_Cascade(int Z, int line, double E, xrl_error **error);
XRL_EXTERN
double CSb_FluorLine_Kissel_Radiative_Cascade(int Z, int line, double E, xrl_error **error);
XRL_EXTERN
double CS_FluorLine_Kissel_no_Cascade(int Z, int line, double E, xrl_error **error);
XRL_EXTERN
double CSb_FluorLine_Kissel_no_Cascade(int Z, int line, double E, xrl_error **error);

XRL_EXTERN
double CS_FluorShell_Kissel(int Z, int shell, double E, xrl_error **error);
XRL_EXTERN
double CSb_FluorShell_Kissel(int Z, int shell, double E, xrl_error **error);
XRL_EXTERN
double CS_FluorShell_Kissel_Cascade(int Z, int shell, double E, xrl_error **error);
XRL_EXTERN
double CSb_FluorShell_Kissel_Cascade(int Z, int shell, double E, xrl_error **error);
XRL_EXTERN
double CS_FluorShell_Kissel_Nonradiative_Cascade(int Z, int shell, double E, xrl_error **error);
XRL_EXTERN
double CSb_FluorShell_Kissel_Nonradiative_Cascade(int Z, int shell, double E, xrl_error **error);
XRL_EXTERN
double CS_FluorShell_Kissel_Radiative_Cascade(int Z, int shell, double E, xrl_error **error);
XRL_EXTERN
double CSb_FluorShell_Kissel_Radiative_Cascade(int Z, int shell, double E, xrl_error **error);
XRL_EXTERN
double CS_FluorShell_Kissel_no_Cascade(int Z, int shell, double E, xrl_error **error);
XRL_EXTERN
double CSb_FluorShell_Kissel_no_Cascade(int Z, int shell, double E, xrl_error **error);


/* Total cross sections (photoionization+Rayleigh+Compton) using Kissel Total photoelectric cross sections */
XRL_EXTERN
double CS_Total_Kissel(int Z, double E, xrl_error **error);
XRL_EXTERN
double CSb_Total_Kissel(int Z, double E, xrl_error **error);

/* Electron configuration (according to Kissel) */
XRL_EXTERN
double ElectronConfig(int Z, int shell, xrl_error **error);


/* Cross Section functions using the compound parser */
XRL_EXTERN
double CS_Total_CP(const char compound[], double E, xrl_error **error);
XRL_EXTERN
double CS_Photo_CP(const char compound[], double E, xrl_error **error);
XRL_EXTERN
double CS_Rayl_CP(const char compound[], double E, xrl_error **error);
XRL_EXTERN
double CS_Compt_CP(const char compound[], double E, xrl_error **error);
XRL_EXTERN
double CSb_Total_CP(const char compound[], double E, xrl_error **error);
XRL_EXTERN
double CSb_Photo_CP(const char compound[], double E, xrl_error **error);
XRL_EXTERN
double CSb_Rayl_CP(const char compound[], double E, xrl_error **error);
XRL_EXTERN
double CSb_Compt_CP(const char compound[], double E, xrl_error **error);
XRL_EXTERN
double DCS_Rayl_CP(const char compound[], double E, double theta, xrl_error **error);
XRL_EXTERN
double DCS_Compt_CP(const char compound[], double E, double theta, xrl_error **error);
XRL_EXTERN
double DCSb_Rayl_CP(const char compound[], double E, double theta, xrl_error **error);
XRL_EXTERN
double DCSb_Compt_CP(const char compound[], double E, double theta, xrl_error **error);
XRL_EXTERN
double DCSP_Rayl_CP(const char compound[], double E, double theta, double phi, xrl_error **error);
XRL_EXTERN
double DCSP_Compt_CP(const char compound[], double E, double theta, double phi, xrl_error **error);
XRL_EXTERN
double DCSPb_Rayl_CP(const char compound[], double E, double theta, double phi, xrl_error **error);
XRL_EXTERN
double DCSPb_Compt_CP(const char compound[], double E, double theta, double phi, xrl_error **error);
XRL_EXTERN
double CS_Photo_Total_CP(const char compound[], double E, xrl_error **error);
XRL_EXTERN
double CSb_Photo_Total_CP(const char compound[], double E, xrl_error **error);
XRL_EXTERN
double CS_Total_Kissel_CP(const char compound[], double E, xrl_error **error);
XRL_EXTERN
double CSb_Total_Kissel_CP(const char compound[], double E, xrl_error **error);
XRL_EXTERN
double CS_Energy_CP(const char compound[], double E, xrl_error **error);

/* Refractive indices functions */
XRL_EXTERN
double Refractive_Index_Re(const char compound[], double E, double density, xrl_error **error);
XRL_EXTERN
double Refractive_Index_Im(const char compound[], double E, double density, xrl_error **error);
XRL_EXTERN
xrlComplex Refractive_Index(const char compound[], double E, double density, xrl_error **error);

/* ComptonProfiles */
XRL_EXTERN
double ComptonProfile(int Z, double pz, xrl_error **error);
XRL_EXTERN
double ComptonProfile_Partial(int Z, int shell, double pz, xrl_error **error);

/* Atomic level widths */
XRL_EXTERN
double AtomicLevelWidth(int Z, int shell, xrl_error **error);


/* Auger non-radiative rates */
XRL_EXTERN
double AugerRate(int Z, int auger_trans, xrl_error **error);

/* Auger yield */
XRL_EXTERN
double AugerYield(int Z, int shell, xrl_error **error);

#ifdef __cplusplus
}
#endif

#endif
