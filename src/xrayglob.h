/*
Copyright (c) 2009, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/



#ifndef GLOBH
#define GLOBH

#include "xrayvars.h"
#include "xraylib-shells.h"

/* Struct to hold info on a particular type of atom */

struct MendelElement {
  int Zatom;              /* Atomic number of atom. */
  char *name;             /* Name of atom. */
};

extern struct MendelElement MendelArray[MENDEL_MAX];
extern struct MendelElement MendelArraySorted[MENDEL_MAX];

extern Crystal_Array Crystal_arr;

extern double AtomicWeight_arr[ZMAX+1];
extern double EdgeEnergy_arr[ZMAX+1][SHELLNUM];
extern double LineEnergy_arr[ZMAX+1][LINENUM];
extern double FluorYield_arr[ZMAX+1][SHELLNUM];
extern double JumpFactor_arr[ZMAX+1][SHELLNUM];
extern double CosKron_arr[ZMAX+1][TRANSNUM];
extern double RadRate_arr[ZMAX+1][LINENUM];
extern double AtomicLevelWidth_arr[ZMAX+1][SHELLNUM];

extern int NE_Photo[ZMAX+1];
extern double *E_Photo_arr[ZMAX+1];
extern double *CS_Photo_arr[ZMAX+1];
extern double *CS_Photo_arr2[ZMAX+1];

extern int NE_Rayl[ZMAX+1];
extern double *E_Rayl_arr[ZMAX+1];
extern double *CS_Rayl_arr[ZMAX+1];
extern double *CS_Rayl_arr2[ZMAX+1];

extern int NE_Compt[ZMAX+1];
extern double *E_Compt_arr[ZMAX+1];
extern double *CS_Compt_arr[ZMAX+1];
extern double *CS_Compt_arr2[ZMAX+1];

extern int Nq_Rayl[ZMAX+1];
extern double *q_Rayl_arr[ZMAX+1];
extern double *FF_Rayl_arr[ZMAX+1];
extern double *FF_Rayl_arr2[ZMAX+1];

extern int Nq_Compt[ZMAX+1];
extern double *q_Compt_arr[ZMAX+1];
extern double *SF_Compt_arr[ZMAX+1];
extern double *SF_Compt_arr2[ZMAX+1];

extern int NE_Energy[ZMAX+1];
extern double *E_Energy_arr[ZMAX+1];
extern double *CS_Energy_arr[ZMAX+1];
extern double *CS_Energy_arr2[ZMAX+1];

extern int NE_Fi[ZMAX+1];
extern double *E_Fi_arr[ZMAX+1];
extern double *Fi_arr[ZMAX+1];
extern double *Fi_arr2[ZMAX+1];

extern int NE_Fii[ZMAX+1];
extern double *E_Fii_arr[ZMAX+1];
extern double *Fii_arr[ZMAX+1];
extern double *Fii_arr2[ZMAX+1];

extern int NE_Photo_Total_Kissel[ZMAX+1];
extern double *E_Photo_Total_Kissel[ZMAX+1];
extern double *Photo_Total_Kissel[ZMAX+1];
extern double *Photo_Total_Kissel2[ZMAX+1];

extern double Electron_Config_Kissel[ZMAX+1][SHELLNUM_K];
extern double EdgeEnergy_Kissel[ZMAX+1][SHELLNUM_K];

extern int NE_Photo_Partial_Kissel[ZMAX+1][SHELLNUM_K];
extern double *E_Photo_Partial_Kissel[ZMAX+1][SHELLNUM_K];
extern double *Photo_Partial_Kissel[ZMAX+1][SHELLNUM_K];
extern double *Photo_Partial_Kissel2[ZMAX+1][SHELLNUM_K];

extern int NShells_ComptonProfiles[ZMAX+1];
extern int Npz_ComptonProfiles[ZMAX+1];
extern double *UOCCUP_ComptonProfiles[ZMAX+1];
extern double *pz_ComptonProfiles[ZMAX+1];
extern double *Total_ComptonProfiles[ZMAX+1];
extern double *Total_ComptonProfiles2[ZMAX+1];
extern double *Partial_ComptonProfiles[ZMAX+1][SHELLNUM_C];
extern double *Partial_ComptonProfiles2[ZMAX+1][SHELLNUM_C];

extern double Auger_Rates[ZMAX+1][AUGERNUM];
extern double Auger_Yields[ZMAX+1][SHELLNUM_A];

extern double ElementDensity_arr[ZMAX+1];

extern double xrf_cross_sections_constants_full[ZMAX+1][M5_SHELL+1][L3_SHELL+1];
extern double xrf_cross_sections_constants_auger_only[ZMAX+1][M5_SHELL+1][L3_SHELL+1];
#endif
