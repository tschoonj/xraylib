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

extern struct MendelElement MendelArray[MENDEL_MAX];
extern struct MendelElement MendelArraySorted[MENDEL_MAX];

extern Crystal_Array Crystal_arr;

extern float AtomicWeight_arr[ZMAX+1];
extern float EdgeEnergy_arr[ZMAX+1][SHELLNUM];
extern float LineEnergy_arr[ZMAX+1][LINENUM];
extern float FluorYield_arr[ZMAX+1][SHELLNUM];
extern float JumpFactor_arr[ZMAX+1][SHELLNUM];
extern float CosKron_arr[ZMAX+1][TRANSNUM];
extern float RadRate_arr[ZMAX+1][LINENUM];
extern float AtomicLevelWidth_arr[ZMAX+1][SHELLNUM];

extern int NE_Photo[ZMAX+1];
extern float *E_Photo_arr[ZMAX+1];
extern float *CS_Photo_arr[ZMAX+1];
extern float *CS_Photo_arr2[ZMAX+1];

extern int NE_Rayl[ZMAX+1];
extern float *E_Rayl_arr[ZMAX+1];
extern float *CS_Rayl_arr[ZMAX+1];
extern float *CS_Rayl_arr2[ZMAX+1];

extern int NE_Compt[ZMAX+1];
extern float *E_Compt_arr[ZMAX+1];
extern float *CS_Compt_arr[ZMAX+1];
extern float *CS_Compt_arr2[ZMAX+1];

extern int Nq_Rayl[ZMAX+1];
extern float *q_Rayl_arr[ZMAX+1];
extern float *FF_Rayl_arr[ZMAX+1];
extern float *FF_Rayl_arr2[ZMAX+1];

extern int Nq_Compt[ZMAX+1];
extern float *q_Compt_arr[ZMAX+1];
extern float *SF_Compt_arr[ZMAX+1];
extern float *SF_Compt_arr2[ZMAX+1];

extern int NE_Fi[ZMAX+1];
extern float *E_Fi_arr[ZMAX+1];
extern float *Fi_arr[ZMAX+1];
extern float *Fi_arr2[ZMAX+1];

extern int NE_Fii[ZMAX+1];
extern float *E_Fii_arr[ZMAX+1];
extern float *Fii_arr[ZMAX+1];
extern float *Fii_arr2[ZMAX+1];

extern int NE_Photo_Total_Kissel[ZMAX+1];
extern double *E_Photo_Total_Kissel[ZMAX+1];
extern double *Photo_Total_Kissel[ZMAX+1];
extern double *Photo_Total_Kissel2[ZMAX+1];

extern float Electron_Config_Kissel[ZMAX+1][SHELLNUM_K];
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

extern double Auger_Transition_Total[ZMAX+1][SHELLNUM_A];
extern double Auger_Transition_Individual[ZMAX+1][AUGERNUM];

#endif


















