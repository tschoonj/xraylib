/*
Copyright (c) 2009, 2010, 2011, Teemu Ikonen and Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Teemu Ikonen and Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Teemu Ikonen and Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <stdio.h>
#include <string.h>
#include "xraylib.h"
#include "xrayglob.h"
#include "xraylib-nist-compounds-internal.h"
#include "xraylib-radionuclides-internal.h"
#include "xrf_cross_sections_aux.h"
#include "xrf_cross_sections_aux-private.h"


extern double Auger_Transition_Total[ZMAX+1][SHELLNUM_A];
extern double Auger_Transition_Individual[ZMAX+1][AUGERNUM];

#define OUTFILE "xraylib.dat"
#define FLOAT_PER_LINE 4
#define INT_PER_LINE 10
#define NAME_PER_LINE 4

void XRayInit(void);
void XRayInitFromPath(char *path);
FILE *f;

#define PR_MATD(ARRNAME) \
	fwrite(ARRNAME, sizeof(double), sizeof(ARRNAME) / sizeof(double), f);

#define PR_MATI(ARRNAME) \
	fwrite(ARRNAME, sizeof(int), sizeof(ARRNAME) / sizeof(int), f);

#define PR_DYNMATD(NVAR, EVAR, ENAME) \
  for(j = 0; j < ZMAX+1; j++) { \
    if(NVAR[j] > 0) {\
      print_doublevec(NVAR[j], EVAR[j]); \
    }\
  }

#define PR_DYNMAT_3DD_K(NVAR2D, EVAR, ENAME) \
  for (i = 0; i < ZMAX+1; i++) { \
    for (j = 0; j < SHELLNUM_K; j++) {\
      if(NVAR2D[i][j] > 0) {\
        print_doublevec(NVAR2D[i][j], EVAR[i][j]);\
      }\
    }\
  }

#define PR_DYNMAT_3DD_C(NVAR2D, NVAR2D2, NVAR2D3, EVAR, ENAME) \
  for (i = 0; i < ZMAX+1 ; i++) { \
    for (j = 0; j < NShells_ComptonProfiles[i]; j++) {\
      if (UOCCUP_ComptonProfiles[i][j] > 0.0) {\
        if(NVAR2D[i] > 0) {\
         print_doublevec(NVAR2D[i], EVAR[i][j]);\
        }\
      }\
    }\
  }

#define PR_NUMVEC1D(NVAR, NNAME) \
  print_intvec(ZMAX+1, NVAR);

static double AugerYield_prdata(int Z, int shell) {

	double rv;

	rv = 0.0;

	if (Z > ZMAX || Z < 1) {
		return rv;
	}
	else if (shell < K_SHELL || shell > M5_SHELL) {
		return rv;
	}

	rv = FluorYield(Z, shell, NULL);

	if (rv == 0.0)
		return 0.0;

	rv = 1.0 - rv;

	if (shell == L1_SHELL) {
		rv -= CosKronTransProb(Z, FL12_TRANS, NULL);
		rv -= CosKronTransProb(Z, FL13_TRANS, NULL);
		rv -= CosKronTransProb(Z, FLP13_TRANS, NULL);
	}
	else if (shell == L2_SHELL) {
		rv -= CosKronTransProb(Z, FL23_TRANS, NULL);
	}
	else if (shell == M1_SHELL) {
		rv -= CosKronTransProb(Z, FM12_TRANS, NULL);
		rv -= CosKronTransProb(Z, FM13_TRANS, NULL);
		rv -= CosKronTransProb(Z, FM14_TRANS, NULL);
		rv -= CosKronTransProb(Z, FM15_TRANS, NULL);
	}
	else if (shell == M2_SHELL) {
		rv -= CosKronTransProb(Z, FM23_TRANS, NULL);
		rv -= CosKronTransProb(Z, FM24_TRANS, NULL);
		rv -= CosKronTransProb(Z, FM25_TRANS, NULL);
	}
	else if (shell == M3_SHELL) {
		rv -= CosKronTransProb(Z, FM34_TRANS, NULL);
		rv -= CosKronTransProb(Z, FM35_TRANS, NULL);
	}
	else if (shell == M4_SHELL) {
		rv -= CosKronTransProb(Z, FM45_TRANS, NULL);
	}

	return rv;
}

static double AugerYield2_prdata(int Z, int shell) {
	double rv;

	rv = 0.0;

	if (Z > ZMAX || Z < 1) {
		return rv;
	}
	else if (shell < K_SHELL || shell > M5_SHELL) {
		return rv;
	}

	rv = Auger_Transition_Total[Z][shell];
	if (shell == L1_SHELL) {
		rv -= Auger_Transition_Individual[Z][L1_L2L2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2L3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2M1_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2M2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2M3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2M4_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2M5_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2N1_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2N2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2N3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2N4_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2N5_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2N6_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2N7_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2O1_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2O2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2O3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2O4_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2O5_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2O6_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2O7_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2P1_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2P2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2P3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2P4_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2P5_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L2Q3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3L2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3L3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3M1_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3M2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3M3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3M4_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3M5_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3N1_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3N2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3N3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3N4_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3N5_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3N6_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3N7_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3O1_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3O2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3O3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3O4_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3O5_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3O6_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3O7_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3P1_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3P2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3P3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3P4_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3P5_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_L3Q3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_M1L2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_M1L3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_M2L2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_M2L3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_M3L2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_M3L3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_M4L2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_M4L3_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_M5L2_AUGER];
		rv -= Auger_Transition_Individual[Z][L1_M5L3_AUGER];
	}
	else if (shell == L2_SHELL) {
		rv -= Auger_Transition_Individual[Z][L2_L3L3_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3M1_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3M2_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3M3_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3M4_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3M5_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3N1_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3N2_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3N3_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3N4_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3N5_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3N6_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3N7_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3O1_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3O2_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3O3_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3O4_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3O5_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3O6_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3O7_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3P1_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3P2_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3P3_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3P4_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3P5_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_L3Q3_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_M1L3_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_M2L3_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_M3L3_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_M4L3_AUGER];
		rv -= Auger_Transition_Individual[Z][L2_M5L3_AUGER];
	}
	else if (shell == M1_SHELL) {
		rv -= Auger_Transition_Individual[Z][M1_M2M2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2M3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2M4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2M5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2N1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2N2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2N3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2N4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2N5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2N6_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2N7_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2O1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2O2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2O3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2O4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2O5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2O6_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2O7_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2P1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2P2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2P3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2P4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2P5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M2Q3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3M2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3M3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3M4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3M5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3N1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3N2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3N3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3N4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3N5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3N6_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3N7_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3O1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3O2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3O3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3O4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3O5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3O6_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3O7_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3P1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3P2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3P3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3P4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3P5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M3Q3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4M2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4M3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4M4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4M5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4N1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4N2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4N3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4N4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4N5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4N6_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4N7_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4O1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4O2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4O3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4O4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4O5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4O6_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4O7_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4P1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4P2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4P3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4P4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4P5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M4Q3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5M2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5M3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5M4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5M5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5N1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5N2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5N3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5N4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5N5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5N6_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5N7_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5O1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5O2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5O3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5O4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5O5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5O6_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5O7_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5P1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5P2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5P3_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5P4_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5P5_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][M1_M5Q3_AUGER];
	}
	else if (shell == M2_SHELL) {
		rv -= Auger_Transition_Individual[Z][M2_M3M3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3M4_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3M5_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3N1_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3N2_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3N3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3N4_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3N5_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3N6_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3N7_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3O1_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3O2_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3O3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3O4_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3O5_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3O6_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3O7_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3P1_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3P2_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3P3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3P4_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3P5_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M3Q3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4M3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4M4_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4M5_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4N1_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4N2_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4N3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4N4_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4N5_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4N6_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4N7_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4O1_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4O2_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4O3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4O4_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4O5_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4O6_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4O7_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4P1_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4P2_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4P3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4P4_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4P5_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M4Q3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5M3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5M4_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5M5_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5N1_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5N2_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5N3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5N4_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5N5_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5N6_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5N7_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5O1_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5O2_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5O3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5O4_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5O5_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5O6_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5O7_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5P1_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5P2_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5P3_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5P4_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5P5_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][M2_M5Q3_AUGER];
	}
	else if (shell == M3_SHELL) {
		rv -= Auger_Transition_Individual[Z][M3_M4M4_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4M5_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4N1_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4N2_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4N3_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4N4_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4N5_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4N6_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4N7_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4O1_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4O2_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4O3_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4O4_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4O5_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4O6_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4O7_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4P1_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4P2_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4P3_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4P4_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4P5_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M4Q3_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5M4_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5M5_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5N1_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5N2_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5N3_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5N4_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5N5_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5N6_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5N7_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5O1_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5O2_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5O3_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5O4_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5O5_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5O6_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5O7_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5P1_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5P2_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5P3_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5P4_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5P5_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][M3_M5Q3_AUGER];
	}
	else if (shell == M4_SHELL) {
		rv -= Auger_Transition_Individual[Z][M4_M5M5_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5N1_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5N2_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5N3_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5N4_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5N5_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5N6_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5N7_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5O1_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5O2_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5O3_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5O4_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5O5_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5O6_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5O7_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5P1_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5P2_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5P3_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5P4_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5P5_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5Q1_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5Q2_AUGER];
		rv -= Auger_Transition_Individual[Z][M4_M5Q3_AUGER];
	}

	return rv;

}

static double AugerRate_prdata(int Z, int auger_trans) {
	double rv;
	double yield, yield2;

	rv = 0.0;

	if (Z > ZMAX || Z < 1) {
		return rv;
	}
	else if (auger_trans < K_L1L1_AUGER || auger_trans > M4_M5Q3_AUGER) {
		return rv;
	}

	switch (auger_trans) {
		case L1_L2L2_AUGER:
		case L1_L2L3_AUGER:
		case L1_L2M1_AUGER:
		case L1_L2M2_AUGER:
		case L1_L2M3_AUGER:
		case L1_L2M4_AUGER:
		case L1_L2M5_AUGER:
		case L1_L2N1_AUGER:
		case L1_L2N2_AUGER:
		case L1_L2N3_AUGER:
		case L1_L2N4_AUGER:
		case L1_L2N5_AUGER:
		case L1_L2N6_AUGER:
		case L1_L2N7_AUGER:
		case L1_L2O1_AUGER:
		case L1_L2O2_AUGER:
		case L1_L2O3_AUGER:
		case L1_L2O4_AUGER:
		case L1_L2O5_AUGER:
		case L1_L2O6_AUGER:
		case L1_L2O7_AUGER:
		case L1_L2P1_AUGER:
		case L1_L2P2_AUGER:
		case L1_L2P3_AUGER:
		case L1_L2P4_AUGER:
		case L1_L2P5_AUGER:
		case L1_L2Q1_AUGER:
		case L1_L2Q2_AUGER:
		case L1_L2Q3_AUGER:
		case L1_L3L2_AUGER:
		case L1_L3L3_AUGER:
		case L1_L3M1_AUGER:
		case L1_L3M2_AUGER:
		case L1_L3M3_AUGER:
		case L1_L3M4_AUGER:
		case L1_L3M5_AUGER:
		case L1_L3N1_AUGER:
		case L1_L3N2_AUGER:
		case L1_L3N3_AUGER:
		case L1_L3N4_AUGER:
		case L1_L3N5_AUGER:
		case L1_L3N6_AUGER:
		case L1_L3N7_AUGER:
		case L1_L3O1_AUGER:
		case L1_L3O2_AUGER:
		case L1_L3O3_AUGER:
		case L1_L3O4_AUGER:
		case L1_L3O5_AUGER:
		case L1_L3O6_AUGER:
		case L1_L3O7_AUGER:
		case L1_L3P1_AUGER:
		case L1_L3P2_AUGER:
		case L1_L3P3_AUGER:
		case L1_L3P4_AUGER:
		case L1_L3P5_AUGER:
		case L1_L3Q1_AUGER:
		case L1_L3Q2_AUGER:
		case L1_L3Q3_AUGER:
		case L1_M1L2_AUGER:
		case L1_M1L3_AUGER:
		case L1_M2L2_AUGER:
		case L1_M2L3_AUGER:
		case L1_M3L2_AUGER:
		case L1_M3L3_AUGER:
		case L1_M4L2_AUGER:
		case L1_M4L3_AUGER:
		case L1_M5L2_AUGER:
		case L1_M5L3_AUGER:
		case L2_L3L3_AUGER:
		case L2_L3M1_AUGER:
		case L2_L3M2_AUGER:
		case L2_L3M3_AUGER:
		case L2_L3M4_AUGER:
		case L2_L3M5_AUGER:
		case L2_L3N1_AUGER:
		case L2_L3N2_AUGER:
		case L2_L3N3_AUGER:
		case L2_L3N4_AUGER:
		case L2_L3N5_AUGER:
		case L2_L3N6_AUGER:
		case L2_L3N7_AUGER:
		case L2_L3O1_AUGER:
		case L2_L3O2_AUGER:
		case L2_L3O3_AUGER:
		case L2_L3O4_AUGER:
		case L2_L3O5_AUGER:
		case L2_L3O6_AUGER:
		case L2_L3O7_AUGER:
		case L2_L3P1_AUGER:
		case L2_L3P2_AUGER:
		case L2_L3P3_AUGER:
		case L2_L3P4_AUGER:
		case L2_L3P5_AUGER:
		case L2_L3Q1_AUGER:
		case L2_L3Q2_AUGER:
		case L2_L3Q3_AUGER:
		case L2_M1L3_AUGER:
		case L2_M2L3_AUGER:
		case L2_M3L3_AUGER:
		case L2_M4L3_AUGER:
		case L2_M5L3_AUGER:
		case M1_M2M2_AUGER:
		case M1_M2M3_AUGER:
		case M1_M2M4_AUGER:
		case M1_M2M5_AUGER:
		case M1_M2N1_AUGER:
		case M1_M2N2_AUGER:
		case M1_M2N3_AUGER:
		case M1_M2N4_AUGER:
		case M1_M2N5_AUGER:
		case M1_M2N6_AUGER:
		case M1_M2N7_AUGER:
		case M1_M2O1_AUGER:
		case M1_M2O2_AUGER:
		case M1_M2O3_AUGER:
		case M1_M2O4_AUGER:
		case M1_M2O5_AUGER:
		case M1_M2O6_AUGER:
		case M1_M2O7_AUGER:
		case M1_M2P1_AUGER:
		case M1_M2P2_AUGER:
		case M1_M2P3_AUGER:
		case M1_M2P4_AUGER:
		case M1_M2P5_AUGER:
		case M1_M2Q1_AUGER:
		case M1_M2Q2_AUGER:
		case M1_M2Q3_AUGER:
		case M1_M3M2_AUGER:
		case M1_M3M3_AUGER:
		case M1_M3M4_AUGER:
		case M1_M3M5_AUGER:
		case M1_M3N1_AUGER:
		case M1_M3N2_AUGER:
		case M1_M3N3_AUGER:
		case M1_M3N4_AUGER:
		case M1_M3N5_AUGER:
		case M1_M3N6_AUGER:
		case M1_M3N7_AUGER:
		case M1_M3O1_AUGER:
		case M1_M3O2_AUGER:
		case M1_M3O3_AUGER:
		case M1_M3O4_AUGER:
		case M1_M3O5_AUGER:
		case M1_M3O6_AUGER:
		case M1_M3O7_AUGER:
		case M1_M3P1_AUGER:
		case M1_M3P2_AUGER:
		case M1_M3P3_AUGER:
		case M1_M3P4_AUGER:
		case M1_M3P5_AUGER:
		case M1_M3Q1_AUGER:
		case M1_M3Q2_AUGER:
		case M1_M3Q3_AUGER:
		case M1_M4M2_AUGER:
		case M1_M4M3_AUGER:
		case M1_M4M4_AUGER:
		case M1_M4M5_AUGER:
		case M1_M4N1_AUGER:
		case M1_M4N2_AUGER:
		case M1_M4N3_AUGER:
		case M1_M4N4_AUGER:
		case M1_M4N5_AUGER:
		case M1_M4N6_AUGER:
		case M1_M4N7_AUGER:
		case M1_M4O1_AUGER:
		case M1_M4O2_AUGER:
		case M1_M4O3_AUGER:
		case M1_M4O4_AUGER:
		case M1_M4O5_AUGER:
		case M1_M4O6_AUGER:
		case M1_M4O7_AUGER:
		case M1_M4P1_AUGER:
		case M1_M4P2_AUGER:
		case M1_M4P3_AUGER:
		case M1_M4P4_AUGER:
		case M1_M4P5_AUGER:
		case M1_M4Q1_AUGER:
		case M1_M4Q2_AUGER:
		case M1_M4Q3_AUGER:
		case M1_M5M2_AUGER:
		case M1_M5M3_AUGER:
		case M1_M5M4_AUGER:
		case M1_M5M5_AUGER:
		case M1_M5N1_AUGER:
		case M1_M5N2_AUGER:
		case M1_M5N3_AUGER:
		case M1_M5N4_AUGER:
		case M1_M5N5_AUGER:
		case M1_M5N6_AUGER:
		case M1_M5N7_AUGER:
		case M1_M5O1_AUGER:
		case M1_M5O2_AUGER:
		case M1_M5O3_AUGER:
		case M1_M5O4_AUGER:
		case M1_M5O5_AUGER:
		case M1_M5O6_AUGER:
		case M1_M5O7_AUGER:
		case M1_M5P1_AUGER:
		case M1_M5P2_AUGER:
		case M1_M5P3_AUGER:
		case M1_M5P4_AUGER:
		case M1_M5P5_AUGER:
		case M1_M5Q1_AUGER:
		case M1_M5Q2_AUGER:
		case M1_M5Q3_AUGER:
		case M2_M3M3_AUGER:
		case M2_M3M4_AUGER:
		case M2_M3M5_AUGER:
		case M2_M3N1_AUGER:
		case M2_M3N2_AUGER:
		case M2_M3N3_AUGER:
		case M2_M3N4_AUGER:
		case M2_M3N5_AUGER:
		case M2_M3N6_AUGER:
		case M2_M3N7_AUGER:
		case M2_M3O1_AUGER:
		case M2_M3O2_AUGER:
		case M2_M3O3_AUGER:
		case M2_M3O4_AUGER:
		case M2_M3O5_AUGER:
		case M2_M3O6_AUGER:
		case M2_M3O7_AUGER:
		case M2_M3P1_AUGER:
		case M2_M3P2_AUGER:
		case M2_M3P3_AUGER:
		case M2_M3P4_AUGER:
		case M2_M3P5_AUGER:
		case M2_M3Q1_AUGER:
		case M2_M3Q2_AUGER:
		case M2_M3Q3_AUGER:
		case M2_M4M3_AUGER:
		case M2_M4M4_AUGER:
		case M2_M4M5_AUGER:
		case M2_M4N1_AUGER:
		case M2_M4N2_AUGER:
		case M2_M4N3_AUGER:
		case M2_M4N4_AUGER:
		case M2_M4N5_AUGER:
		case M2_M4N6_AUGER:
		case M2_M4N7_AUGER:
		case M2_M4O1_AUGER:
		case M2_M4O2_AUGER:
		case M2_M4O3_AUGER:
		case M2_M4O4_AUGER:
		case M2_M4O5_AUGER:
		case M2_M4O6_AUGER:
		case M2_M4O7_AUGER:
		case M2_M4P1_AUGER:
		case M2_M4P2_AUGER:
		case M2_M4P3_AUGER:
		case M2_M4P4_AUGER:
		case M2_M4P5_AUGER:
		case M2_M4Q1_AUGER:
		case M2_M4Q2_AUGER:
		case M2_M4Q3_AUGER:
		case M2_M5M3_AUGER:
		case M2_M5M4_AUGER:
		case M2_M5M5_AUGER:
		case M2_M5N1_AUGER:
		case M2_M5N2_AUGER:
		case M2_M5N3_AUGER:
		case M2_M5N4_AUGER:
		case M2_M5N5_AUGER:
		case M2_M5N6_AUGER:
		case M2_M5N7_AUGER:
		case M2_M5O1_AUGER:
		case M2_M5O2_AUGER:
		case M2_M5O3_AUGER:
		case M2_M5O4_AUGER:
		case M2_M5O5_AUGER:
		case M2_M5O6_AUGER:
		case M2_M5O7_AUGER:
		case M2_M5P1_AUGER:
		case M2_M5P2_AUGER:
		case M2_M5P3_AUGER:
		case M2_M5P4_AUGER:
		case M2_M5P5_AUGER:
		case M2_M5Q1_AUGER:
		case M2_M5Q2_AUGER:
		case M2_M5Q3_AUGER:
		case M3_M4M4_AUGER:
		case M3_M4M5_AUGER:
		case M3_M4N1_AUGER:
		case M3_M4N2_AUGER:
		case M3_M4N3_AUGER:
		case M3_M4N4_AUGER:
		case M3_M4N5_AUGER:
		case M3_M4N6_AUGER:
		case M3_M4N7_AUGER:
		case M3_M4O1_AUGER:
		case M3_M4O2_AUGER:
		case M3_M4O3_AUGER:
		case M3_M4O4_AUGER:
		case M3_M4O5_AUGER:
		case M3_M4O6_AUGER:
		case M3_M4O7_AUGER:
		case M3_M4P1_AUGER:
		case M3_M4P2_AUGER:
		case M3_M4P3_AUGER:
		case M3_M4P4_AUGER:
		case M3_M4P5_AUGER:
		case M3_M4Q1_AUGER:
		case M3_M4Q2_AUGER:
		case M3_M4Q3_AUGER:
		case M3_M5M4_AUGER:
		case M3_M5M5_AUGER:
		case M3_M5N1_AUGER:
		case M3_M5N2_AUGER:
		case M3_M5N3_AUGER:
		case M3_M5N4_AUGER:
		case M3_M5N5_AUGER:
		case M3_M5N6_AUGER:
		case M3_M5N7_AUGER:
		case M3_M5O1_AUGER:
		case M3_M5O2_AUGER:
		case M3_M5O3_AUGER:
		case M3_M5O4_AUGER:
		case M3_M5O5_AUGER:
		case M3_M5O6_AUGER:
		case M3_M5O7_AUGER:
		case M3_M5P1_AUGER:
		case M3_M5P2_AUGER:
		case M3_M5P3_AUGER:
		case M3_M5P4_AUGER:
		case M3_M5P5_AUGER:
		case M3_M5Q1_AUGER:
		case M3_M5Q2_AUGER:
		case M3_M5Q3_AUGER:
		case M4_M5M5_AUGER:
		case M4_M5N1_AUGER:
		case M4_M5N2_AUGER:
		case M4_M5N3_AUGER:
		case M4_M5N4_AUGER:
		case M4_M5N5_AUGER:
		case M4_M5N6_AUGER:
		case M4_M5N7_AUGER:
		case M4_M5O1_AUGER:
		case M4_M5O2_AUGER:
		case M4_M5O3_AUGER:
		case M4_M5O4_AUGER:
		case M4_M5O5_AUGER:
		case M4_M5O6_AUGER:
		case M4_M5O7_AUGER:
		case M4_M5P1_AUGER:
		case M4_M5P2_AUGER:
		case M4_M5P3_AUGER:
		case M4_M5P4_AUGER:
		case M4_M5P5_AUGER:
		case M4_M5Q1_AUGER:
		case M4_M5Q2_AUGER:
		case M4_M5Q3_AUGER:
		return rv;
	}

	if (Auger_Transition_Individual[Z][auger_trans] == 0.0)
		return rv;

	if (auger_trans >= K_L1L1_AUGER && auger_trans < L1_L2L2_AUGER  ) {
		yield2 = AugerYield2_prdata(Z, K_SHELL);
		if (yield2 < 1E-8)
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]/yield2;
	}
	else if (auger_trans >= L1_L2L2_AUGER && auger_trans < L2_L3L3_AUGER) {
		yield2 = AugerYield2_prdata(Z, L1_SHELL);
		if (yield2 < 1E-8)
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]/yield2;
	}
	else if (auger_trans >= L2_L3L3_AUGER && auger_trans < L3_M1M1_AUGER) {
		yield2 = AugerYield2_prdata(Z, L2_SHELL);
		if (yield2 < 1E-8)
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]/yield2;
	}
	else if (auger_trans >= L3_M1M1_AUGER && auger_trans < M1_M2M2_AUGER) {
		yield2 = AugerYield2_prdata(Z, L3_SHELL);
		if (yield2 < 1E-8)
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]/yield2;
	}
	else if (auger_trans >= M1_M2M2_AUGER && auger_trans < M2_M3M3_AUGER) {
		yield2 = AugerYield2_prdata(Z, M1_SHELL);
		if (yield2 < 1E-8)
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]/yield2;
	}
	else if (auger_trans >= M2_M3M3_AUGER && auger_trans < M3_M4M4_AUGER) {
		yield2 = AugerYield2_prdata(Z, M2_SHELL);
		if (yield2 < 1E-8)
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]/yield2;
	}
	else if (auger_trans >= M3_M4M4_AUGER && auger_trans < M4_M5M5_AUGER) {
		yield2 = AugerYield2_prdata(Z, M3_SHELL);
		if (yield2 < 1E-8)
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]/yield2;
	}
	else if (auger_trans >= M4_M5M5_AUGER && auger_trans <= M4_M5Q3_AUGER) {
		yield2 = AugerYield2_prdata(Z, M4_SHELL);
		if (yield2 < 1E-8)
			return rv;
		return Auger_Transition_Individual[Z][auger_trans]/yield2;
	}

	return rv;
}

/*----------------------------------------------------- */

void print_mendelvec(int arrmax, struct MendelElement *arr)
{
  int i;
  int MENDEL_PER_LINE = 10;
  fprintf(f, "{\n");
  for(i = 0; i < arrmax; i++) {
    fprintf(f, "{%d,\"%s\"}, ", arr[i].Zatom, arr[i].name);
    if(i%MENDEL_PER_LINE == (MENDEL_PER_LINE-1))
      fprintf(f, "\n");
  }
  fprintf(f, "}");
  fprintf(f, ";\n\n");
}

void print_doublevec(int arrmax, double *arr)
{
  fwrite(arr, sizeof(double), arrmax, f);
}

void print_intvec(int arrmax, int *arr)
{
  fwrite(arr, sizeof(int), arrmax, f);
}


int main(int argc, char *argv[])
{

  int i,j;
  Crystal_Struct* crystal;
  Crystal_Atom* atom;
  int zmax = ZMAX;
  int shellnum = SHELLNUM;
  int shellnum_k = SHELLNUM_K;
  int shellnum_a = SHELLNUM_A;
  int transnum = TRANSNUM;
  int linenum = LINENUM;
  int augernum = AUGERNUM;
  double re2 = RE2;
  double mec2 = MEC2;
  double avognum = AVOGNUM;
  double kev2angst = KEV2ANGST;
  double r_e = R_E;

  if (argc != 2) {
	  fprintf(stderr, "Invoke this program with the xraylib source root directory as only argument!\n");
	  return 1;
  }

  XRayInit();
  XRayInitFromPath(argv[1]);

  f = fopen(OUTFILE, "w");
  if(f == NULL) {
    perror("file open");
  }


  //fprintf(f, "public static final int ZMAX = %i;\n", ZMAX);
  //fprintf(f, "public static final int SHELLNUM = %i;\n", SHELLNUM);
  //fprintf(f, "public static final int TRANSNUM = %i;\n", TRANSNUM);



  fwrite(&zmax, sizeof(int), 1, f);
  fwrite(&shellnum, sizeof(int), 1, f);
  fwrite(&shellnum_k, sizeof(int), 1, f);
  fwrite(&shellnum_a, sizeof(int), 1, f);
  fwrite(&transnum, sizeof(int), 1, f);
  fwrite(&linenum, sizeof(int), 1, f);
  fwrite(&augernum, sizeof(int), 1, f);
  fwrite(&re2, sizeof(double), 1, f);
  fwrite(&mec2, sizeof(double), 1, f);
  fwrite(&avognum, sizeof(double), 1, f);
  fwrite(&kev2angst, sizeof(double), 1, f);
  fwrite(&r_e, sizeof(double), 1, f);

  /*
  fprintf(f, "#include \"xraylib-defs.h\"\n\n");
  fprintf(f, "#include \"xrayglob.h\"\n\n");
  fprintf(f, "#include \"stddef.h\"\n\n");
  */

  /*fprintf(f, "struct MendelElement MendelArray[MENDEL_MAX] = \n");
  print_mendelvec(MENDEL_MAX, MendelArray);

  fprintf(f, "struct MendelElement MendelArraySorted[MENDEL_MAX] = \n");
  print_mendelvec(MENDEL_MAX, MendelArraySorted);


  fprintf (f, "};\n\n");

  fprintf(f, "Crystal_Array Crystal_arr = {%i, %i, __Crystal_arr};\n\n", Crystal_arr.n_crystal, Crystal_arr.n_alloc);
  */

  print_doublevec(ZMAX+1, AtomicWeight_arr);

  print_doublevec(ZMAX+1, ElementDensity_arr);

  PR_MATD(EdgeEnergy_arr);

  PR_MATD(AtomicLevelWidth_arr);

  PR_MATD(LineEnergy_arr);

  PR_MATD(FluorYield_arr);

  PR_MATD(JumpFactor_arr);

  PR_MATD(CosKron_arr);

  PR_MATD(RadRate_arr);

  PR_NUMVEC1D(NE_Photo, "NE_Photo");
  PR_DYNMATD(NE_Photo, E_Photo_arr, "E_Photo_arr");
  PR_DYNMATD(NE_Photo, CS_Photo_arr, "CS_Photo_arr");
  PR_DYNMATD(NE_Photo, CS_Photo_arr2, "CS_Photo_arr2");


  PR_NUMVEC1D(NE_Rayl, "NE_Rayl");
  PR_DYNMATD(NE_Rayl, E_Rayl_arr, "E_Rayl_arr");
  PR_DYNMATD(NE_Rayl, CS_Rayl_arr, "CS_Rayl_arr");
  PR_DYNMATD(NE_Rayl, CS_Rayl_arr2, "CS_Rayl_arr2");

  PR_NUMVEC1D(NE_Compt, "NE_Compt");
  PR_DYNMATD(NE_Compt, E_Compt_arr, "E_Compt_arr");
  PR_DYNMATD(NE_Compt, CS_Compt_arr, "CS_Compt_arr");
  PR_DYNMATD(NE_Compt, CS_Compt_arr2, "CS_Compt_arr2");

  PR_NUMVEC1D(NE_Energy, "NE_Energy");
  PR_DYNMATD(NE_Energy, E_Energy_arr, "E_Energy_arr");
  PR_DYNMATD(NE_Energy, CS_Energy_arr, "CS_Energy_arr");
  PR_DYNMATD(NE_Energy, CS_Energy_arr2, "CS_Energy_arr2");


  PR_NUMVEC1D(Nq_Rayl, "Nq_Rayl");
  PR_DYNMATD(Nq_Rayl, q_Rayl_arr, "q_Rayl_arr");
  PR_DYNMATD(Nq_Rayl, FF_Rayl_arr, "FF_Rayl_arr");
  PR_DYNMATD(Nq_Rayl, FF_Rayl_arr2, "FF_Rayl_arr2");

  PR_NUMVEC1D(Nq_Compt, "Nq_Compt");
  PR_DYNMATD(Nq_Compt, q_Compt_arr, "q_Compt_arr");
  PR_DYNMATD(Nq_Compt, SF_Compt_arr, "SF_Compt_arr");
  PR_DYNMATD(Nq_Compt, SF_Compt_arr2, "SF_Compt_arr2");

  PR_NUMVEC1D(NE_Fi, "NE_Fi");
  PR_DYNMATD(NE_Fi, E_Fi_arr, "E_Fi_arr");
  PR_DYNMATD(NE_Fi, Fi_arr, "Fi_arr");
  PR_DYNMATD(NE_Fi, Fi_arr2, "Fi_arr2");

  PR_NUMVEC1D(NE_Fii, "NE_Fii");
  PR_DYNMATD(NE_Fii, E_Fii_arr, "E_Fii_arr");
  PR_DYNMATD(NE_Fii, Fii_arr, "Fii_arr");
  PR_DYNMATD(NE_Fii, Fii_arr2, "Fii_arr2");

  PR_MATD(Electron_Config_Kissel);

  PR_MATD(EdgeEnergy_Kissel);

  PR_NUMVEC1D(NE_Photo_Total_Kissel, "NE_Photo_Total_Kissel");

  PR_MATI(NE_Photo_Partial_Kissel);
  PR_DYNMAT_3DD_K(NE_Photo_Partial_Kissel, E_Photo_Partial_Kissel, "E_Photo_Partial_Kissel");
  PR_DYNMAT_3DD_K(NE_Photo_Partial_Kissel, Photo_Partial_Kissel, "Photo_Partial_Kissel");
  PR_DYNMAT_3DD_K(NE_Photo_Partial_Kissel, Photo_Partial_Kissel2, "Photo_Partial_Kissel2");

  PR_NUMVEC1D(NShells_ComptonProfiles, "NShells_ComptonProfiles");
  PR_NUMVEC1D(Npz_ComptonProfiles, "Npz_ComptonProfiles");
  PR_DYNMATD(NShells_ComptonProfiles,UOCCUP_ComptonProfiles,"UOCCUP_ComptonProfiles");
  PR_DYNMATD(Npz_ComptonProfiles,pz_ComptonProfiles,"pz_ComptonProfiles");
  PR_DYNMATD(Npz_ComptonProfiles,Total_ComptonProfiles,"Total_ComptonProfiles");
  PR_DYNMATD(Npz_ComptonProfiles,Total_ComptonProfiles2,"Total_ComptonProfiles2");
  PR_DYNMAT_3DD_C(Npz_ComptonProfiles, NShells_ComptonProfiles, UOCCUP_ComptonProfiles, Partial_ComptonProfiles,"Partial_ComptonProfiles");
  PR_DYNMAT_3DD_C(Npz_ComptonProfiles, NShells_ComptonProfiles, UOCCUP_ComptonProfiles, Partial_ComptonProfiles2,"Partial_ComptonProfiles2");

  for (i = 1 ; i < ZMAX ; i++) {
  	for (j = K_L1L1_AUGER ; j <= M4_M5Q3_AUGER ; j++)
		Auger_Rates[i][j] = AugerRate_prdata(i, j);

	for (j = K_SHELL ; j <= M5_SHELL ; j++)
		Auger_Yields[i][j] = AugerYield_prdata(i, j);
  }
  PR_MATD(Auger_Yields);
  PR_MATD(Auger_Rates);

  // NIST compounds
  fwrite(&nCompoundDataNISTList, sizeof(int), 1, f);
  for (i = 0 ; i < nCompoundDataNISTList ; i++) {
  	fwrite(compoundDataNISTList[i].name, sizeof(char), strlen(compoundDataNISTList[i].name)+1, f);
	fwrite(&compoundDataNISTList[i].nElements, sizeof(int), 1, f);
	fwrite(compoundDataNISTList[i].Elements, sizeof(int), compoundDataNISTList[i].nElements, f);
	fwrite(compoundDataNISTList[i].massFractions, sizeof(double), compoundDataNISTList[i].nElements, f);
	fwrite(&compoundDataNISTList[i].density, sizeof(double), 1, f);
  }

  fwrite(&nNuclideDataList, sizeof(int), 1, f);
  for (i = 0 ; i < nNuclideDataList ; i++) {
    fwrite(nuclideDataList[i].name, sizeof(char), strlen(nuclideDataList[i].name)+1, f);
    fwrite(&nuclideDataList[i].Z, sizeof(int), 1, f);
    fwrite(&nuclideDataList[i].A, sizeof(int), 1, f);
    fwrite(&nuclideDataList[i].N, sizeof(int), 1, f);
    fwrite(&nuclideDataList[i].Z_xray, sizeof(int), 1, f);
    fwrite(&nuclideDataList[i].nXrays, sizeof(int), 1, f);
    fwrite(nuclideDataList[i].XrayLines, sizeof(int), nuclideDataList[i].nXrays, f);
    fwrite(nuclideDataList[i].XrayIntensities, sizeof(double), nuclideDataList[i].nXrays, f);
    fwrite(&nuclideDataList[i].nGammas, sizeof(int), 1, f);
    fwrite(nuclideDataList[i].GammaEnergies, sizeof(double), nuclideDataList[i].nGammas, f);
    fwrite(nuclideDataList[i].GammaIntensities, sizeof(double), nuclideDataList[i].nGammas, f);
  }

  fwrite(&Crystal_arr.n_crystal, sizeof(int), 1, f);
  for (i = 0; i < Crystal_arr.n_crystal; i++) {
    crystal = &Crystal_arr.crystal[i];
    fwrite(crystal->name, sizeof(char), strlen(crystal->name)+1, f);
    fwrite(&crystal->a, sizeof(double), 1, f);
    fwrite(&crystal->b, sizeof(double), 1, f);
    fwrite(&crystal->c, sizeof(double), 1, f);
    fwrite(&crystal->alpha, sizeof(double), 1, f);
    fwrite(&crystal->beta, sizeof(double), 1, f);
    fwrite(&crystal->gamma, sizeof(double), 1, f);
    fwrite(&crystal->volume, sizeof(double), 1, f);
    fwrite(&crystal->n_atom, sizeof(int), 1, f);
    for (j = 0; j < crystal->n_atom; j++) {
      atom = &crystal->atom[j];
      fwrite(&atom->Zatom, sizeof(int), 1, f);
      fwrite(&atom->fraction, sizeof(double), 1, f);
      fwrite(&atom->x, sizeof(double), 1, f);
      fwrite(&atom->y, sizeof(double), 1, f);
      fwrite(&atom->z, sizeof(double), 1, f);
    }
  }

#define IF_XRF_CS(shell) \
      if (shell1 == shell ## _SHELL) { \
	int shell2; \
        for (shell2 = K_SHELL ; shell2 <= L3_SHELL ; shell2++) { \
          xrf_cross_sections_constants_full[Z][shell1][shell2] = P ## shell ## _get_cross_sections_constant_full(Z, shell2); \
          xrf_cross_sections_constants_auger_only[Z][shell1][shell2] = P ## shell ## _get_cross_sections_constant_auger_only(Z, shell2); \
	} \
      }

  /* precalculated xrf_cross_sections constants */
  int Z;
  for (Z = 1 ; Z <= ZMAX ; Z++) {
    int shell1;
    for (shell1 = L1_SHELL ; shell1 <= M5_SHELL ; shell1++) {
      IF_XRF_CS(L1)
      else IF_XRF_CS(L2)
      else IF_XRF_CS(L3)
      else IF_XRF_CS(M1)
      else IF_XRF_CS(M2)
      else IF_XRF_CS(M3)
      else IF_XRF_CS(M4)
      else IF_XRF_CS(M5)
    }
  }

  PR_MATD(xrf_cross_sections_constants_full)
  PR_MATD(xrf_cross_sections_constants_auger_only)

  fclose(f);

  return 0;
}
