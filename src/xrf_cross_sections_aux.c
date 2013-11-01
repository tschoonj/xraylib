/*
Copyright (c) 2009, 2010, 2011 Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "xrf_cross_sections_aux.h"


double PL1_pure_kissel(int Z, double E) {
	return CS_Photo_Partial(Z, L1_SHELL, E);
}

double PL1_rad_cascade_kissel(int Z, double E, double PK) {
	double rv;
	rv = CS_Photo_Partial(Z,L1_SHELL, E);

	if (PK > 0.0 && RadRate(Z,KL1_LINE) > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KL1_LINE);

	return rv;
}

double PL1_auger_cascade_kissel(int Z, double E, double PK) {
	double rv;
	
	rv = CS_Photo_Partial(Z,L1_SHELL, E);
	if (PK > 0.0)
		rv += (AugerYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1L1_AUGER)+
		AugerRate(Z,K_L1L2_AUGER)+
		AugerRate(Z,K_L1L3_AUGER)+
		AugerRate(Z,K_L1M1_AUGER)+
		AugerRate(Z,K_L1M2_AUGER)+
		AugerRate(Z,K_L1M3_AUGER)+
		AugerRate(Z,K_L1M4_AUGER)+
		AugerRate(Z,K_L1M5_AUGER)+
		AugerRate(Z,K_L1N1_AUGER)+
		AugerRate(Z,K_L1N2_AUGER)+
		AugerRate(Z,K_L1N3_AUGER)+
		AugerRate(Z,K_L1N4_AUGER)+
		AugerRate(Z,K_L1N5_AUGER)+
		AugerRate(Z,K_L1N6_AUGER)+
		AugerRate(Z,K_L1N7_AUGER)+
		AugerRate(Z,K_L1O1_AUGER)+
		AugerRate(Z,K_L1O2_AUGER)+
		AugerRate(Z,K_L1O3_AUGER)+
		AugerRate(Z,K_L1O4_AUGER)+
		AugerRate(Z,K_L1O5_AUGER)+
		AugerRate(Z,K_L1O6_AUGER)+
		AugerRate(Z,K_L1O7_AUGER)+
		AugerRate(Z,K_L1P1_AUGER)+
		AugerRate(Z,K_L1P2_AUGER)+
		AugerRate(Z,K_L1P3_AUGER)+
		AugerRate(Z,K_L1P4_AUGER)+
		AugerRate(Z,K_L1P5_AUGER)+
		AugerRate(Z,K_L1Q1_AUGER)+
		AugerRate(Z,K_L1Q2_AUGER)+
		AugerRate(Z,K_L1Q3_AUGER)+
		AugerRate(Z,K_L2L1_AUGER)+
		AugerRate(Z,K_L3L1_AUGER)+
		AugerRate(Z,K_M1L1_AUGER)+
		AugerRate(Z,K_M2L1_AUGER)+
		AugerRate(Z,K_M3L1_AUGER)+
		AugerRate(Z,K_M4L1_AUGER)+
		AugerRate(Z,K_M5L1_AUGER)
		);

	return rv;	
}

double PL1_full_cascade_kissel(int Z, double E, double PK) {
	double rv;

	rv = CS_Photo_Partial(Z,L1_SHELL, E);
	if (PK > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KL1_LINE)+
		(AugerYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1L1_AUGER)+
		AugerRate(Z,K_L1L2_AUGER)+
		AugerRate(Z,K_L1L3_AUGER)+
		AugerRate(Z,K_L1M1_AUGER)+
		AugerRate(Z,K_L1M2_AUGER)+
		AugerRate(Z,K_L1M3_AUGER)+
		AugerRate(Z,K_L1M4_AUGER)+
		AugerRate(Z,K_L1M5_AUGER)+
		AugerRate(Z,K_L1N1_AUGER)+
		AugerRate(Z,K_L1N2_AUGER)+
		AugerRate(Z,K_L1N3_AUGER)+
		AugerRate(Z,K_L1N4_AUGER)+
		AugerRate(Z,K_L1N5_AUGER)+
		AugerRate(Z,K_L1N6_AUGER)+
		AugerRate(Z,K_L1N7_AUGER)+
		AugerRate(Z,K_L1O1_AUGER)+
		AugerRate(Z,K_L1O2_AUGER)+
		AugerRate(Z,K_L1O3_AUGER)+
		AugerRate(Z,K_L1O4_AUGER)+
		AugerRate(Z,K_L1O5_AUGER)+
		AugerRate(Z,K_L1O6_AUGER)+
		AugerRate(Z,K_L1O7_AUGER)+
		AugerRate(Z,K_L1P1_AUGER)+
		AugerRate(Z,K_L1P2_AUGER)+
		AugerRate(Z,K_L1P3_AUGER)+
		AugerRate(Z,K_L1P4_AUGER)+
		AugerRate(Z,K_L1P5_AUGER)+
		AugerRate(Z,K_L1Q1_AUGER)+
		AugerRate(Z,K_L1Q2_AUGER)+
		AugerRate(Z,K_L1Q3_AUGER)+
		AugerRate(Z,K_L2L1_AUGER)+
		AugerRate(Z,K_L3L1_AUGER)+
		AugerRate(Z,K_M1L1_AUGER)+
		AugerRate(Z,K_M2L1_AUGER)+
		AugerRate(Z,K_M3L1_AUGER)+
		AugerRate(Z,K_M4L1_AUGER)+
		AugerRate(Z,K_M5L1_AUGER)
		);
	return rv;
}

double PL2_pure_kissel(int Z, double E, double PL1) {
	double rv;

	rv = CS_Photo_Partial(Z, L2_SHELL, E);
	if (PL1 > 0.0)
		rv +=CosKronTransProb(Z,FL12_TRANS)*PL1;
	return rv;	
}

double PL2_rad_cascade_kissel(int Z, double E, double PK, double PL1) {
	double rv;

	rv = CS_Photo_Partial(Z,L2_SHELL, E);
	if (PK > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KL2_LINE);

	if (PL1 > 0.0)
		rv += CosKronTransProb(Z,FL12_TRANS)*PL1;
	return  rv;
}

double PL2_auger_cascade_kissel(int Z, double E, double PK, double PL1) {
	double rv;

	rv = CS_Photo_Partial(Z,L2_SHELL, E);

	if (PK > 0.0)
		rv += AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1L2_AUGER)+
		AugerRate(Z,K_L2L1_AUGER)+
		AugerRate(Z,K_L2L2_AUGER)+
		AugerRate(Z,K_L2L3_AUGER)+
		AugerRate(Z,K_L2M1_AUGER)+
		AugerRate(Z,K_L2M2_AUGER)+
		AugerRate(Z,K_L2M3_AUGER)+
		AugerRate(Z,K_L2M4_AUGER)+
		AugerRate(Z,K_L2M5_AUGER)+
		AugerRate(Z,K_L2N1_AUGER)+
		AugerRate(Z,K_L2N2_AUGER)+
		AugerRate(Z,K_L2N3_AUGER)+
		AugerRate(Z,K_L2N4_AUGER)+
		AugerRate(Z,K_L2N5_AUGER)+
		AugerRate(Z,K_L2N6_AUGER)+
		AugerRate(Z,K_L2N7_AUGER)+
		AugerRate(Z,K_L2O1_AUGER)+
		AugerRate(Z,K_L2O2_AUGER)+
		AugerRate(Z,K_L2O3_AUGER)+
		AugerRate(Z,K_L2O4_AUGER)+
		AugerRate(Z,K_L2O5_AUGER)+
		AugerRate(Z,K_L2O6_AUGER)+
		AugerRate(Z,K_L2O7_AUGER)+
		AugerRate(Z,K_L2P1_AUGER)+
		AugerRate(Z,K_L2P2_AUGER)+
		AugerRate(Z,K_L2P3_AUGER)+
		AugerRate(Z,K_L2P4_AUGER)+
		AugerRate(Z,K_L2P5_AUGER)+
		AugerRate(Z,K_L2Q1_AUGER)+
		AugerRate(Z,K_L2Q2_AUGER)+
		AugerRate(Z,K_L2Q3_AUGER)+
		AugerRate(Z,K_L3L2_AUGER)+
		AugerRate(Z,K_M1L2_AUGER)+
		AugerRate(Z,K_M2L2_AUGER)+
		AugerRate(Z,K_M3L2_AUGER)+
		AugerRate(Z,K_M4L2_AUGER)+
		AugerRate(Z,K_M5L2_AUGER)
		);

	if (PL1 > 0.0)
		rv += CosKronTransProb(Z,FL12_TRANS)*PL1;
	return  rv;
	
}

double PL2_full_cascade_kissel(int Z, double E, double PK, double PL1) {
	double rv;

	rv = CS_Photo_Partial(Z,L2_SHELL, E);

	if (PK > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KL2_LINE)+
		AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1L2_AUGER)+
		AugerRate(Z,K_L2L1_AUGER)+
		AugerRate(Z,K_L2L2_AUGER)+
		AugerRate(Z,K_L2L3_AUGER)+
		AugerRate(Z,K_L2M1_AUGER)+
		AugerRate(Z,K_L2M2_AUGER)+
		AugerRate(Z,K_L2M3_AUGER)+
		AugerRate(Z,K_L2M4_AUGER)+
		AugerRate(Z,K_L2M5_AUGER)+
		AugerRate(Z,K_L2N1_AUGER)+
		AugerRate(Z,K_L2N2_AUGER)+
		AugerRate(Z,K_L2N3_AUGER)+
		AugerRate(Z,K_L2N4_AUGER)+
		AugerRate(Z,K_L2N5_AUGER)+
		AugerRate(Z,K_L2N6_AUGER)+
		AugerRate(Z,K_L2N7_AUGER)+
		AugerRate(Z,K_L2O1_AUGER)+
		AugerRate(Z,K_L2O2_AUGER)+
		AugerRate(Z,K_L2O3_AUGER)+
		AugerRate(Z,K_L2O4_AUGER)+
		AugerRate(Z,K_L2O5_AUGER)+
		AugerRate(Z,K_L2O6_AUGER)+
		AugerRate(Z,K_L2O7_AUGER)+
		AugerRate(Z,K_L2P1_AUGER)+
		AugerRate(Z,K_L2P2_AUGER)+
		AugerRate(Z,K_L2P3_AUGER)+
		AugerRate(Z,K_L2P4_AUGER)+
		AugerRate(Z,K_L2P5_AUGER)+
		AugerRate(Z,K_L2Q1_AUGER)+
		AugerRate(Z,K_L2Q2_AUGER)+
		AugerRate(Z,K_L2Q3_AUGER)+
		AugerRate(Z,K_L3L2_AUGER)+
		AugerRate(Z,K_M1L2_AUGER)+
		AugerRate(Z,K_M2L2_AUGER)+
		AugerRate(Z,K_M3L2_AUGER)+
		AugerRate(Z,K_M4L2_AUGER)+
		AugerRate(Z,K_M5L2_AUGER)
		);
		
	if (PL1 > 0.0)
		rv += CosKronTransProb(Z,FL12_TRANS)*PL1;
	return rv;
}

double PL3_pure_kissel(int Z, double E, double PL1, double PL2) {
	double rv;

	rv = CS_Photo_Partial(Z, L3_SHELL, E);

	if (PL1 > 0.0)
		rv += CosKronTransProb(Z,FL13_TRANS)*PL1;

	if (PL2 > 0.0)
		rv += CosKronTransProb(Z,FL23_TRANS)*PL2;


	return rv;
}

double PL3_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2) {
	double rv;

	rv = CS_Photo_Partial(Z,L3_SHELL, E);

	if (PK > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KL3_LINE);

	if (PL1 > 0.0)
		rv += CosKronTransProb(Z,FL13_TRANS)*PL1;

	if (PL2 > 0.0)
		rv += CosKronTransProb(Z,FL23_TRANS)*PL2;

	return  rv;
}

double PL3_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2) {
	double rv;

	rv = CS_Photo_Partial(Z,L3_SHELL, E);

	if (PK > 0.0)
		rv += AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1L3_AUGER)+
		AugerRate(Z,K_L2L3_AUGER)+
		AugerRate(Z,K_L3L1_AUGER)+
		AugerRate(Z,K_L3L2_AUGER)+
		AugerRate(Z,K_L3L3_AUGER)+
		AugerRate(Z,K_L3M1_AUGER)+
		AugerRate(Z,K_L3M2_AUGER)+
		AugerRate(Z,K_L3M3_AUGER)+
		AugerRate(Z,K_L3M4_AUGER)+
		AugerRate(Z,K_L3M5_AUGER)+
		AugerRate(Z,K_L3N1_AUGER)+
		AugerRate(Z,K_L3N2_AUGER)+
		AugerRate(Z,K_L3N3_AUGER)+
		AugerRate(Z,K_L3N4_AUGER)+
		AugerRate(Z,K_L3N5_AUGER)+
		AugerRate(Z,K_L3N6_AUGER)+
		AugerRate(Z,K_L3N7_AUGER)+
		AugerRate(Z,K_L3O1_AUGER)+
		AugerRate(Z,K_L3O2_AUGER)+
		AugerRate(Z,K_L3O3_AUGER)+
		AugerRate(Z,K_L3O4_AUGER)+
		AugerRate(Z,K_L3O5_AUGER)+
		AugerRate(Z,K_L3O6_AUGER)+
		AugerRate(Z,K_L3O7_AUGER)+
		AugerRate(Z,K_L3P1_AUGER)+
		AugerRate(Z,K_L3P2_AUGER)+
		AugerRate(Z,K_L3P3_AUGER)+
		AugerRate(Z,K_L3P4_AUGER)+
		AugerRate(Z,K_L3P5_AUGER)+
		AugerRate(Z,K_L3Q1_AUGER)+
		AugerRate(Z,K_L3Q2_AUGER)+
		AugerRate(Z,K_L3Q3_AUGER)+
		AugerRate(Z,K_M1L3_AUGER)+
		AugerRate(Z,K_M2L3_AUGER)+
		AugerRate(Z,K_M3L3_AUGER)+
		AugerRate(Z,K_M4L3_AUGER)+
		AugerRate(Z,K_M5L3_AUGER)
		);

	if (PL1 > 0.0)
		rv += CosKronTransProb(Z,FL13_TRANS)*PL1;

	if (PL2 > 0.0)
		rv += CosKronTransProb(Z,FL23_TRANS)*PL2;


	return  rv;
}

double PL3_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2) {
	double rv;

	rv = CS_Photo_Partial(Z,L3_SHELL, E);

	if (PK > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KL3_LINE)+
		AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1L3_AUGER)+
		AugerRate(Z,K_L2L3_AUGER)+
		AugerRate(Z,K_L3L1_AUGER)+
		AugerRate(Z,K_L3L2_AUGER)+
		AugerRate(Z,K_L3L3_AUGER)+
		AugerRate(Z,K_L3M1_AUGER)+
		AugerRate(Z,K_L3M2_AUGER)+
		AugerRate(Z,K_L3M3_AUGER)+
		AugerRate(Z,K_L3M4_AUGER)+
		AugerRate(Z,K_L3M5_AUGER)+
		AugerRate(Z,K_L3N1_AUGER)+
		AugerRate(Z,K_L3N2_AUGER)+
		AugerRate(Z,K_L3N3_AUGER)+
		AugerRate(Z,K_L3N4_AUGER)+
		AugerRate(Z,K_L3N5_AUGER)+
		AugerRate(Z,K_L3N6_AUGER)+
		AugerRate(Z,K_L3N7_AUGER)+
		AugerRate(Z,K_L3O1_AUGER)+
		AugerRate(Z,K_L3O2_AUGER)+
		AugerRate(Z,K_L3O3_AUGER)+
		AugerRate(Z,K_L3O4_AUGER)+
		AugerRate(Z,K_L3O5_AUGER)+
		AugerRate(Z,K_L3O6_AUGER)+
		AugerRate(Z,K_L3O7_AUGER)+
		AugerRate(Z,K_L3P1_AUGER)+
		AugerRate(Z,K_L3P2_AUGER)+
		AugerRate(Z,K_L3P3_AUGER)+
		AugerRate(Z,K_L3P4_AUGER)+
		AugerRate(Z,K_L3P5_AUGER)+
		AugerRate(Z,K_L3Q1_AUGER)+
		AugerRate(Z,K_L3Q2_AUGER)+
		AugerRate(Z,K_L3Q3_AUGER)+
		AugerRate(Z,K_M1L3_AUGER)+
		AugerRate(Z,K_M2L3_AUGER)+
		AugerRate(Z,K_M3L3_AUGER)+
		AugerRate(Z,K_M4L3_AUGER)+
		AugerRate(Z,K_M5L3_AUGER)
		);

	if (PL1 > 0.0)
		rv += CosKronTransProb(Z,FL13_TRANS)*PL1;

	if (PL2 > 0.0)
		rv += CosKronTransProb(Z,FL23_TRANS)*PL2;

	return rv;
}

double PM1_pure_kissel(int Z, double E) {
	return CS_Photo_Partial(Z, M1_SHELL, E);
}

double PM1_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3) {
	double rv;

	rv = CS_Photo_Partial(Z, M1_SHELL, E);

	if (PK > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM1_LINE);
	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M1_LINE);
	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M1_LINE);
	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M1_LINE);

	return rv; 
}

double PM1_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3) {
	double rv;

	rv = CS_Photo_Partial(Z, M1_SHELL, E);

	if (PK > 0.0)
		rv += AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1M1_AUGER)+
		AugerRate(Z,K_L2M1_AUGER)+
		AugerRate(Z,K_L3M1_AUGER)+
		AugerRate(Z,K_M1L1_AUGER)+
		AugerRate(Z,K_M1L2_AUGER)+
		AugerRate(Z,K_M1L3_AUGER)+
		AugerRate(Z,K_M1M1_AUGER)+
		AugerRate(Z,K_M1M2_AUGER)+
		AugerRate(Z,K_M1M3_AUGER)+
		AugerRate(Z,K_M1M4_AUGER)+
		AugerRate(Z,K_M1M5_AUGER)+
		AugerRate(Z,K_M2M1_AUGER)+
		AugerRate(Z,K_M3M1_AUGER)+
		AugerRate(Z,K_M4M1_AUGER)+
		AugerRate(Z,K_M5M1_AUGER)
		);
	
	if (PL1 > 0.0)
		rv += AugerYield(Z,L1_SHELL)*PL1*(
		AugerRate(Z,L1_M1M1_AUGER)+
		AugerRate(Z,L1_M1M2_AUGER)+
		AugerRate(Z,L1_M1M3_AUGER)+
		AugerRate(Z,L1_M1M4_AUGER)+
		AugerRate(Z,L1_M1M5_AUGER)+
		AugerRate(Z,L1_M1N1_AUGER)+
		AugerRate(Z,L1_M1N2_AUGER)+
		AugerRate(Z,L1_M1N3_AUGER)+
		AugerRate(Z,L1_M1N4_AUGER)+
		AugerRate(Z,L1_M1N5_AUGER)+
		AugerRate(Z,L1_M1N6_AUGER)+
		AugerRate(Z,L1_M1N7_AUGER)+
		AugerRate(Z,L1_M1O1_AUGER)+
		AugerRate(Z,L1_M1O2_AUGER)+
		AugerRate(Z,L1_M1O3_AUGER)+
		AugerRate(Z,L1_M1O4_AUGER)+
		AugerRate(Z,L1_M1O5_AUGER)+
		AugerRate(Z,L1_M1O6_AUGER)+
		AugerRate(Z,L1_M1O7_AUGER)+
		AugerRate(Z,L1_M1P1_AUGER)+
		AugerRate(Z,L1_M1P2_AUGER)+
		AugerRate(Z,L1_M1P3_AUGER)+
		AugerRate(Z,L1_M1P4_AUGER)+
		AugerRate(Z,L1_M1P5_AUGER)+
		AugerRate(Z,L1_M1Q1_AUGER)+
		AugerRate(Z,L1_M1Q2_AUGER)+
		AugerRate(Z,L1_M1Q3_AUGER)+
		AugerRate(Z,L1_M2M1_AUGER)+
		AugerRate(Z,L1_M3M1_AUGER)+
		AugerRate(Z,L1_M4M1_AUGER)+
		AugerRate(Z,L1_M5M1_AUGER)
		);

	if (PL2 > 0.0) 
		rv += AugerYield(Z,L2_SHELL)*PL2*(
		AugerRate(Z,L2_M1M1_AUGER)+
		AugerRate(Z,L2_M1M2_AUGER)+
		AugerRate(Z,L2_M1M3_AUGER)+
		AugerRate(Z,L2_M1M4_AUGER)+
		AugerRate(Z,L2_M1M5_AUGER)+
		AugerRate(Z,L2_M1N1_AUGER)+
		AugerRate(Z,L2_M1N2_AUGER)+
		AugerRate(Z,L2_M1N3_AUGER)+
		AugerRate(Z,L2_M1N4_AUGER)+
		AugerRate(Z,L2_M1N5_AUGER)+
		AugerRate(Z,L2_M1N6_AUGER)+
		AugerRate(Z,L2_M1N7_AUGER)+
		AugerRate(Z,L2_M1O1_AUGER)+
		AugerRate(Z,L2_M1O2_AUGER)+
		AugerRate(Z,L2_M1O3_AUGER)+
		AugerRate(Z,L2_M1O4_AUGER)+
		AugerRate(Z,L2_M1O5_AUGER)+
		AugerRate(Z,L2_M1O6_AUGER)+
		AugerRate(Z,L2_M1O7_AUGER)+
		AugerRate(Z,L2_M1P1_AUGER)+
		AugerRate(Z,L2_M1P2_AUGER)+
		AugerRate(Z,L2_M1P3_AUGER)+
		AugerRate(Z,L2_M1P4_AUGER)+
		AugerRate(Z,L2_M1P5_AUGER)+
		AugerRate(Z,L2_M1Q1_AUGER)+
		AugerRate(Z,L2_M1Q2_AUGER)+
		AugerRate(Z,L2_M1Q3_AUGER)+
		AugerRate(Z,L2_M2M1_AUGER)+
		AugerRate(Z,L2_M3M1_AUGER)+
		AugerRate(Z,L2_M4M1_AUGER)+
		AugerRate(Z,L2_M5M1_AUGER)
		);
	
	if (PL3 > 0.0)
		rv += AugerYield(Z,L3_SHELL)*PL3*(
		AugerRate(Z,L3_M1M1_AUGER)+
		AugerRate(Z,L3_M1M2_AUGER)+
		AugerRate(Z,L3_M1M3_AUGER)+
		AugerRate(Z,L3_M1M4_AUGER)+
		AugerRate(Z,L3_M1M5_AUGER)+
		AugerRate(Z,L3_M1N1_AUGER)+
		AugerRate(Z,L3_M1N2_AUGER)+
		AugerRate(Z,L3_M1N3_AUGER)+
		AugerRate(Z,L3_M1N4_AUGER)+
		AugerRate(Z,L3_M1N5_AUGER)+
		AugerRate(Z,L3_M1N6_AUGER)+
		AugerRate(Z,L3_M1N7_AUGER)+
		AugerRate(Z,L3_M1O1_AUGER)+
		AugerRate(Z,L3_M1O2_AUGER)+
		AugerRate(Z,L3_M1O3_AUGER)+
		AugerRate(Z,L3_M1O4_AUGER)+
		AugerRate(Z,L3_M1O5_AUGER)+
		AugerRate(Z,L3_M1O6_AUGER)+
		AugerRate(Z,L3_M1O7_AUGER)+
		AugerRate(Z,L3_M1P1_AUGER)+
		AugerRate(Z,L3_M1P2_AUGER)+
		AugerRate(Z,L3_M1P3_AUGER)+
		AugerRate(Z,L3_M1P4_AUGER)+
		AugerRate(Z,L3_M1P5_AUGER)+
		AugerRate(Z,L3_M1Q1_AUGER)+
		AugerRate(Z,L3_M1Q2_AUGER)+
		AugerRate(Z,L3_M1Q3_AUGER)+
		AugerRate(Z,L3_M2M1_AUGER)+
		AugerRate(Z,L3_M3M1_AUGER)+
		AugerRate(Z,L3_M4M1_AUGER)+
		AugerRate(Z,L3_M5M1_AUGER)
		);
	return rv;
}

double PM1_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3) {
	double rv;

	rv = CS_Photo_Partial(Z, M1_SHELL, E);

	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM1_LINE)+
		AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1M1_AUGER)+
		AugerRate(Z,K_L2M1_AUGER)+
		AugerRate(Z,K_L3M1_AUGER)+
		AugerRate(Z,K_M1L1_AUGER)+
		AugerRate(Z,K_M1L2_AUGER)+
		AugerRate(Z,K_M1L3_AUGER)+
		AugerRate(Z,K_M1M1_AUGER)+
		AugerRate(Z,K_M1M2_AUGER)+
		AugerRate(Z,K_M1M3_AUGER)+
		AugerRate(Z,K_M1M4_AUGER)+
		AugerRate(Z,K_M1M5_AUGER)+
		AugerRate(Z,K_M2M1_AUGER)+
		AugerRate(Z,K_M3M1_AUGER)+
		AugerRate(Z,K_M4M1_AUGER)+
		AugerRate(Z,K_M5M1_AUGER)
		);
		

	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M1_LINE)+
		AugerYield(Z,L1_SHELL)*PL1*(
		AugerRate(Z,L1_M1M1_AUGER)+
		AugerRate(Z,L1_M1M2_AUGER)+
		AugerRate(Z,L1_M1M3_AUGER)+
		AugerRate(Z,L1_M1M4_AUGER)+
		AugerRate(Z,L1_M1M5_AUGER)+
		AugerRate(Z,L1_M1N1_AUGER)+
		AugerRate(Z,L1_M1N2_AUGER)+
		AugerRate(Z,L1_M1N3_AUGER)+
		AugerRate(Z,L1_M1N4_AUGER)+
		AugerRate(Z,L1_M1N5_AUGER)+
		AugerRate(Z,L1_M1N6_AUGER)+
		AugerRate(Z,L1_M1N7_AUGER)+
		AugerRate(Z,L1_M1O1_AUGER)+
		AugerRate(Z,L1_M1O2_AUGER)+
		AugerRate(Z,L1_M1O3_AUGER)+
		AugerRate(Z,L1_M1O4_AUGER)+
		AugerRate(Z,L1_M1O5_AUGER)+
		AugerRate(Z,L1_M1O6_AUGER)+
		AugerRate(Z,L1_M1O7_AUGER)+
		AugerRate(Z,L1_M1P1_AUGER)+
		AugerRate(Z,L1_M1P2_AUGER)+
		AugerRate(Z,L1_M1P3_AUGER)+
		AugerRate(Z,L1_M1P4_AUGER)+
		AugerRate(Z,L1_M1P5_AUGER)+
		AugerRate(Z,L1_M1Q1_AUGER)+
		AugerRate(Z,L1_M1Q2_AUGER)+
		AugerRate(Z,L1_M1Q3_AUGER)+
		AugerRate(Z,L1_M2M1_AUGER)+
		AugerRate(Z,L1_M3M1_AUGER)+
		AugerRate(Z,L1_M4M1_AUGER)+
		AugerRate(Z,L1_M5M1_AUGER)
		);
	
	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M1_LINE)+
		AugerYield(Z,L2_SHELL)*PL2*(
		AugerRate(Z,L2_M1M1_AUGER)+
		AugerRate(Z,L2_M1M2_AUGER)+
		AugerRate(Z,L2_M1M3_AUGER)+
		AugerRate(Z,L2_M1M4_AUGER)+
		AugerRate(Z,L2_M1M5_AUGER)+
		AugerRate(Z,L2_M1N1_AUGER)+
		AugerRate(Z,L2_M1N2_AUGER)+
		AugerRate(Z,L2_M1N3_AUGER)+
		AugerRate(Z,L2_M1N4_AUGER)+
		AugerRate(Z,L2_M1N5_AUGER)+
		AugerRate(Z,L2_M1N6_AUGER)+
		AugerRate(Z,L2_M1N7_AUGER)+
		AugerRate(Z,L2_M1O1_AUGER)+
		AugerRate(Z,L2_M1O2_AUGER)+
		AugerRate(Z,L2_M1O3_AUGER)+
		AugerRate(Z,L2_M1O4_AUGER)+
		AugerRate(Z,L2_M1O5_AUGER)+
		AugerRate(Z,L2_M1O6_AUGER)+
		AugerRate(Z,L2_M1O7_AUGER)+
		AugerRate(Z,L2_M1P1_AUGER)+
		AugerRate(Z,L2_M1P2_AUGER)+
		AugerRate(Z,L2_M1P3_AUGER)+
		AugerRate(Z,L2_M1P4_AUGER)+
		AugerRate(Z,L2_M1P5_AUGER)+
		AugerRate(Z,L2_M1Q1_AUGER)+
		AugerRate(Z,L2_M1Q2_AUGER)+
		AugerRate(Z,L2_M1Q3_AUGER)+
		AugerRate(Z,L2_M2M1_AUGER)+
		AugerRate(Z,L2_M3M1_AUGER)+
		AugerRate(Z,L2_M4M1_AUGER)+
		AugerRate(Z,L2_M5M1_AUGER)
		);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M1_LINE)+
		AugerYield(Z,L3_SHELL)*PL3*(
		AugerRate(Z,L3_M1M1_AUGER)+
		AugerRate(Z,L3_M1M2_AUGER)+
		AugerRate(Z,L3_M1M3_AUGER)+
		AugerRate(Z,L3_M1M4_AUGER)+
		AugerRate(Z,L3_M1M5_AUGER)+
		AugerRate(Z,L3_M1N1_AUGER)+
		AugerRate(Z,L3_M1N2_AUGER)+
		AugerRate(Z,L3_M1N3_AUGER)+
		AugerRate(Z,L3_M1N4_AUGER)+
		AugerRate(Z,L3_M1N5_AUGER)+
		AugerRate(Z,L3_M1N6_AUGER)+
		AugerRate(Z,L3_M1N7_AUGER)+
		AugerRate(Z,L3_M1O1_AUGER)+
		AugerRate(Z,L3_M1O2_AUGER)+
		AugerRate(Z,L3_M1O3_AUGER)+
		AugerRate(Z,L3_M1O4_AUGER)+
		AugerRate(Z,L3_M1O5_AUGER)+
		AugerRate(Z,L3_M1O6_AUGER)+
		AugerRate(Z,L3_M1O7_AUGER)+
		AugerRate(Z,L3_M1P1_AUGER)+
		AugerRate(Z,L3_M1P2_AUGER)+
		AugerRate(Z,L3_M1P3_AUGER)+
		AugerRate(Z,L3_M1P4_AUGER)+
		AugerRate(Z,L3_M1P5_AUGER)+
		AugerRate(Z,L3_M1Q1_AUGER)+
		AugerRate(Z,L3_M1Q2_AUGER)+
		AugerRate(Z,L3_M1Q3_AUGER)+
		AugerRate(Z,L3_M2M1_AUGER)+
		AugerRate(Z,L3_M3M1_AUGER)+
		AugerRate(Z,L3_M4M1_AUGER)+
		AugerRate(Z,L3_M5M1_AUGER)
		);


	return rv;
}


double PM2_pure_kissel(int Z, double E, double PM1) {
	double rv;

	rv = CS_Photo_Partial(Z, M2_SHELL, E);
	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM12_TRANS)*PM1;
		
	return rv; 
}

double PM2_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1) {
	double rv;

	rv = CS_Photo_Partial(Z, M2_SHELL, E);

	if (PK > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM2_LINE);

	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M2_LINE);

	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M2_LINE);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M2_LINE);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM12_TRANS)*PM1;

	return rv;
}

double PM2_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1) {
	double rv;

	rv = CS_Photo_Partial(Z, M2_SHELL, E);

	if (PK > 0.0)
		rv += AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1M2_AUGER)+
		AugerRate(Z,K_L2M2_AUGER)+
		AugerRate(Z,K_L3M2_AUGER)+
		AugerRate(Z,K_M1M2_AUGER)+
		AugerRate(Z,K_M2L1_AUGER)+
		AugerRate(Z,K_M2L2_AUGER)+
		AugerRate(Z,K_M2L3_AUGER)+
		AugerRate(Z,K_M2M1_AUGER)+
		AugerRate(Z,K_M2M2_AUGER)+
		AugerRate(Z,K_M2M3_AUGER)+
		AugerRate(Z,K_M2M4_AUGER)+
		AugerRate(Z,K_M2M5_AUGER)+
		AugerRate(Z,K_M2N1_AUGER)+
		AugerRate(Z,K_M2N2_AUGER)+
		AugerRate(Z,K_M2N3_AUGER)+
		AugerRate(Z,K_M2N4_AUGER)+
		AugerRate(Z,K_M2N5_AUGER)+
		AugerRate(Z,K_M2N6_AUGER)+
		AugerRate(Z,K_M2N7_AUGER)+
		AugerRate(Z,K_M2O1_AUGER)+
		AugerRate(Z,K_M2O2_AUGER)+
		AugerRate(Z,K_M2O3_AUGER)+
		AugerRate(Z,K_M2O4_AUGER)+
		AugerRate(Z,K_M2O5_AUGER)+
		AugerRate(Z,K_M2O6_AUGER)+
		AugerRate(Z,K_M2O7_AUGER)+
		AugerRate(Z,K_M2P1_AUGER)+
		AugerRate(Z,K_M2P2_AUGER)+
		AugerRate(Z,K_M2P3_AUGER)+
		AugerRate(Z,K_M2P4_AUGER)+
		AugerRate(Z,K_M2P5_AUGER)+
		AugerRate(Z,K_M2Q1_AUGER)+
		AugerRate(Z,K_M2Q2_AUGER)+
		AugerRate(Z,K_M2Q3_AUGER)+
		AugerRate(Z,K_M3M2_AUGER)+
		AugerRate(Z,K_M4M2_AUGER)+
		AugerRate(Z,K_M5M2_AUGER)
		);
	if (PL1 > 0.0)
		rv += AugerYield(Z,L1_SHELL)*PL1*(
		AugerRate(Z,L1_M1M2_AUGER)+
		AugerRate(Z,L1_M2M1_AUGER)+
		AugerRate(Z,L1_M2M2_AUGER)+
		AugerRate(Z,L1_M2M3_AUGER)+
		AugerRate(Z,L1_M2M4_AUGER)+
		AugerRate(Z,L1_M2M5_AUGER)+
		AugerRate(Z,L1_M2N1_AUGER)+
		AugerRate(Z,L1_M2N2_AUGER)+
		AugerRate(Z,L1_M2N3_AUGER)+
		AugerRate(Z,L1_M2N4_AUGER)+
		AugerRate(Z,L1_M2N5_AUGER)+
		AugerRate(Z,L1_M2N6_AUGER)+
		AugerRate(Z,L1_M2N7_AUGER)+
		AugerRate(Z,L1_M2O1_AUGER)+
		AugerRate(Z,L1_M2O2_AUGER)+
		AugerRate(Z,L1_M2O3_AUGER)+
		AugerRate(Z,L1_M2O4_AUGER)+
		AugerRate(Z,L1_M2O5_AUGER)+
		AugerRate(Z,L1_M2O6_AUGER)+
		AugerRate(Z,L1_M2O7_AUGER)+
		AugerRate(Z,L1_M2P1_AUGER)+
		AugerRate(Z,L1_M2P2_AUGER)+
		AugerRate(Z,L1_M2P3_AUGER)+
		AugerRate(Z,L1_M2P4_AUGER)+
		AugerRate(Z,L1_M2P5_AUGER)+
		AugerRate(Z,L1_M2Q1_AUGER)+
		AugerRate(Z,L1_M2Q2_AUGER)+
		AugerRate(Z,L1_M2Q3_AUGER)+
		AugerRate(Z,L1_M3M2_AUGER)+
		AugerRate(Z,L1_M4M2_AUGER)+
		AugerRate(Z,L1_M5M2_AUGER)
		);

	if (PL2 > 0.0)
		rv += AugerYield(Z,L2_SHELL)*PL2*(
		AugerRate(Z,L2_M1M2_AUGER)+
		AugerRate(Z,L2_M2M1_AUGER)+
		AugerRate(Z,L2_M2M2_AUGER)+
		AugerRate(Z,L2_M2M3_AUGER)+
		AugerRate(Z,L2_M2M4_AUGER)+
		AugerRate(Z,L2_M2M5_AUGER)+
		AugerRate(Z,L2_M2N1_AUGER)+
		AugerRate(Z,L2_M2N2_AUGER)+
		AugerRate(Z,L2_M2N3_AUGER)+
		AugerRate(Z,L2_M2N4_AUGER)+
		AugerRate(Z,L2_M2N5_AUGER)+
		AugerRate(Z,L2_M2N6_AUGER)+
		AugerRate(Z,L2_M2N7_AUGER)+
		AugerRate(Z,L2_M2O1_AUGER)+
		AugerRate(Z,L2_M2O2_AUGER)+
		AugerRate(Z,L2_M2O3_AUGER)+
		AugerRate(Z,L2_M2O4_AUGER)+
		AugerRate(Z,L2_M2O5_AUGER)+
		AugerRate(Z,L2_M2O6_AUGER)+
		AugerRate(Z,L2_M2O7_AUGER)+
		AugerRate(Z,L2_M2P1_AUGER)+
		AugerRate(Z,L2_M2P2_AUGER)+
		AugerRate(Z,L2_M2P3_AUGER)+
		AugerRate(Z,L2_M2P4_AUGER)+
		AugerRate(Z,L2_M2P5_AUGER)+
		AugerRate(Z,L2_M2Q1_AUGER)+
		AugerRate(Z,L2_M2Q2_AUGER)+
		AugerRate(Z,L2_M2Q3_AUGER)+
		AugerRate(Z,L2_M3M2_AUGER)+
		AugerRate(Z,L2_M4M2_AUGER)+
		AugerRate(Z,L2_M5M2_AUGER)
		);
	if (PL3 > 0.0)
		rv += AugerYield(Z,L3_SHELL)*PL3*(
		AugerRate(Z,L3_M1M2_AUGER)+
		AugerRate(Z,L3_M2M1_AUGER)+
		AugerRate(Z,L3_M2M2_AUGER)+
		AugerRate(Z,L3_M2M3_AUGER)+
		AugerRate(Z,L3_M2M4_AUGER)+
		AugerRate(Z,L3_M2M5_AUGER)+
		AugerRate(Z,L3_M2N1_AUGER)+
		AugerRate(Z,L3_M2N2_AUGER)+
		AugerRate(Z,L3_M2N3_AUGER)+
		AugerRate(Z,L3_M2N4_AUGER)+
		AugerRate(Z,L3_M2N5_AUGER)+
		AugerRate(Z,L3_M2N6_AUGER)+
		AugerRate(Z,L3_M2N7_AUGER)+
		AugerRate(Z,L3_M2O1_AUGER)+
		AugerRate(Z,L3_M2O2_AUGER)+
		AugerRate(Z,L3_M2O3_AUGER)+
		AugerRate(Z,L3_M2O4_AUGER)+
		AugerRate(Z,L3_M2O5_AUGER)+
		AugerRate(Z,L3_M2O6_AUGER)+
		AugerRate(Z,L3_M2O7_AUGER)+
		AugerRate(Z,L3_M2P1_AUGER)+
		AugerRate(Z,L3_M2P2_AUGER)+
		AugerRate(Z,L3_M2P3_AUGER)+
		AugerRate(Z,L3_M2P4_AUGER)+
		AugerRate(Z,L3_M2P5_AUGER)+
		AugerRate(Z,L3_M2Q1_AUGER)+
		AugerRate(Z,L3_M2Q2_AUGER)+
		AugerRate(Z,L3_M2Q3_AUGER)+
		AugerRate(Z,L3_M3M2_AUGER)+
		AugerRate(Z,L3_M4M2_AUGER)+
		AugerRate(Z,L3_M5M2_AUGER)
		);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM12_TRANS)*PM1;

	return rv;
}

double PM2_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1) {
	double rv;

	rv = CS_Photo_Partial(Z, M2_SHELL, E);

	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM2_LINE)+
		AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1M2_AUGER)+
		AugerRate(Z,K_L2M2_AUGER)+
		AugerRate(Z,K_L3M2_AUGER)+
		AugerRate(Z,K_M1M2_AUGER)+
		AugerRate(Z,K_M2L1_AUGER)+
		AugerRate(Z,K_M2L2_AUGER)+
		AugerRate(Z,K_M2L3_AUGER)+
		AugerRate(Z,K_M2M1_AUGER)+
		AugerRate(Z,K_M2M2_AUGER)+
		AugerRate(Z,K_M2M3_AUGER)+
		AugerRate(Z,K_M2M4_AUGER)+
		AugerRate(Z,K_M2M5_AUGER)+
		AugerRate(Z,K_M2N1_AUGER)+
		AugerRate(Z,K_M2N2_AUGER)+
		AugerRate(Z,K_M2N3_AUGER)+
		AugerRate(Z,K_M2N4_AUGER)+
		AugerRate(Z,K_M2N5_AUGER)+
		AugerRate(Z,K_M2N6_AUGER)+
		AugerRate(Z,K_M2N7_AUGER)+
		AugerRate(Z,K_M2O1_AUGER)+
		AugerRate(Z,K_M2O2_AUGER)+
		AugerRate(Z,K_M2O3_AUGER)+
		AugerRate(Z,K_M2O4_AUGER)+
		AugerRate(Z,K_M2O5_AUGER)+
		AugerRate(Z,K_M2O6_AUGER)+
		AugerRate(Z,K_M2O7_AUGER)+
		AugerRate(Z,K_M2P1_AUGER)+
		AugerRate(Z,K_M2P2_AUGER)+
		AugerRate(Z,K_M2P3_AUGER)+
		AugerRate(Z,K_M2P4_AUGER)+
		AugerRate(Z,K_M2P5_AUGER)+
		AugerRate(Z,K_M2Q1_AUGER)+
		AugerRate(Z,K_M2Q2_AUGER)+
		AugerRate(Z,K_M2Q3_AUGER)+
		AugerRate(Z,K_M3M2_AUGER)+
		AugerRate(Z,K_M4M2_AUGER)+
		AugerRate(Z,K_M5M2_AUGER)
		);

	if (PL1 > 0.0) 
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M2_LINE)+
		AugerYield(Z,L1_SHELL)*PL1*(
		AugerRate(Z,L1_M1M2_AUGER)+
		AugerRate(Z,L1_M2M1_AUGER)+
		AugerRate(Z,L1_M2M2_AUGER)+
		AugerRate(Z,L1_M2M3_AUGER)+
		AugerRate(Z,L1_M2M4_AUGER)+
		AugerRate(Z,L1_M2M5_AUGER)+
		AugerRate(Z,L1_M2N1_AUGER)+
		AugerRate(Z,L1_M2N2_AUGER)+
		AugerRate(Z,L1_M2N3_AUGER)+
		AugerRate(Z,L1_M2N4_AUGER)+
		AugerRate(Z,L1_M2N5_AUGER)+
		AugerRate(Z,L1_M2N6_AUGER)+
		AugerRate(Z,L1_M2N7_AUGER)+
		AugerRate(Z,L1_M2O1_AUGER)+
		AugerRate(Z,L1_M2O2_AUGER)+
		AugerRate(Z,L1_M2O3_AUGER)+
		AugerRate(Z,L1_M2O4_AUGER)+
		AugerRate(Z,L1_M2O5_AUGER)+
		AugerRate(Z,L1_M2O6_AUGER)+
		AugerRate(Z,L1_M2O7_AUGER)+
		AugerRate(Z,L1_M2P1_AUGER)+
		AugerRate(Z,L1_M2P2_AUGER)+
		AugerRate(Z,L1_M2P3_AUGER)+
		AugerRate(Z,L1_M2P4_AUGER)+
		AugerRate(Z,L1_M2P5_AUGER)+
		AugerRate(Z,L1_M2Q1_AUGER)+
		AugerRate(Z,L1_M2Q2_AUGER)+
		AugerRate(Z,L1_M2Q3_AUGER)+
		AugerRate(Z,L1_M3M2_AUGER)+
		AugerRate(Z,L1_M4M2_AUGER)+
		AugerRate(Z,L1_M5M2_AUGER)
		);
	
	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M2_LINE)+
		AugerYield(Z,L2_SHELL)*PL2*(
		AugerRate(Z,L2_M1M2_AUGER)+
		AugerRate(Z,L2_M2M1_AUGER)+
		AugerRate(Z,L2_M2M2_AUGER)+
		AugerRate(Z,L2_M2M3_AUGER)+
		AugerRate(Z,L2_M2M4_AUGER)+
		AugerRate(Z,L2_M2M5_AUGER)+
		AugerRate(Z,L2_M2N1_AUGER)+
		AugerRate(Z,L2_M2N2_AUGER)+
		AugerRate(Z,L2_M2N3_AUGER)+
		AugerRate(Z,L2_M2N4_AUGER)+
		AugerRate(Z,L2_M2N5_AUGER)+
		AugerRate(Z,L2_M2N6_AUGER)+
		AugerRate(Z,L2_M2N7_AUGER)+
		AugerRate(Z,L2_M2O1_AUGER)+
		AugerRate(Z,L2_M2O2_AUGER)+
		AugerRate(Z,L2_M2O3_AUGER)+
		AugerRate(Z,L2_M2O4_AUGER)+
		AugerRate(Z,L2_M2O5_AUGER)+
		AugerRate(Z,L2_M2O6_AUGER)+
		AugerRate(Z,L2_M2O7_AUGER)+
		AugerRate(Z,L2_M2P1_AUGER)+
		AugerRate(Z,L2_M2P2_AUGER)+
		AugerRate(Z,L2_M2P3_AUGER)+
		AugerRate(Z,L2_M2P4_AUGER)+
		AugerRate(Z,L2_M2P5_AUGER)+
		AugerRate(Z,L2_M2Q1_AUGER)+
		AugerRate(Z,L2_M2Q2_AUGER)+
		AugerRate(Z,L2_M2Q3_AUGER)+
		AugerRate(Z,L2_M3M2_AUGER)+
		AugerRate(Z,L2_M4M2_AUGER)+
		AugerRate(Z,L2_M5M2_AUGER)
		);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M2_LINE) +
		AugerYield(Z,L3_SHELL)*PL3*(
		AugerRate(Z,L3_M1M2_AUGER)+
		AugerRate(Z,L3_M2M1_AUGER)+
		AugerRate(Z,L3_M2M2_AUGER)+
		AugerRate(Z,L3_M2M3_AUGER)+
		AugerRate(Z,L3_M2M4_AUGER)+
		AugerRate(Z,L3_M2M5_AUGER)+
		AugerRate(Z,L3_M2N1_AUGER)+
		AugerRate(Z,L3_M2N2_AUGER)+
		AugerRate(Z,L3_M2N3_AUGER)+
		AugerRate(Z,L3_M2N4_AUGER)+
		AugerRate(Z,L3_M2N5_AUGER)+
		AugerRate(Z,L3_M2N6_AUGER)+
		AugerRate(Z,L3_M2N7_AUGER)+
		AugerRate(Z,L3_M2O1_AUGER)+
		AugerRate(Z,L3_M2O2_AUGER)+
		AugerRate(Z,L3_M2O3_AUGER)+
		AugerRate(Z,L3_M2O4_AUGER)+
		AugerRate(Z,L3_M2O5_AUGER)+
		AugerRate(Z,L3_M2O6_AUGER)+
		AugerRate(Z,L3_M2O7_AUGER)+
		AugerRate(Z,L3_M2P1_AUGER)+
		AugerRate(Z,L3_M2P2_AUGER)+
		AugerRate(Z,L3_M2P3_AUGER)+
		AugerRate(Z,L3_M2P4_AUGER)+
		AugerRate(Z,L3_M2P5_AUGER)+
		AugerRate(Z,L3_M2Q1_AUGER)+
		AugerRate(Z,L3_M2Q2_AUGER)+
		AugerRate(Z,L3_M2Q3_AUGER)+
		AugerRate(Z,L3_M3M2_AUGER)+
		AugerRate(Z,L3_M4M2_AUGER)+
		AugerRate(Z,L3_M5M2_AUGER)
		);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM12_TRANS)*PM1;

	return rv;
}

double PM3_pure_kissel(int Z, double E, double PM1, double PM2) {
	double rv;

	rv = CS_Photo_Partial(Z, M3_SHELL, E);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM13_TRANS)*PM1;

	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM23_TRANS)*PM2;

	return rv;
}

double PM3_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2) {
	double rv;

	rv = CS_Photo_Partial(Z, M3_SHELL, E);

	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM3_LINE);

	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M3_LINE);

	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M3_LINE);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M3_LINE);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM13_TRANS)*PM1;
	
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM23_TRANS)*PM2;

	return rv;
}

double PM3_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2) {
	double rv;

	rv = CS_Photo_Partial(Z, M3_SHELL, E);

	if (PK > 0.0)
		rv += AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1M3_AUGER)+
		AugerRate(Z,K_L2M3_AUGER)+
		AugerRate(Z,K_L3M3_AUGER)+
		AugerRate(Z,K_M1M3_AUGER)+
		AugerRate(Z,K_M2M3_AUGER)+
		AugerRate(Z,K_M3L1_AUGER)+
		AugerRate(Z,K_M3L2_AUGER)+
		AugerRate(Z,K_M3L3_AUGER)+
		AugerRate(Z,K_M3M1_AUGER)+
		AugerRate(Z,K_M3M2_AUGER)+
		AugerRate(Z,K_M3M3_AUGER)+
		AugerRate(Z,K_M3M4_AUGER)+
		AugerRate(Z,K_M3M5_AUGER)+
		AugerRate(Z,K_M3N1_AUGER)+
		AugerRate(Z,K_M3N2_AUGER)+
		AugerRate(Z,K_M3N3_AUGER)+
		AugerRate(Z,K_M3N4_AUGER)+
		AugerRate(Z,K_M3N5_AUGER)+
		AugerRate(Z,K_M3N6_AUGER)+
		AugerRate(Z,K_M3N7_AUGER)+
		AugerRate(Z,K_M3O1_AUGER)+
		AugerRate(Z,K_M3O2_AUGER)+
		AugerRate(Z,K_M3O3_AUGER)+
		AugerRate(Z,K_M3O4_AUGER)+
		AugerRate(Z,K_M3O5_AUGER)+
		AugerRate(Z,K_M3O6_AUGER)+
		AugerRate(Z,K_M3O7_AUGER)+
		AugerRate(Z,K_M3P1_AUGER)+
		AugerRate(Z,K_M3P2_AUGER)+
		AugerRate(Z,K_M3P3_AUGER)+
		AugerRate(Z,K_M3P4_AUGER)+
		AugerRate(Z,K_M3P5_AUGER)+
		AugerRate(Z,K_M3Q1_AUGER)+
		AugerRate(Z,K_M3Q2_AUGER)+
		AugerRate(Z,K_M3Q3_AUGER)+
		AugerRate(Z,K_M4M3_AUGER)+
		AugerRate(Z,K_M5M3_AUGER)
		);
	if (PL1 > 0.0)
		rv += AugerYield(Z,L1_SHELL)*PL1*(
		AugerRate(Z,L1_M1M3_AUGER)+
		AugerRate(Z,L1_M2M3_AUGER)+
		AugerRate(Z,L1_M3M1_AUGER)+
		AugerRate(Z,L1_M3M2_AUGER)+
		AugerRate(Z,L1_M3M3_AUGER)+
		AugerRate(Z,L1_M3M4_AUGER)+
		AugerRate(Z,L1_M3M5_AUGER)+
		AugerRate(Z,L1_M3N1_AUGER)+
		AugerRate(Z,L1_M3N2_AUGER)+
		AugerRate(Z,L1_M3N3_AUGER)+
		AugerRate(Z,L1_M3N4_AUGER)+
		AugerRate(Z,L1_M3N5_AUGER)+
		AugerRate(Z,L1_M3N6_AUGER)+
		AugerRate(Z,L1_M3N7_AUGER)+
		AugerRate(Z,L1_M3O1_AUGER)+
		AugerRate(Z,L1_M3O2_AUGER)+
		AugerRate(Z,L1_M3O3_AUGER)+
		AugerRate(Z,L1_M3O4_AUGER)+
		AugerRate(Z,L1_M3O5_AUGER)+
		AugerRate(Z,L1_M3O6_AUGER)+
		AugerRate(Z,L1_M3O7_AUGER)+
		AugerRate(Z,L1_M3P1_AUGER)+
		AugerRate(Z,L1_M3P2_AUGER)+
		AugerRate(Z,L1_M3P3_AUGER)+
		AugerRate(Z,L1_M3P4_AUGER)+
		AugerRate(Z,L1_M3P5_AUGER)+
		AugerRate(Z,L1_M3Q1_AUGER)+
		AugerRate(Z,L1_M3Q2_AUGER)+
		AugerRate(Z,L1_M3Q3_AUGER)+
		AugerRate(Z,L1_M4M3_AUGER)+
		AugerRate(Z,L1_M5M3_AUGER)
		);
	if (PL2 > 0.0)
		rv += AugerYield(Z,L2_SHELL)*PL2*(
		AugerRate(Z,L2_M1M3_AUGER)+
		AugerRate(Z,L2_M2M3_AUGER)+
		AugerRate(Z,L2_M3M1_AUGER)+
		AugerRate(Z,L2_M3M2_AUGER)+
		AugerRate(Z,L2_M3M3_AUGER)+
		AugerRate(Z,L2_M3M4_AUGER)+
		AugerRate(Z,L2_M3M5_AUGER)+
		AugerRate(Z,L2_M3N1_AUGER)+
		AugerRate(Z,L2_M3N2_AUGER)+
		AugerRate(Z,L2_M3N3_AUGER)+
		AugerRate(Z,L2_M3N4_AUGER)+
		AugerRate(Z,L2_M3N5_AUGER)+
		AugerRate(Z,L2_M3N6_AUGER)+
		AugerRate(Z,L2_M3N7_AUGER)+
		AugerRate(Z,L2_M3O1_AUGER)+
		AugerRate(Z,L2_M3O2_AUGER)+
		AugerRate(Z,L2_M3O3_AUGER)+
		AugerRate(Z,L2_M3O4_AUGER)+
		AugerRate(Z,L2_M3O5_AUGER)+
		AugerRate(Z,L2_M3O6_AUGER)+
		AugerRate(Z,L2_M3O7_AUGER)+
		AugerRate(Z,L2_M3P1_AUGER)+
		AugerRate(Z,L2_M3P2_AUGER)+
		AugerRate(Z,L2_M3P3_AUGER)+
		AugerRate(Z,L2_M3P4_AUGER)+
		AugerRate(Z,L2_M3P5_AUGER)+
		AugerRate(Z,L2_M3Q1_AUGER)+
		AugerRate(Z,L2_M3Q2_AUGER)+
		AugerRate(Z,L2_M3Q3_AUGER)+
		AugerRate(Z,L2_M4M3_AUGER)+
		AugerRate(Z,L2_M5M3_AUGER)
		);
	if (PL3 > 0.0) 
		rv += AugerYield(Z,L3_SHELL)*PL3*(
		AugerRate(Z,L3_M1M3_AUGER)+
		AugerRate(Z,L3_M2M3_AUGER)+
		AugerRate(Z,L3_M3M1_AUGER)+
		AugerRate(Z,L3_M3M2_AUGER)+
		AugerRate(Z,L3_M3M3_AUGER)+
		AugerRate(Z,L3_M3M4_AUGER)+
		AugerRate(Z,L3_M3M5_AUGER)+
		AugerRate(Z,L3_M3N1_AUGER)+
		AugerRate(Z,L3_M3N2_AUGER)+
		AugerRate(Z,L3_M3N3_AUGER)+
		AugerRate(Z,L3_M3N4_AUGER)+
		AugerRate(Z,L3_M3N5_AUGER)+
		AugerRate(Z,L3_M3N6_AUGER)+
		AugerRate(Z,L3_M3N7_AUGER)+
		AugerRate(Z,L3_M3O1_AUGER)+
		AugerRate(Z,L3_M3O2_AUGER)+
		AugerRate(Z,L3_M3O3_AUGER)+
		AugerRate(Z,L3_M3O4_AUGER)+
		AugerRate(Z,L3_M3O5_AUGER)+
		AugerRate(Z,L3_M3O6_AUGER)+
		AugerRate(Z,L3_M3O7_AUGER)+
		AugerRate(Z,L3_M3P1_AUGER)+
		AugerRate(Z,L3_M3P2_AUGER)+
		AugerRate(Z,L3_M3P3_AUGER)+
		AugerRate(Z,L3_M3P4_AUGER)+
		AugerRate(Z,L3_M3P5_AUGER)+
		AugerRate(Z,L3_M3Q1_AUGER)+
		AugerRate(Z,L3_M3Q2_AUGER)+
		AugerRate(Z,L3_M3Q3_AUGER)+
		AugerRate(Z,L3_M4M3_AUGER)+
		AugerRate(Z,L3_M5M3_AUGER)
		);
	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM13_TRANS)*PM1;
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM23_TRANS)*PM2;

	return rv;
}

double PM3_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2) {
	double rv;

	rv = CS_Photo_Partial(Z, M3_SHELL, E);

	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM3_LINE)+
		(1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1M3_AUGER)+
		AugerRate(Z,K_L2M3_AUGER)+
		AugerRate(Z,K_L3M3_AUGER)+
		AugerRate(Z,K_M1M3_AUGER)+
		AugerRate(Z,K_M2M3_AUGER)+
		AugerRate(Z,K_M3L1_AUGER)+
		AugerRate(Z,K_M3L2_AUGER)+
		AugerRate(Z,K_M3L3_AUGER)+
		AugerRate(Z,K_M3M1_AUGER)+
		AugerRate(Z,K_M3M2_AUGER)+
		AugerRate(Z,K_M3M3_AUGER)+
		AugerRate(Z,K_M3M4_AUGER)+
		AugerRate(Z,K_M3M5_AUGER)+
		AugerRate(Z,K_M3N1_AUGER)+
		AugerRate(Z,K_M3N2_AUGER)+
		AugerRate(Z,K_M3N3_AUGER)+
		AugerRate(Z,K_M3N4_AUGER)+
		AugerRate(Z,K_M3N5_AUGER)+
		AugerRate(Z,K_M3N6_AUGER)+
		AugerRate(Z,K_M3N7_AUGER)+
		AugerRate(Z,K_M3O1_AUGER)+
		AugerRate(Z,K_M3O2_AUGER)+
		AugerRate(Z,K_M3O3_AUGER)+
		AugerRate(Z,K_M3O4_AUGER)+
		AugerRate(Z,K_M3O5_AUGER)+
		AugerRate(Z,K_M3O6_AUGER)+
		AugerRate(Z,K_M3O7_AUGER)+
		AugerRate(Z,K_M3P1_AUGER)+
		AugerRate(Z,K_M3P2_AUGER)+
		AugerRate(Z,K_M3P3_AUGER)+
		AugerRate(Z,K_M3P4_AUGER)+
		AugerRate(Z,K_M3P5_AUGER)+
		AugerRate(Z,K_M3Q1_AUGER)+
		AugerRate(Z,K_M3Q2_AUGER)+
		AugerRate(Z,K_M3Q3_AUGER)+
		AugerRate(Z,K_M4M3_AUGER)+
		AugerRate(Z,K_M5M3_AUGER)
		);

	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M3_LINE)+
		AugerYield(Z,L1_SHELL)*PL1*(
		AugerRate(Z,L1_M1M3_AUGER)+
		AugerRate(Z,L1_M2M3_AUGER)+
		AugerRate(Z,L1_M3M1_AUGER)+
		AugerRate(Z,L1_M3M2_AUGER)+
		AugerRate(Z,L1_M3M3_AUGER)+
		AugerRate(Z,L1_M3M4_AUGER)+
		AugerRate(Z,L1_M3M5_AUGER)+
		AugerRate(Z,L1_M3N1_AUGER)+
		AugerRate(Z,L1_M3N2_AUGER)+
		AugerRate(Z,L1_M3N3_AUGER)+
		AugerRate(Z,L1_M3N4_AUGER)+
		AugerRate(Z,L1_M3N5_AUGER)+
		AugerRate(Z,L1_M3N6_AUGER)+
		AugerRate(Z,L1_M3N7_AUGER)+
		AugerRate(Z,L1_M3O1_AUGER)+
		AugerRate(Z,L1_M3O2_AUGER)+
		AugerRate(Z,L1_M3O3_AUGER)+
		AugerRate(Z,L1_M3O4_AUGER)+
		AugerRate(Z,L1_M3O5_AUGER)+
		AugerRate(Z,L1_M3O6_AUGER)+
		AugerRate(Z,L1_M3O7_AUGER)+
		AugerRate(Z,L1_M3P1_AUGER)+
		AugerRate(Z,L1_M3P2_AUGER)+
		AugerRate(Z,L1_M3P3_AUGER)+
		AugerRate(Z,L1_M3P4_AUGER)+
		AugerRate(Z,L1_M3P5_AUGER)+
		AugerRate(Z,L1_M3Q1_AUGER)+
		AugerRate(Z,L1_M3Q2_AUGER)+
		AugerRate(Z,L1_M3Q3_AUGER)+
		AugerRate(Z,L1_M4M3_AUGER)+
		AugerRate(Z,L1_M5M3_AUGER)
		);

	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M3_LINE)+
		AugerYield(Z,L2_SHELL)*PL2*(
		AugerRate(Z,L2_M1M3_AUGER)+
		AugerRate(Z,L2_M2M3_AUGER)+
		AugerRate(Z,L2_M3M1_AUGER)+
		AugerRate(Z,L2_M3M2_AUGER)+
		AugerRate(Z,L2_M3M3_AUGER)+
		AugerRate(Z,L2_M3M4_AUGER)+
		AugerRate(Z,L2_M3M5_AUGER)+
		AugerRate(Z,L2_M3N1_AUGER)+
		AugerRate(Z,L2_M3N2_AUGER)+
		AugerRate(Z,L2_M3N3_AUGER)+
		AugerRate(Z,L2_M3N4_AUGER)+
		AugerRate(Z,L2_M3N5_AUGER)+
		AugerRate(Z,L2_M3N6_AUGER)+
		AugerRate(Z,L2_M3N7_AUGER)+
		AugerRate(Z,L2_M3O1_AUGER)+
		AugerRate(Z,L2_M3O2_AUGER)+
		AugerRate(Z,L2_M3O3_AUGER)+
		AugerRate(Z,L2_M3O4_AUGER)+
		AugerRate(Z,L2_M3O5_AUGER)+
		AugerRate(Z,L2_M3O6_AUGER)+
		AugerRate(Z,L2_M3O7_AUGER)+
		AugerRate(Z,L2_M3P1_AUGER)+
		AugerRate(Z,L2_M3P2_AUGER)+
		AugerRate(Z,L2_M3P3_AUGER)+
		AugerRate(Z,L2_M3P4_AUGER)+
		AugerRate(Z,L2_M3P5_AUGER)+
		AugerRate(Z,L2_M3Q1_AUGER)+
		AugerRate(Z,L2_M3Q2_AUGER)+
		AugerRate(Z,L2_M3Q3_AUGER)+
		AugerRate(Z,L2_M4M3_AUGER)+
		AugerRate(Z,L2_M5M3_AUGER)
		);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M3_LINE)+
		AugerYield(Z,L3_SHELL)*PL3*(
		AugerRate(Z,L3_M1M3_AUGER)+
		AugerRate(Z,L3_M2M3_AUGER)+
		AugerRate(Z,L3_M3M1_AUGER)+
		AugerRate(Z,L3_M3M2_AUGER)+
		AugerRate(Z,L3_M3M3_AUGER)+
		AugerRate(Z,L3_M3M4_AUGER)+
		AugerRate(Z,L3_M3M5_AUGER)+
		AugerRate(Z,L3_M3N1_AUGER)+
		AugerRate(Z,L3_M3N2_AUGER)+
		AugerRate(Z,L3_M3N3_AUGER)+
		AugerRate(Z,L3_M3N4_AUGER)+
		AugerRate(Z,L3_M3N5_AUGER)+
		AugerRate(Z,L3_M3N6_AUGER)+
		AugerRate(Z,L3_M3N7_AUGER)+
		AugerRate(Z,L3_M3O1_AUGER)+
		AugerRate(Z,L3_M3O2_AUGER)+
		AugerRate(Z,L3_M3O3_AUGER)+
		AugerRate(Z,L3_M3O4_AUGER)+
		AugerRate(Z,L3_M3O5_AUGER)+
		AugerRate(Z,L3_M3O6_AUGER)+
		AugerRate(Z,L3_M3O7_AUGER)+
		AugerRate(Z,L3_M3P1_AUGER)+
		AugerRate(Z,L3_M3P2_AUGER)+
		AugerRate(Z,L3_M3P3_AUGER)+
		AugerRate(Z,L3_M3P4_AUGER)+
		AugerRate(Z,L3_M3P5_AUGER)+
		AugerRate(Z,L3_M3Q1_AUGER)+
		AugerRate(Z,L3_M3Q2_AUGER)+
		AugerRate(Z,L3_M3Q3_AUGER)+
		AugerRate(Z,L3_M4M3_AUGER)+
		AugerRate(Z,L3_M5M3_AUGER)
		);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM13_TRANS)*PM1;
	
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM23_TRANS)*PM2;

	return rv;
}

double PM4_pure_kissel(int Z, double E, double PM1, double PM2, double PM3) {
	double rv;

	rv = CS_Photo_Partial(Z, M4_SHELL, E);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM14_TRANS)*PM1;

	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM24_TRANS)*PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z,FM34_TRANS)*PM3;

	return rv;
}

double PM4_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3) {
	double rv;

	rv = CS_Photo_Partial(Z, M4_SHELL, E);

	/*yes I know that KM4 lines are forbidden... */
	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM4_LINE);

	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M4_LINE);

	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M4_LINE);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M4_LINE);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM14_TRANS)*PM1;
	
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM24_TRANS)*PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z,FM34_TRANS)*PM3;

	return rv;

}

double PM4_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3) {
	double rv;

	rv = CS_Photo_Partial(Z, M4_SHELL, E);

	if (PK > 0.0) 
		rv += AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1M4_AUGER)+
		AugerRate(Z,K_L2M4_AUGER)+
		AugerRate(Z,K_L3M4_AUGER)+
		AugerRate(Z,K_M1M4_AUGER)+
		AugerRate(Z,K_M2M4_AUGER)+
		AugerRate(Z,K_M3M4_AUGER)+
		AugerRate(Z,K_M4L1_AUGER)+
		AugerRate(Z,K_M4L2_AUGER)+
		AugerRate(Z,K_M4L3_AUGER)+
		AugerRate(Z,K_M4M1_AUGER)+
		AugerRate(Z,K_M4M2_AUGER)+
		AugerRate(Z,K_M4M3_AUGER)+
		AugerRate(Z,K_M4M4_AUGER)+
		AugerRate(Z,K_M4M5_AUGER)+
		AugerRate(Z,K_M4N1_AUGER)+
		AugerRate(Z,K_M4N2_AUGER)+
		AugerRate(Z,K_M4N3_AUGER)+
		AugerRate(Z,K_M4N4_AUGER)+
		AugerRate(Z,K_M4N5_AUGER)+
		AugerRate(Z,K_M4N6_AUGER)+
		AugerRate(Z,K_M4N7_AUGER)+
		AugerRate(Z,K_M4O1_AUGER)+
		AugerRate(Z,K_M4O2_AUGER)+
		AugerRate(Z,K_M4O3_AUGER)+
		AugerRate(Z,K_M4O4_AUGER)+
		AugerRate(Z,K_M4O5_AUGER)+
		AugerRate(Z,K_M4O6_AUGER)+
		AugerRate(Z,K_M4O7_AUGER)+
		AugerRate(Z,K_M4P1_AUGER)+
		AugerRate(Z,K_M4P2_AUGER)+
		AugerRate(Z,K_M4P3_AUGER)+
		AugerRate(Z,K_M4P4_AUGER)+
		AugerRate(Z,K_M4P5_AUGER)+
		AugerRate(Z,K_M4Q1_AUGER)+
		AugerRate(Z,K_M4Q2_AUGER)+
		AugerRate(Z,K_M4Q3_AUGER)+
		AugerRate(Z,K_M5M4_AUGER)
		);
	if (PL1 > 0.0)
		rv += AugerYield(Z,L1_SHELL)*PL1*(
		AugerRate(Z,L1_M1M4_AUGER)+
		AugerRate(Z,L1_M2M4_AUGER)+
		AugerRate(Z,L1_M3M4_AUGER)+
		AugerRate(Z,L1_M4M1_AUGER)+
		AugerRate(Z,L1_M4M2_AUGER)+
		AugerRate(Z,L1_M4M3_AUGER)+
		AugerRate(Z,L1_M4M4_AUGER)+
		AugerRate(Z,L1_M4M5_AUGER)+
		AugerRate(Z,L1_M4N1_AUGER)+
		AugerRate(Z,L1_M4N2_AUGER)+
		AugerRate(Z,L1_M4N3_AUGER)+
		AugerRate(Z,L1_M4N4_AUGER)+
		AugerRate(Z,L1_M4N5_AUGER)+
		AugerRate(Z,L1_M4N6_AUGER)+
		AugerRate(Z,L1_M4N7_AUGER)+
		AugerRate(Z,L1_M4O1_AUGER)+
		AugerRate(Z,L1_M4O2_AUGER)+
		AugerRate(Z,L1_M4O3_AUGER)+
		AugerRate(Z,L1_M4O4_AUGER)+
		AugerRate(Z,L1_M4O5_AUGER)+
		AugerRate(Z,L1_M4O6_AUGER)+
		AugerRate(Z,L1_M4O7_AUGER)+
		AugerRate(Z,L1_M4P1_AUGER)+
		AugerRate(Z,L1_M4P2_AUGER)+
		AugerRate(Z,L1_M4P3_AUGER)+
		AugerRate(Z,L1_M4P4_AUGER)+
		AugerRate(Z,L1_M4P5_AUGER)+
		AugerRate(Z,L1_M4Q1_AUGER)+
		AugerRate(Z,L1_M4Q2_AUGER)+
		AugerRate(Z,L1_M4Q3_AUGER)+
		AugerRate(Z,L1_M5M4_AUGER)
		);
	if (PL2 > 0.0)
		rv += AugerYield(Z,L2_SHELL)*PL2*(
		AugerRate(Z,L2_M1M4_AUGER)+
		AugerRate(Z,L2_M2M4_AUGER)+
		AugerRate(Z,L2_M3M4_AUGER)+
		AugerRate(Z,L2_M4M1_AUGER)+
		AugerRate(Z,L2_M4M2_AUGER)+
		AugerRate(Z,L2_M4M3_AUGER)+
		AugerRate(Z,L2_M4M4_AUGER)+
		AugerRate(Z,L2_M4M5_AUGER)+
		AugerRate(Z,L2_M4N1_AUGER)+
		AugerRate(Z,L2_M4N2_AUGER)+
		AugerRate(Z,L2_M4N3_AUGER)+
		AugerRate(Z,L2_M4N4_AUGER)+
		AugerRate(Z,L2_M4N5_AUGER)+
		AugerRate(Z,L2_M4N6_AUGER)+
		AugerRate(Z,L2_M4N7_AUGER)+
		AugerRate(Z,L2_M4O1_AUGER)+
		AugerRate(Z,L2_M4O2_AUGER)+
		AugerRate(Z,L2_M4O3_AUGER)+
		AugerRate(Z,L2_M4O4_AUGER)+
		AugerRate(Z,L2_M4O5_AUGER)+
		AugerRate(Z,L2_M4O6_AUGER)+
		AugerRate(Z,L2_M4O7_AUGER)+
		AugerRate(Z,L2_M4P1_AUGER)+
		AugerRate(Z,L2_M4P2_AUGER)+
		AugerRate(Z,L2_M4P3_AUGER)+
		AugerRate(Z,L2_M4P4_AUGER)+
		AugerRate(Z,L2_M4P5_AUGER)+
		AugerRate(Z,L2_M4Q1_AUGER)+
		AugerRate(Z,L2_M4Q2_AUGER)+
		AugerRate(Z,L2_M4Q3_AUGER)+
		AugerRate(Z,L2_M5M4_AUGER)
		);
	if (PL3 > 0.0)
		rv += AugerYield(Z,L3_SHELL)*PL3*(
		AugerRate(Z,L3_M1M4_AUGER)+
		AugerRate(Z,L3_M2M4_AUGER)+
		AugerRate(Z,L3_M3M4_AUGER)+
		AugerRate(Z,L3_M4M1_AUGER)+
		AugerRate(Z,L3_M4M2_AUGER)+
		AugerRate(Z,L3_M4M3_AUGER)+
		AugerRate(Z,L3_M4M4_AUGER)+
		AugerRate(Z,L3_M4M5_AUGER)+
		AugerRate(Z,L3_M4N1_AUGER)+
		AugerRate(Z,L3_M4N2_AUGER)+
		AugerRate(Z,L3_M4N3_AUGER)+
		AugerRate(Z,L3_M4N4_AUGER)+
		AugerRate(Z,L3_M4N5_AUGER)+
		AugerRate(Z,L3_M4N6_AUGER)+
		AugerRate(Z,L3_M4N7_AUGER)+
		AugerRate(Z,L3_M4O1_AUGER)+
		AugerRate(Z,L3_M4O2_AUGER)+
		AugerRate(Z,L3_M4O3_AUGER)+
		AugerRate(Z,L3_M4O4_AUGER)+
		AugerRate(Z,L3_M4O5_AUGER)+
		AugerRate(Z,L3_M4O6_AUGER)+
		AugerRate(Z,L3_M4O7_AUGER)+
		AugerRate(Z,L3_M4P1_AUGER)+
		AugerRate(Z,L3_M4P2_AUGER)+
		AugerRate(Z,L3_M4P3_AUGER)+
		AugerRate(Z,L3_M4P4_AUGER)+
		AugerRate(Z,L3_M4P5_AUGER)+
		AugerRate(Z,L3_M4Q1_AUGER)+
		AugerRate(Z,L3_M4Q2_AUGER)+
		AugerRate(Z,L3_M4Q3_AUGER)+
		AugerRate(Z,L3_M5M4_AUGER)
		);
	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM14_TRANS)*PM1;

	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM24_TRANS)*PM2;

	if (PM3 > 0.0)	
		rv += CosKronTransProb(Z,FM34_TRANS)*PM3;

	return rv;
}

double PM4_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3) {
	double rv;

	rv = CS_Photo_Partial(Z, M4_SHELL, E);

	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM4_LINE)+
		AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1M4_AUGER)+
		AugerRate(Z,K_L2M4_AUGER)+
		AugerRate(Z,K_L3M4_AUGER)+
		AugerRate(Z,K_M1M4_AUGER)+
		AugerRate(Z,K_M2M4_AUGER)+
		AugerRate(Z,K_M3M4_AUGER)+
		AugerRate(Z,K_M4L1_AUGER)+
		AugerRate(Z,K_M4L2_AUGER)+
		AugerRate(Z,K_M4L3_AUGER)+
		AugerRate(Z,K_M4M1_AUGER)+
		AugerRate(Z,K_M4M2_AUGER)+
		AugerRate(Z,K_M4M3_AUGER)+
		AugerRate(Z,K_M4M4_AUGER)+
		AugerRate(Z,K_M4M5_AUGER)+
		AugerRate(Z,K_M4N1_AUGER)+
		AugerRate(Z,K_M4N2_AUGER)+
		AugerRate(Z,K_M4N3_AUGER)+
		AugerRate(Z,K_M4N4_AUGER)+
		AugerRate(Z,K_M4N5_AUGER)+
		AugerRate(Z,K_M4N6_AUGER)+
		AugerRate(Z,K_M4N7_AUGER)+
		AugerRate(Z,K_M4O1_AUGER)+
		AugerRate(Z,K_M4O2_AUGER)+
		AugerRate(Z,K_M4O3_AUGER)+
		AugerRate(Z,K_M4O4_AUGER)+
		AugerRate(Z,K_M4O5_AUGER)+
		AugerRate(Z,K_M4O6_AUGER)+
		AugerRate(Z,K_M4O7_AUGER)+
		AugerRate(Z,K_M4P1_AUGER)+
		AugerRate(Z,K_M4P2_AUGER)+
		AugerRate(Z,K_M4P3_AUGER)+
		AugerRate(Z,K_M4P4_AUGER)+
		AugerRate(Z,K_M4P5_AUGER)+
		AugerRate(Z,K_M4Q1_AUGER)+
		AugerRate(Z,K_M4Q2_AUGER)+
		AugerRate(Z,K_M4Q3_AUGER)+
		AugerRate(Z,K_M5M4_AUGER)
		);

	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M4_LINE)+
		AugerYield(Z,L1_SHELL)*PL1*(
		AugerRate(Z,L1_M1M4_AUGER)+
		AugerRate(Z,L1_M2M4_AUGER)+
		AugerRate(Z,L1_M3M4_AUGER)+
		AugerRate(Z,L1_M4M1_AUGER)+
		AugerRate(Z,L1_M4M2_AUGER)+
		AugerRate(Z,L1_M4M3_AUGER)+
		AugerRate(Z,L1_M4M4_AUGER)+
		AugerRate(Z,L1_M4M5_AUGER)+
		AugerRate(Z,L1_M4N1_AUGER)+
		AugerRate(Z,L1_M4N2_AUGER)+
		AugerRate(Z,L1_M4N3_AUGER)+
		AugerRate(Z,L1_M4N4_AUGER)+
		AugerRate(Z,L1_M4N5_AUGER)+
		AugerRate(Z,L1_M4N6_AUGER)+
		AugerRate(Z,L1_M4N7_AUGER)+
		AugerRate(Z,L1_M4O1_AUGER)+
		AugerRate(Z,L1_M4O2_AUGER)+
		AugerRate(Z,L1_M4O3_AUGER)+
		AugerRate(Z,L1_M4O4_AUGER)+
		AugerRate(Z,L1_M4O5_AUGER)+
		AugerRate(Z,L1_M4O6_AUGER)+
		AugerRate(Z,L1_M4O7_AUGER)+
		AugerRate(Z,L1_M4P1_AUGER)+
		AugerRate(Z,L1_M4P2_AUGER)+
		AugerRate(Z,L1_M4P3_AUGER)+
		AugerRate(Z,L1_M4P4_AUGER)+
		AugerRate(Z,L1_M4P5_AUGER)+
		AugerRate(Z,L1_M4Q1_AUGER)+
		AugerRate(Z,L1_M4Q2_AUGER)+
		AugerRate(Z,L1_M4Q3_AUGER)+
		AugerRate(Z,L1_M5M4_AUGER)
		);

	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M4_LINE)+
		AugerYield(Z,L2_SHELL)*PL2*(
		AugerRate(Z,L2_M1M4_AUGER)+
		AugerRate(Z,L2_M2M4_AUGER)+
		AugerRate(Z,L2_M3M4_AUGER)+
		AugerRate(Z,L2_M4M1_AUGER)+
		AugerRate(Z,L2_M4M2_AUGER)+
		AugerRate(Z,L2_M4M3_AUGER)+
		AugerRate(Z,L2_M4M4_AUGER)+
		AugerRate(Z,L2_M4M5_AUGER)+
		AugerRate(Z,L2_M4N1_AUGER)+
		AugerRate(Z,L2_M4N2_AUGER)+
		AugerRate(Z,L2_M4N3_AUGER)+
		AugerRate(Z,L2_M4N4_AUGER)+
		AugerRate(Z,L2_M4N5_AUGER)+
		AugerRate(Z,L2_M4N6_AUGER)+
		AugerRate(Z,L2_M4N7_AUGER)+
		AugerRate(Z,L2_M4O1_AUGER)+
		AugerRate(Z,L2_M4O2_AUGER)+
		AugerRate(Z,L2_M4O3_AUGER)+
		AugerRate(Z,L2_M4O4_AUGER)+
		AugerRate(Z,L2_M4O5_AUGER)+
		AugerRate(Z,L2_M4O6_AUGER)+
		AugerRate(Z,L2_M4O7_AUGER)+
		AugerRate(Z,L2_M4P1_AUGER)+
		AugerRate(Z,L2_M4P2_AUGER)+
		AugerRate(Z,L2_M4P3_AUGER)+
		AugerRate(Z,L2_M4P4_AUGER)+
		AugerRate(Z,L2_M4P5_AUGER)+
		AugerRate(Z,L2_M4Q1_AUGER)+
		AugerRate(Z,L2_M4Q2_AUGER)+
		AugerRate(Z,L2_M4Q3_AUGER)+
		AugerRate(Z,L2_M5M4_AUGER)
		);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M4_LINE)+
		AugerYield(Z,L3_SHELL)*PL3*(
		AugerRate(Z,L3_M1M4_AUGER)+
		AugerRate(Z,L3_M2M4_AUGER)+
		AugerRate(Z,L3_M3M4_AUGER)+
		AugerRate(Z,L3_M4M1_AUGER)+
		AugerRate(Z,L3_M4M2_AUGER)+
		AugerRate(Z,L3_M4M3_AUGER)+
		AugerRate(Z,L3_M4M4_AUGER)+
		AugerRate(Z,L3_M4M5_AUGER)+
		AugerRate(Z,L3_M4N1_AUGER)+
		AugerRate(Z,L3_M4N2_AUGER)+
		AugerRate(Z,L3_M4N3_AUGER)+
		AugerRate(Z,L3_M4N4_AUGER)+
		AugerRate(Z,L3_M4N5_AUGER)+
		AugerRate(Z,L3_M4N6_AUGER)+
		AugerRate(Z,L3_M4N7_AUGER)+
		AugerRate(Z,L3_M4O1_AUGER)+
		AugerRate(Z,L3_M4O2_AUGER)+
		AugerRate(Z,L3_M4O3_AUGER)+
		AugerRate(Z,L3_M4O4_AUGER)+
		AugerRate(Z,L3_M4O5_AUGER)+
		AugerRate(Z,L3_M4O6_AUGER)+
		AugerRate(Z,L3_M4O7_AUGER)+
		AugerRate(Z,L3_M4P1_AUGER)+
		AugerRate(Z,L3_M4P2_AUGER)+
		AugerRate(Z,L3_M4P3_AUGER)+
		AugerRate(Z,L3_M4P4_AUGER)+
		AugerRate(Z,L3_M4P5_AUGER)+
		AugerRate(Z,L3_M4Q1_AUGER)+
		AugerRate(Z,L3_M4Q2_AUGER)+
		AugerRate(Z,L3_M4Q3_AUGER)+
		AugerRate(Z,L3_M5M4_AUGER)
		);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM14_TRANS)*PM1;
	
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM24_TRANS)*PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z,FM34_TRANS)*PM3;

	return rv;
}

double PM5_pure_kissel(int Z, double E, double PM1, double PM2, double PM3, double PM4) {
	double rv;

	rv = CS_Photo_Partial(Z, M5_SHELL, E);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM15_TRANS)*PM1;

	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM25_TRANS)*PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z,FM35_TRANS)*PM3;

	if (PM4 > 0.0)
		rv += CosKronTransProb(Z,FM45_TRANS)*PM4;

	return rv;
}

double PM5_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4) {
	double rv;

	rv = CS_Photo_Partial(Z, M5_SHELL, E);

	/*yes I know that KM5 lines are forbidden... */
	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM5_LINE);

	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M5_LINE);

	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M5_LINE);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M5_LINE);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM15_TRANS)*PM1;
	
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM25_TRANS)*PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z,FM35_TRANS)*PM3;

	if (PM4 > 0.0)
		rv += CosKronTransProb(Z,FM45_TRANS)*PM4;

	return rv;
}

double PM5_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4) {
	double rv;

	rv = CS_Photo_Partial(Z, M5_SHELL, E);

	if (PK > 0.0) 
		rv += AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1M5_AUGER)+
		AugerRate(Z,K_L2M5_AUGER)+
		AugerRate(Z,K_L3M5_AUGER)+
		AugerRate(Z,K_M1M5_AUGER)+
		AugerRate(Z,K_M2M5_AUGER)+
		AugerRate(Z,K_M3M5_AUGER)+
		AugerRate(Z,K_M4M5_AUGER)+
		AugerRate(Z,K_M5L1_AUGER)+
		AugerRate(Z,K_M5L2_AUGER)+
		AugerRate(Z,K_M5L3_AUGER)+
		AugerRate(Z,K_M5M1_AUGER)+
		AugerRate(Z,K_M5M2_AUGER)+
		AugerRate(Z,K_M5M3_AUGER)+
		AugerRate(Z,K_M5M4_AUGER)+
		AugerRate(Z,K_M5M5_AUGER)+
		AugerRate(Z,K_M5N1_AUGER)+
		AugerRate(Z,K_M5N2_AUGER)+
		AugerRate(Z,K_M5N3_AUGER)+
		AugerRate(Z,K_M5N4_AUGER)+
		AugerRate(Z,K_M5N5_AUGER)+
		AugerRate(Z,K_M5N6_AUGER)+
		AugerRate(Z,K_M5N7_AUGER)+
		AugerRate(Z,K_M5O1_AUGER)+
		AugerRate(Z,K_M5O2_AUGER)+
		AugerRate(Z,K_M5O3_AUGER)+
		AugerRate(Z,K_M5O4_AUGER)+
		AugerRate(Z,K_M5O5_AUGER)+
		AugerRate(Z,K_M5O6_AUGER)+
		AugerRate(Z,K_M5O7_AUGER)+
		AugerRate(Z,K_M5P1_AUGER)+
		AugerRate(Z,K_M5P2_AUGER)+
		AugerRate(Z,K_M5P3_AUGER)+
		AugerRate(Z,K_M5P4_AUGER)+
		AugerRate(Z,K_M5P5_AUGER)+
		AugerRate(Z,K_M5Q1_AUGER)+
		AugerRate(Z,K_M5Q2_AUGER)+
		AugerRate(Z,K_M5Q3_AUGER)
		);
	if (PL1 > 0.0)
		rv += AugerYield(Z,L1_SHELL)*PL1*(
		AugerRate(Z,L1_M1M5_AUGER)+
		AugerRate(Z,L1_M2M5_AUGER)+
		AugerRate(Z,L1_M3M5_AUGER)+
		AugerRate(Z,L1_M4M5_AUGER)+
		AugerRate(Z,L1_M5M1_AUGER)+
		AugerRate(Z,L1_M5M2_AUGER)+
		AugerRate(Z,L1_M5M3_AUGER)+
		AugerRate(Z,L1_M5M4_AUGER)+
		AugerRate(Z,L1_M5M5_AUGER)+
		AugerRate(Z,L1_M5N1_AUGER)+
		AugerRate(Z,L1_M5N2_AUGER)+
		AugerRate(Z,L1_M5N3_AUGER)+
		AugerRate(Z,L1_M5N4_AUGER)+
		AugerRate(Z,L1_M5N5_AUGER)+
		AugerRate(Z,L1_M5N6_AUGER)+
		AugerRate(Z,L1_M5N7_AUGER)+
		AugerRate(Z,L1_M5O1_AUGER)+
		AugerRate(Z,L1_M5O2_AUGER)+
		AugerRate(Z,L1_M5O3_AUGER)+
		AugerRate(Z,L1_M5O4_AUGER)+
		AugerRate(Z,L1_M5O5_AUGER)+
		AugerRate(Z,L1_M5O6_AUGER)+
		AugerRate(Z,L1_M5O7_AUGER)+
		AugerRate(Z,L1_M5P1_AUGER)+
		AugerRate(Z,L1_M5P2_AUGER)+
		AugerRate(Z,L1_M5P3_AUGER)+
		AugerRate(Z,L1_M5P4_AUGER)+
		AugerRate(Z,L1_M5P5_AUGER)+
		AugerRate(Z,L1_M5Q1_AUGER)+
		AugerRate(Z,L1_M5Q2_AUGER)+
		AugerRate(Z,L1_M5Q3_AUGER)
		);
	if (PL2 > 0.0)
		rv += AugerYield(Z,L2_SHELL)*PL2*(
		AugerRate(Z,L2_M1M5_AUGER)+
		AugerRate(Z,L2_M2M5_AUGER)+
		AugerRate(Z,L2_M3M5_AUGER)+
		AugerRate(Z,L2_M4M5_AUGER)+
		AugerRate(Z,L2_M5M1_AUGER)+
		AugerRate(Z,L2_M5M2_AUGER)+
		AugerRate(Z,L2_M5M3_AUGER)+
		AugerRate(Z,L2_M5M4_AUGER)+
		AugerRate(Z,L2_M5M5_AUGER)+
		AugerRate(Z,L2_M5N1_AUGER)+
		AugerRate(Z,L2_M5N2_AUGER)+
		AugerRate(Z,L2_M5N3_AUGER)+
		AugerRate(Z,L2_M5N4_AUGER)+
		AugerRate(Z,L2_M5N5_AUGER)+
		AugerRate(Z,L2_M5N6_AUGER)+
		AugerRate(Z,L2_M5N7_AUGER)+
		AugerRate(Z,L2_M5O1_AUGER)+
		AugerRate(Z,L2_M5O2_AUGER)+
		AugerRate(Z,L2_M5O3_AUGER)+
		AugerRate(Z,L2_M5O4_AUGER)+
		AugerRate(Z,L2_M5O5_AUGER)+
		AugerRate(Z,L2_M5O6_AUGER)+
		AugerRate(Z,L2_M5O7_AUGER)+
		AugerRate(Z,L2_M5P1_AUGER)+
		AugerRate(Z,L2_M5P2_AUGER)+
		AugerRate(Z,L2_M5P3_AUGER)+
		AugerRate(Z,L2_M5P4_AUGER)+
		AugerRate(Z,L2_M5P5_AUGER)+
		AugerRate(Z,L2_M5Q1_AUGER)+
		AugerRate(Z,L2_M5Q2_AUGER)+
		AugerRate(Z,L2_M5Q3_AUGER)
		);
	if (PL3 > 0.0)
		rv += AugerYield(Z,L3_SHELL)*PL3*(
		AugerRate(Z,L3_M1M5_AUGER)+
		AugerRate(Z,L3_M2M5_AUGER)+
		AugerRate(Z,L3_M3M5_AUGER)+
		AugerRate(Z,L3_M4M5_AUGER)+
		AugerRate(Z,L3_M5M1_AUGER)+
		AugerRate(Z,L3_M5M2_AUGER)+
		AugerRate(Z,L3_M5M3_AUGER)+
		AugerRate(Z,L3_M5M4_AUGER)+
		AugerRate(Z,L3_M5M5_AUGER)+
		AugerRate(Z,L3_M5N1_AUGER)+
		AugerRate(Z,L3_M5N2_AUGER)+
		AugerRate(Z,L3_M5N3_AUGER)+
		AugerRate(Z,L3_M5N4_AUGER)+
		AugerRate(Z,L3_M5N5_AUGER)+
		AugerRate(Z,L3_M5N6_AUGER)+
		AugerRate(Z,L3_M5N7_AUGER)+
		AugerRate(Z,L3_M5O1_AUGER)+
		AugerRate(Z,L3_M5O2_AUGER)+
		AugerRate(Z,L3_M5O3_AUGER)+
		AugerRate(Z,L3_M5O4_AUGER)+
		AugerRate(Z,L3_M5O5_AUGER)+
		AugerRate(Z,L3_M5O6_AUGER)+
		AugerRate(Z,L3_M5O7_AUGER)+
		AugerRate(Z,L3_M5P1_AUGER)+
		AugerRate(Z,L3_M5P2_AUGER)+
		AugerRate(Z,L3_M5P3_AUGER)+
		AugerRate(Z,L3_M5P4_AUGER)+
		AugerRate(Z,L3_M5P5_AUGER)+
		AugerRate(Z,L3_M5Q1_AUGER)+
		AugerRate(Z,L3_M5Q2_AUGER)+
		AugerRate(Z,L3_M5Q3_AUGER)
		);
	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM15_TRANS)*PM1;
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM25_TRANS)*PM2;
	if (PM3 > 0.0)	
		rv += CosKronTransProb(Z,FM35_TRANS)*PM3;
	if (PM4 > 0.0)	
		rv += CosKronTransProb(Z,FM45_TRANS)*PM4;

	return rv;
}

double PM5_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4) {
	double rv;

	rv = CS_Photo_Partial(Z, M5_SHELL, E);

	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM5_LINE)+
		AugerYield(Z,K_SHELL)*PK*(
		AugerRate(Z,K_L1M4_AUGER)+
		AugerRate(Z,K_L2M4_AUGER)+
		AugerRate(Z,K_L3M4_AUGER)+
		AugerRate(Z,K_M1M4_AUGER)+
		AugerRate(Z,K_M2M4_AUGER)+
		AugerRate(Z,K_M3M4_AUGER)+
		AugerRate(Z,K_M4L1_AUGER)+
		AugerRate(Z,K_M4L2_AUGER)+
		AugerRate(Z,K_M4L3_AUGER)+
		AugerRate(Z,K_M4M1_AUGER)+
		AugerRate(Z,K_M4M2_AUGER)+
		AugerRate(Z,K_M4M3_AUGER)+
		AugerRate(Z,K_M4M4_AUGER)+
		AugerRate(Z,K_M4M5_AUGER)+
		AugerRate(Z,K_M4N1_AUGER)+
		AugerRate(Z,K_M4N2_AUGER)+
		AugerRate(Z,K_M4N3_AUGER)+
		AugerRate(Z,K_M4N4_AUGER)+
		AugerRate(Z,K_M4N5_AUGER)+
		AugerRate(Z,K_M4N6_AUGER)+
		AugerRate(Z,K_M4N7_AUGER)+
		AugerRate(Z,K_M4O1_AUGER)+
		AugerRate(Z,K_M4O2_AUGER)+
		AugerRate(Z,K_M4O3_AUGER)+
		AugerRate(Z,K_M4O4_AUGER)+
		AugerRate(Z,K_M4O5_AUGER)+
		AugerRate(Z,K_M4O6_AUGER)+
		AugerRate(Z,K_M4O7_AUGER)+
		AugerRate(Z,K_M4P1_AUGER)+
		AugerRate(Z,K_M4P2_AUGER)+
		AugerRate(Z,K_M4P3_AUGER)+
		AugerRate(Z,K_M4P4_AUGER)+
		AugerRate(Z,K_M4P5_AUGER)+
		AugerRate(Z,K_M4Q1_AUGER)+
		AugerRate(Z,K_M4Q2_AUGER)+
		AugerRate(Z,K_M4Q3_AUGER)+
		AugerRate(Z,K_M5M4_AUGER)
		);

	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M5_LINE)+
		AugerYield(Z,L1_SHELL)*PL1*(
		AugerRate(Z,L1_M1M4_AUGER)+
		AugerRate(Z,L1_M2M4_AUGER)+
		AugerRate(Z,L1_M3M4_AUGER)+
		AugerRate(Z,L1_M4M1_AUGER)+
		AugerRate(Z,L1_M4M2_AUGER)+
		AugerRate(Z,L1_M4M3_AUGER)+
		AugerRate(Z,L1_M4M4_AUGER)+
		AugerRate(Z,L1_M4M5_AUGER)+
		AugerRate(Z,L1_M4N1_AUGER)+
		AugerRate(Z,L1_M4N2_AUGER)+
		AugerRate(Z,L1_M4N3_AUGER)+
		AugerRate(Z,L1_M4N4_AUGER)+
		AugerRate(Z,L1_M4N5_AUGER)+
		AugerRate(Z,L1_M4N6_AUGER)+
		AugerRate(Z,L1_M4N7_AUGER)+
		AugerRate(Z,L1_M4O1_AUGER)+
		AugerRate(Z,L1_M4O2_AUGER)+
		AugerRate(Z,L1_M4O3_AUGER)+
		AugerRate(Z,L1_M4O4_AUGER)+
		AugerRate(Z,L1_M4O5_AUGER)+
		AugerRate(Z,L1_M4O6_AUGER)+
		AugerRate(Z,L1_M4O7_AUGER)+
		AugerRate(Z,L1_M4P1_AUGER)+
		AugerRate(Z,L1_M4P2_AUGER)+
		AugerRate(Z,L1_M4P3_AUGER)+
		AugerRate(Z,L1_M4P4_AUGER)+
		AugerRate(Z,L1_M4P5_AUGER)+
		AugerRate(Z,L1_M4Q1_AUGER)+
		AugerRate(Z,L1_M4Q2_AUGER)+
		AugerRate(Z,L1_M4Q3_AUGER)+
		AugerRate(Z,L1_M5M4_AUGER)
		);

	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M5_LINE)+
		AugerYield(Z,L2_SHELL)*PL2*(
		AugerRate(Z,L2_M1M4_AUGER)+
		AugerRate(Z,L2_M2M4_AUGER)+
		AugerRate(Z,L2_M3M4_AUGER)+
		AugerRate(Z,L2_M4M1_AUGER)+
		AugerRate(Z,L2_M4M2_AUGER)+
		AugerRate(Z,L2_M4M3_AUGER)+
		AugerRate(Z,L2_M4M4_AUGER)+
		AugerRate(Z,L2_M4M5_AUGER)+
		AugerRate(Z,L2_M4N1_AUGER)+
		AugerRate(Z,L2_M4N2_AUGER)+
		AugerRate(Z,L2_M4N3_AUGER)+
		AugerRate(Z,L2_M4N4_AUGER)+
		AugerRate(Z,L2_M4N5_AUGER)+
		AugerRate(Z,L2_M4N6_AUGER)+
		AugerRate(Z,L2_M4N7_AUGER)+
		AugerRate(Z,L2_M4O1_AUGER)+
		AugerRate(Z,L2_M4O2_AUGER)+
		AugerRate(Z,L2_M4O3_AUGER)+
		AugerRate(Z,L2_M4O4_AUGER)+
		AugerRate(Z,L2_M4O5_AUGER)+
		AugerRate(Z,L2_M4O6_AUGER)+
		AugerRate(Z,L2_M4O7_AUGER)+
		AugerRate(Z,L2_M4P1_AUGER)+
		AugerRate(Z,L2_M4P2_AUGER)+
		AugerRate(Z,L2_M4P3_AUGER)+
		AugerRate(Z,L2_M4P4_AUGER)+
		AugerRate(Z,L2_M4P5_AUGER)+
		AugerRate(Z,L2_M4Q1_AUGER)+
		AugerRate(Z,L2_M4Q2_AUGER)+
		AugerRate(Z,L2_M4Q3_AUGER)+
		AugerRate(Z,L2_M5M4_AUGER)
		);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M5_LINE)+
		AugerYield(Z,L3_SHELL)*PL3*(
		AugerRate(Z,L3_M1M4_AUGER)+
		AugerRate(Z,L3_M2M4_AUGER)+
		AugerRate(Z,L3_M3M4_AUGER)+
		AugerRate(Z,L3_M4M1_AUGER)+
		AugerRate(Z,L3_M4M2_AUGER)+
		AugerRate(Z,L3_M4M3_AUGER)+
		AugerRate(Z,L3_M4M4_AUGER)+
		AugerRate(Z,L3_M4M5_AUGER)+
		AugerRate(Z,L3_M4N1_AUGER)+
		AugerRate(Z,L3_M4N2_AUGER)+
		AugerRate(Z,L3_M4N3_AUGER)+
		AugerRate(Z,L3_M4N4_AUGER)+
		AugerRate(Z,L3_M4N5_AUGER)+
		AugerRate(Z,L3_M4N6_AUGER)+
		AugerRate(Z,L3_M4N7_AUGER)+
		AugerRate(Z,L3_M4O1_AUGER)+
		AugerRate(Z,L3_M4O2_AUGER)+
		AugerRate(Z,L3_M4O3_AUGER)+
		AugerRate(Z,L3_M4O4_AUGER)+
		AugerRate(Z,L3_M4O5_AUGER)+
		AugerRate(Z,L3_M4O6_AUGER)+
		AugerRate(Z,L3_M4O7_AUGER)+
		AugerRate(Z,L3_M4P1_AUGER)+
		AugerRate(Z,L3_M4P2_AUGER)+
		AugerRate(Z,L3_M4P3_AUGER)+
		AugerRate(Z,L3_M4P4_AUGER)+
		AugerRate(Z,L3_M4P5_AUGER)+
		AugerRate(Z,L3_M4Q1_AUGER)+
		AugerRate(Z,L3_M4Q2_AUGER)+
		AugerRate(Z,L3_M4Q3_AUGER)+
		AugerRate(Z,L3_M5M4_AUGER)
		);
	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM15_TRANS)*PM1;
	
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM25_TRANS)*PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z,FM35_TRANS)*PM3;

	if (PM4 > 0.0)
		rv += CosKronTransProb(Z,FM45_TRANS)*PM4;

	return rv;
}


