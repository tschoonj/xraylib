/*
Copyright (c) 2009-2019 Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "config.h"
#include "xrf_cross_sections_aux.h"
#include "xraylib-error-private.h"
#include "xrayglob.h"
#include <stddef.h>
#include <stdio.h>


double PL1_pure_kissel(int Z, double E, xrl_error **error) {
	return CS_Photo_Partial(Z, L1_SHELL, E, error);
}

double PL1_rad_cascade_kissel(int Z, double E, double PK, xrl_error **error) {
	double rv;
	rv = CS_Photo_Partial(Z, L1_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0) {
		rv += FluorYield(Z, K_SHELL, NULL) * PK * RadRate(Z, KL1_LINE, NULL);
	}

	return rv;
}

double PL1_auger_cascade_kissel(int Z, double E, double PK, xrl_error **error) {
	double rv;
	rv = CS_Photo_Partial(Z, L1_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0) {
		rv += PK * xrf_cross_sections_constants_auger_only[Z][L1_SHELL][K_SHELL];
	}

	return rv;	
}

double PL1_full_cascade_kissel(int Z, double E, double PK, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, L1_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;
	if (PK > 0.0) {
		rv += PK * xrf_cross_sections_constants_full[Z][L1_SHELL][K_SHELL];
	}
	return rv;
}

double PL2_pure_kissel(int Z, double E, double PL1, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, L2_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PL1 > 0.0) {
		rv += CosKronTransProb(Z, FL12_TRANS, NULL) * PL1;
	}
	return rv;	
}

double PL2_rad_cascade_kissel(int Z, double E, double PK, double PL1, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, L2_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0) {
		rv += FluorYield(Z, K_SHELL, NULL) * PK * RadRate(Z, KL2_LINE, NULL);
	}

	if (PL1 > 0.0) {
		rv +=  CosKronTransProb(Z, FL12_TRANS, NULL) * PL1;
	}
	return  rv;
}

double PL2_auger_cascade_kissel(int Z, double E, double PK, double PL1, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z,L2_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0) {
		rv += PK * xrf_cross_sections_constants_auger_only[Z][L2_SHELL][K_SHELL];
	}

	if (PL1 > 0.0) {
		rv += CosKronTransProb(Z, FL12_TRANS, NULL) * PL1;
	}
	return  rv;
}

double PL2_full_cascade_kissel(int Z, double E, double PK, double PL1, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, L2_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0) {
		rv += PK * xrf_cross_sections_constants_full[Z][L2_SHELL][K_SHELL];
	}
		
	if (PL1 > 0.0)
		rv += CosKronTransProb(Z, FL12_TRANS, NULL) * PL1;
	return rv;
}

double PL3_pure_kissel(int Z, double E, double PL1, double PL2, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, L3_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PL1 > 0.0)
		rv += (CosKronTransProb(Z, FL13_TRANS, NULL) + CosKronTransProb(Z, FLP13_TRANS, NULL)) * PL1;

	if (PL2 > 0.0)
		rv += CosKronTransProb(Z, FL23_TRANS, NULL) * PL2;

	return rv;
}

double PL3_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, L3_SHELL, E, error);
	if (rv == 0.0)
		return rv;

	if (PK > 0.0)
		rv += FluorYield(Z, K_SHELL, NULL) * PK * RadRate(Z, KL3_LINE, NULL);

	if (PL1 > 0.0)
		rv += (CosKronTransProb(Z, FL13_TRANS, NULL) + CosKronTransProb(Z, FLP13_TRANS, NULL)) * PL1;

	if (PL2 > 0.0)
		rv += CosKronTransProb(Z, FL23_TRANS, NULL) * PL2;

	return rv;
}

double PL3_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, L3_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0)
		rv += PK * xrf_cross_sections_constants_auger_only[Z][L3_SHELL][K_SHELL];

	if (PL1 > 0.0)
		rv += (CosKronTransProb(Z, FL13_TRANS, NULL) + CosKronTransProb(Z, FLP13_TRANS, NULL)) * PL1;

	if (PL2 > 0.0)
		rv += CosKronTransProb(Z, FL23_TRANS, NULL) * PL2;

	return  rv;
}

double PL3_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, L3_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0)
		rv += PK * xrf_cross_sections_constants_full[Z][L3_SHELL][K_SHELL];

	if (PL1 > 0.0)
		rv += (CosKronTransProb(Z, FL13_TRANS, NULL) + CosKronTransProb(Z, FLP13_TRANS, NULL)) * PL1;

	if (PL2 > 0.0)
		rv += CosKronTransProb(Z, FL23_TRANS, NULL) * PL2;

	return rv;
}

double PM1_pure_kissel(int Z, double E, xrl_error **error) {
	return CS_Photo_Partial(Z, M1_SHELL, E, error);
}

double PM1_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M1_SHELL, E, error);
	if (rv == 0.0)
		return rv;

	if (PK > 0.0)
		rv += FluorYield(Z, K_SHELL, NULL) * PK * RadRate(Z ,KM1_LINE, NULL);
	if (PL1 > 0.0)
		rv += FluorYield(Z, L1_SHELL, NULL) * PL1 * RadRate(Z, L1M1_LINE, NULL);
	if (PL2 > 0.0)
		rv += FluorYield(Z, L2_SHELL, NULL) * PL2 * RadRate(Z, L2M1_LINE, NULL);
	if (PL3 > 0.0)
		rv += FluorYield(Z, L3_SHELL, NULL) * PL3 * RadRate(Z, L3M1_LINE, NULL);

	return rv; 
}

double PM1_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M1_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0)
		rv += PK * xrf_cross_sections_constants_auger_only[Z][M1_SHELL][K_SHELL];
	
	if (PL1 > 0.0)
		rv += PL1 * xrf_cross_sections_constants_auger_only[Z][M1_SHELL][L1_SHELL];

	if (PL2 > 0.0) 
		rv += PL2 * xrf_cross_sections_constants_auger_only[Z][M1_SHELL][L2_SHELL];
	
	if (PL3 > 0.0)
		rv += PL3 * xrf_cross_sections_constants_auger_only[Z][M1_SHELL][L3_SHELL];
	return rv;
}

double PM1_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M1_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0) 
		rv += PK * xrf_cross_sections_constants_full[Z][M1_SHELL][K_SHELL];
		
	if (PL1 > 0.0)
		rv += PL1 * xrf_cross_sections_constants_full[Z][M1_SHELL][L1_SHELL];
	
	if (PL2 > 0.0)
		rv += PL2 * xrf_cross_sections_constants_full[Z][M1_SHELL][L2_SHELL];

	if (PL3 > 0.0)
		rv += PL3 * xrf_cross_sections_constants_full[Z][M1_SHELL][L3_SHELL];

	return rv;
}

double PM2_pure_kissel(int Z, double E, double PM1, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M2_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM12_TRANS, NULL) * PM1;
		
	return rv; 
}

double PM2_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M2_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0)
		rv += FluorYield(Z, K_SHELL, NULL) * PK * RadRate(Z, KM2_LINE, NULL);

	if (PL1 > 0.0)
		rv += FluorYield(Z, L1_SHELL, NULL) * PL1 * RadRate(Z, L1M2_LINE, NULL);

	if (PL2 > 0.0)
		rv += FluorYield(Z, L2_SHELL, NULL) * PL2 * RadRate(Z, L2M2_LINE, NULL);

	if (PL3 > 0.0)
		rv += FluorYield(Z, L3_SHELL, NULL) * PL3 * RadRate(Z, L3M2_LINE, NULL);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM12_TRANS, NULL) * PM1;

	return rv;
}

double PM2_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M2_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0)
		rv += PK * xrf_cross_sections_constants_auger_only[Z][M2_SHELL][K_SHELL];

	if (PL1 > 0.0)
		rv += PL1 * xrf_cross_sections_constants_auger_only[Z][M2_SHELL][L1_SHELL];

	if (PL2 > 0.0)
		rv += PL2 * xrf_cross_sections_constants_auger_only[Z][M2_SHELL][L2_SHELL];

	if (PL3 > 0.0)
		rv += PL3 * xrf_cross_sections_constants_auger_only[Z][M2_SHELL][L3_SHELL];

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM12_TRANS, NULL) * PM1;

	return rv;
}

double PM2_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M2_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0) 
		rv += PK * xrf_cross_sections_constants_full[Z][M2_SHELL][K_SHELL];

	if (PL1 > 0.0) 
		rv += PL1 * xrf_cross_sections_constants_full[Z][M2_SHELL][L1_SHELL];
	
	if (PL2 > 0.0)
		rv += PL2 * xrf_cross_sections_constants_full[Z][M2_SHELL][L2_SHELL];

	if (PL3 > 0.0)
		rv += PL3 * xrf_cross_sections_constants_full[Z][M2_SHELL][L3_SHELL];

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM12_TRANS, NULL) * PM1;

	return rv;
}

double PM3_pure_kissel(int Z, double E, double PM1, double PM2, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M3_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM13_TRANS, NULL) * PM1;

	if (PM2 > 0.0)
		rv += CosKronTransProb(Z, FM23_TRANS, NULL) * PM2;

	return rv;
}

double PM3_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M3_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0) 
		rv += FluorYield(Z, K_SHELL, NULL) * PK * RadRate(Z, KM3_LINE, NULL);

	if (PL1 > 0.0)
		rv += FluorYield(Z, L1_SHELL, NULL) * PL1 * RadRate(Z, L1M3_LINE, NULL);

	if (PL2 > 0.0)
		rv += FluorYield(Z, L2_SHELL, NULL) * PL2 * RadRate(Z, L2M3_LINE, NULL);

	if (PL3 > 0.0)
		rv += FluorYield(Z, L3_SHELL, NULL) * PL3 * RadRate(Z, L3M3_LINE, NULL);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM13_TRANS, NULL) * PM1;
	
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z, FM23_TRANS, NULL) * PM2;

	return rv;
}

double PM3_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M3_SHELL, E, error);
	if (rv == 0.0)
		return rv;

	if (PK > 0.0)
		rv += PK * xrf_cross_sections_constants_auger_only[Z][M3_SHELL][K_SHELL];
	if (PL1 > 0.0)
		rv += PL1 * xrf_cross_sections_constants_auger_only[Z][M3_SHELL][L1_SHELL];
	if (PL2 > 0.0)
		rv += PL2 * xrf_cross_sections_constants_auger_only[Z][M3_SHELL][L2_SHELL];
	if (PL3 > 0.0) 
		rv += PL3 * xrf_cross_sections_constants_auger_only[Z][M3_SHELL][L3_SHELL];
	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM13_TRANS, NULL) * PM1;
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z, FM23_TRANS, NULL) * PM2;

	return rv;
}

double PM3_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M3_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0) 
		rv += PK * xrf_cross_sections_constants_full[Z][M3_SHELL][K_SHELL];

	if (PL1 > 0.0)
		rv += PL1 * xrf_cross_sections_constants_full[Z][M3_SHELL][L1_SHELL];

	if (PL2 > 0.0)
		rv += PL2 * xrf_cross_sections_constants_full[Z][M3_SHELL][L2_SHELL];

	if (PL3 > 0.0)
		rv += PL3 * xrf_cross_sections_constants_full[Z][M3_SHELL][L3_SHELL];

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM13_TRANS, NULL) * PM1;
	
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z, FM23_TRANS, NULL) * PM2;

	return rv;
}

double PM4_pure_kissel(int Z, double E, double PM1, double PM2, double PM3, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M4_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM14_TRANS, NULL) * PM1;

	if (PM2 > 0.0)
		rv += CosKronTransProb(Z, FM24_TRANS, NULL) * PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z, FM34_TRANS, NULL) * PM3;

	return rv;
}

double PM4_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M4_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	/*yes I know that KM4 lines are forbidden... */
	if (PK > 0.0) 
		rv += FluorYield(Z, K_SHELL, NULL) * PK *RadRate(Z, KM4_LINE, NULL);

	if (PL1 > 0.0)
		rv += FluorYield(Z, L1_SHELL, NULL) * PL1 * RadRate(Z, L1M4_LINE, NULL);

	if (PL2 > 0.0)
		rv += FluorYield(Z, L2_SHELL, NULL) * PL2 * RadRate(Z, L2M4_LINE, NULL);

	if (PL3 > 0.0)
		rv += FluorYield(Z, L3_SHELL, NULL) * PL3 * RadRate(Z, L3M4_LINE, NULL);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM14_TRANS, NULL) * PM1;
	
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z, FM24_TRANS, NULL) * PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z, FM34_TRANS, NULL) * PM3;

	return rv;

}

double PM4_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M4_SHELL, E, error);
	if (rv == 0.0)
		return rv;

	if (PK > 0.0) 
		rv += PK * xrf_cross_sections_constants_auger_only[Z][M4_SHELL][K_SHELL];
	if (PL1 > 0.0)
		rv += PL1 * xrf_cross_sections_constants_auger_only[Z][M4_SHELL][L1_SHELL];
	if (PL2 > 0.0)
		rv += PL2 * xrf_cross_sections_constants_auger_only[Z][M4_SHELL][L2_SHELL];
	if (PL3 > 0.0)
		rv += PL3 * xrf_cross_sections_constants_auger_only[Z][M4_SHELL][L3_SHELL];

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM14_TRANS, NULL) * PM1;

	if (PM2 > 0.0)
		rv += CosKronTransProb(Z, FM24_TRANS, NULL) * PM2;

	if (PM3 > 0.0)	
		rv += CosKronTransProb(Z, FM34_TRANS, NULL) * PM3;

	return rv;
}

double PM4_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M4_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0) 
		rv += PK * xrf_cross_sections_constants_full[Z][M4_SHELL][K_SHELL];

	if (PL1 > 0.0)
		rv += PL1 * xrf_cross_sections_constants_full[Z][M4_SHELL][L1_SHELL];

	if (PL2 > 0.0)
		rv += PL2 * xrf_cross_sections_constants_full[Z][M4_SHELL][L2_SHELL];

	if (PL3 > 0.0)
		rv += PL3 * xrf_cross_sections_constants_full[Z][M4_SHELL][L3_SHELL];

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM14_TRANS, NULL) * PM1;
	
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z, FM24_TRANS, NULL) * PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z, FM34_TRANS, NULL) * PM3;

	return rv;
}

double PM5_pure_kissel(int Z, double E, double PM1, double PM2, double PM3, double PM4, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M5_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM15_TRANS, NULL) * PM1;

	if (PM2 > 0.0)
		rv += CosKronTransProb(Z, FM25_TRANS, NULL) * PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z, FM35_TRANS, NULL) * PM3;

	if (PM4 > 0.0)
		rv += CosKronTransProb(Z, FM45_TRANS, NULL) * PM4;

	return rv;
}

double PM5_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M5_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	/*yes I know that KM5 lines are forbidden... */
	if (PK > 0.0) 
		rv += FluorYield(Z, K_SHELL, NULL) * PK * RadRate(Z, KM5_LINE, NULL);

	if (PL1 > 0.0)
		rv += FluorYield(Z, L1_SHELL, NULL) * PL1 * RadRate(Z, L1M5_LINE, NULL);

	if (PL2 > 0.0)
		rv += FluorYield(Z, L2_SHELL, NULL) * PL2 * RadRate(Z, L2M5_LINE, NULL);

	if (PL3 > 0.0)
		rv += FluorYield(Z, L3_SHELL, NULL) * PL3 * RadRate(Z, L3M5_LINE, NULL);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM15_TRANS, NULL) * PM1;
	
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z, FM25_TRANS, NULL) * PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z, FM35_TRANS, NULL) * PM3;

	if (PM4 > 0.0)
		rv += CosKronTransProb(Z, FM45_TRANS, NULL) * PM4;

	return rv;
}

double PM5_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M5_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0) 
		rv += PK * xrf_cross_sections_constants_auger_only[Z][M5_SHELL][K_SHELL];
	if (PL1 > 0.0)
		rv += PL1 * xrf_cross_sections_constants_auger_only[Z][M5_SHELL][L1_SHELL];
	if (PL2 > 0.0)
		rv += PL2 * xrf_cross_sections_constants_auger_only[Z][M5_SHELL][L2_SHELL];
	if (PL3 > 0.0)
		rv += PL3 * xrf_cross_sections_constants_auger_only[Z][M5_SHELL][L3_SHELL];
	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM15_TRANS, NULL) * PM1;
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z, FM25_TRANS, NULL) * PM2;
	if (PM3 > 0.0)	
		rv += CosKronTransProb(Z, FM35_TRANS, NULL) * PM3;
	if (PM4 > 0.0)	
		rv += CosKronTransProb(Z, FM45_TRANS, NULL) * PM4;

	return rv;
}

double PM5_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4, xrl_error **error) {
	double rv;

	rv = CS_Photo_Partial(Z, M5_SHELL, E, error);
	if (rv == 0.0)
		return 0.0;

	if (PK > 0.0) 
		rv += PK * xrf_cross_sections_constants_full[Z][M5_SHELL][K_SHELL];

	if (PL1 > 0.0)
		rv += PL1 * xrf_cross_sections_constants_full[Z][M5_SHELL][L1_SHELL];

	if (PL2 > 0.0)
		rv += PL2 * xrf_cross_sections_constants_full[Z][M5_SHELL][L2_SHELL];

	if (PL3 > 0.0)
		rv += PL3 * xrf_cross_sections_constants_full[Z][M5_SHELL][L3_SHELL];

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z, FM15_TRANS, NULL) * PM1;
	
	if (PM2 > 0.0)
		rv += CosKronTransProb(Z, FM25_TRANS, NULL) * PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z, FM35_TRANS, NULL) * PM3;

	if (PM4 > 0.0)
		rv += CosKronTransProb(Z, FM45_TRANS, NULL) * PM4;

	return rv;
}
