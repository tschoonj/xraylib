#include "xrf_cross_sections_aux.h"


float PL1_pure_kissel(int Z, float E) {
	return CS_Photo_Partial(Z, L1_SHELL, E);
}

float PL1_rad_cascade_kissel(int Z, float E, float PK) {
	//this function will probably not be very useful...
	float rv;
	rv = CS_Photo_Partial(Z,L1_SHELL, E);

	if (PK > 0.0 && RadRate(Z,KL1_LINE) > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KL1_LINE);

	return rv;
}

float PL1_auger_cascade_kissel(int Z, float E, float PK) {
	float rv;
	
	rv = CS_Photo_Partial(Z,L1_SHELL, E);
	if (PK > 0.0)
		rv += (1.0-FluorYield(Z,K_SHELL))*PK*(
	2.0*AugerRate(Z,K_L1L1_AUGER)+
	AugerRate(Z,K_L1L2_AUGER)+
	AugerRate(Z,K_L1L3_AUGER)+
	AugerRate(Z,K_L1M1_AUGER)+
	AugerRate(Z,K_L1M2_AUGER)+
	AugerRate(Z,K_L1M3_AUGER)+
	AugerRate(Z,K_L1M4_AUGER)+
	AugerRate(Z,K_L1M5_AUGER)+
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

float PL1_full_cascade_kissel(int Z, float E, float PK) {
	float rv;

	rv = CS_Photo_Partial(Z,L1_SHELL, E);
	if (PK > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KL1_LINE)+
	(1.0-FluorYield(Z,K_SHELL))*PK*(
	2.0*AugerRate(Z,K_L1L1_AUGER)+
	AugerRate(Z,K_L1L2_AUGER)+
	AugerRate(Z,K_L1L3_AUGER)+
	AugerRate(Z,K_L1M1_AUGER)+
	AugerRate(Z,K_L1M2_AUGER)+
	AugerRate(Z,K_L1M3_AUGER)+
	AugerRate(Z,K_L1M4_AUGER)+
	AugerRate(Z,K_L1M5_AUGER)+
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

float PL2_pure_kissel(int Z, float E, float PL1) {
	float rv;

	rv = CS_Photo_Partial(Z, L2_SHELL, E);
	if (PL1 > 0.0)
		rv +=CosKronTransProb(Z,FL12_TRANS)*PL1;
	return rv;	
}

float PL2_rad_cascade_kissel(int Z, float E, float PK, float PL1) {
	float rv;

	rv = CS_Photo_Partial(Z,L2_SHELL, E);
	if (PK > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KL2_LINE);

	if (PL1 > 0.0)
		rv += CosKronTransProb(Z,FL12_TRANS)*PL1;
	return  rv;
}

float PL2_auger_cascade_kissel(int Z, float E, float PK, float PL1) {
	float rv;

	rv = CS_Photo_Partial(Z,L2_SHELL, E);

	//K contributions
	if (PK > 0.0)
		rv += (1.0-FluorYield(Z,K_SHELL))*PK*(
	AugerRate(Z,K_L1L2_AUGER)+
	AugerRate(Z,K_L2L1_AUGER)+
	2.0*AugerRate(Z,K_L2L2_AUGER)+
	AugerRate(Z,K_L2L3_AUGER)+
	AugerRate(Z,K_L2M1_AUGER)+
	AugerRate(Z,K_L2M2_AUGER)+
	AugerRate(Z,K_L2M3_AUGER)+
	AugerRate(Z,K_L2M4_AUGER)+
	AugerRate(Z,K_L2M5_AUGER)+
	AugerRate(Z,K_L3L2_AUGER)+
	AugerRate(Z,K_M1L2_AUGER)+
	AugerRate(Z,K_M2L2_AUGER)+
	AugerRate(Z,K_M3L2_AUGER)+
	AugerRate(Z,K_M4L2_AUGER)+
	AugerRate(Z,K_M5L2_AUGER));

	//L1 contributions
	if (PL1 > 0.0)
		rv += (1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		2.0*AugerRate(Z,L1_L2L2_AUGER)+
		AugerRate(Z,L1_L2L3_AUGER)+
		AugerRate(Z,L1_L2M1_AUGER)+
		AugerRate(Z,L1_L2M2_AUGER)+
		AugerRate(Z,L1_L2M3_AUGER)+
		AugerRate(Z,L1_L2M4_AUGER)+
		AugerRate(Z,L1_L2M5_AUGER)+
		AugerRate(Z,L1_L3L2_AUGER)+
		AugerRate(Z,L1_M1L2_AUGER)+
		AugerRate(Z,L1_M2L2_AUGER)+
		AugerRate(Z,L1_M3L2_AUGER)+
		AugerRate(Z,L1_M4L2_AUGER)+
		AugerRate(Z,L1_M5L2_AUGER)
		)+
		CosKronTransProb(Z,FL12_TRANS)*PL1;
	return  rv;
	
}

float PL2_full_cascade_kissel(int Z, float E, float PK, float PL1) {
	float rv;

	rv = CS_Photo_Partial(Z,L2_SHELL, E);

	//K contributions
	if (PK > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KL2_LINE)+
		(1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1L2_AUGER)+
		AugerRate(Z,K_L2L1_AUGER)+
		2.0*AugerRate(Z,K_L2L2_AUGER)+
		AugerRate(Z,K_L2L3_AUGER)+
		AugerRate(Z,K_L2M1_AUGER)+
		AugerRate(Z,K_L2M2_AUGER)+
		AugerRate(Z,K_L2M3_AUGER)+
		AugerRate(Z,K_L2M4_AUGER)+
		AugerRate(Z,K_L2M5_AUGER)+
		AugerRate(Z,K_L3L2_AUGER)+
		AugerRate(Z,K_M1L2_AUGER)+
		AugerRate(Z,K_M2L2_AUGER)+
		AugerRate(Z,K_M3L2_AUGER)+
		AugerRate(Z,K_M4L2_AUGER)+
		AugerRate(Z,K_M5L2_AUGER)
		);
		
	//L1 contributions
	if (PL1 > 0.0)
		rv += (1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		2.0*AugerRate(Z,L1_L2L2_AUGER)+
		AugerRate(Z,L1_L2L3_AUGER)+
		AugerRate(Z,L1_L2M1_AUGER)+
		AugerRate(Z,L1_L2M2_AUGER)+
		AugerRate(Z,L1_L2M3_AUGER)+
		AugerRate(Z,L1_L2M4_AUGER)+
		AugerRate(Z,L1_L2M5_AUGER)+
		AugerRate(Z,L1_L3L2_AUGER)+
		AugerRate(Z,L1_M1L2_AUGER)+
		AugerRate(Z,L1_M2L2_AUGER)+
		AugerRate(Z,L1_M3L2_AUGER)+
		AugerRate(Z,L1_M4L2_AUGER)+
		AugerRate(Z,L1_M5L2_AUGER)
		)+
		CosKronTransProb(Z,FL12_TRANS)*PL1;
	return rv;
}

float PL3_pure_kissel(int Z, float E, float PL1, float PL2) {
	float rv;

	rv = CS_Photo_Partial(Z, L3_SHELL, E);

	if (PL1 > 0.0)
		rv += CosKronTransProb(Z,FL13_TRANS)*PL1;

	if (PL2 > 0.0)
		rv += CosKronTransProb(Z,FL23_TRANS)*PL2;


	return rv;
}

float PL3_rad_cascade_kissel(int Z, float E, float PK, float PL1, float PL2) {
	float rv;

	rv = CS_Photo_Partial(Z,L3_SHELL, E);

	if (PK > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KL3_LINE);

	if (PL1 > 0.0)
		rv += CosKronTransProb(Z,FL13_TRANS)*PL1;

	if (PL2 > 0.0)
		rv += CosKronTransProb(Z,FL23_TRANS)*PL2;

	return  rv;
}

float PL3_auger_cascade_kissel(int Z, float E, float PK, float PL1, float PL2) {
	float rv;

	rv = CS_Photo_Partial(Z,L3_SHELL, E);

	//K-shell
	if (PK > 0.0)
		rv += (1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1L3_AUGER)+
		AugerRate(Z,K_L2L3_AUGER)+
		AugerRate(Z,K_L3L1_AUGER)+
		AugerRate(Z,K_L3L2_AUGER)+
		2.0*AugerRate(Z,K_L3L3_AUGER)+
		AugerRate(Z,K_L3M1_AUGER)+
		AugerRate(Z,K_L3M2_AUGER)+
		AugerRate(Z,K_L3M3_AUGER)+
		AugerRate(Z,K_L3M4_AUGER)+
		AugerRate(Z,K_L3M5_AUGER)+
		AugerRate(Z,K_M1L3_AUGER)+
		AugerRate(Z,K_M2L3_AUGER)+
		AugerRate(Z,K_M3L3_AUGER)+
		AugerRate(Z,K_M4L3_AUGER)+
		AugerRate(Z,K_M5L3_AUGER)
		);

	//L1-shell
	if (PL1 > 0.0)
		rv += (1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		AugerRate(Z,L1_L2L3_AUGER)+
		AugerRate(Z,L1_L3L2_AUGER)+
		2.0*AugerRate(Z,L1_L3L3_AUGER)+
		AugerRate(Z,L1_L3M1_AUGER)+
		AugerRate(Z,L1_L3M2_AUGER)+
		AugerRate(Z,L1_L3M3_AUGER)+
		AugerRate(Z,L1_L3M4_AUGER)+
		AugerRate(Z,L1_L3M5_AUGER)+
		AugerRate(Z,L1_M1L3_AUGER)+
		AugerRate(Z,L1_M2L3_AUGER)+
		AugerRate(Z,L1_M3L3_AUGER)+
		AugerRate(Z,L1_M4L3_AUGER)+
		AugerRate(Z,L1_M5L3_AUGER)
		)+CosKronTransProb(Z,FL13_TRANS)*PL1;

	//L2-shell
	if (PL2 > 0.0)
		rv += (1.0-FluorYield(Z,L2_SHELL)-CosKronTransProb(Z,FL23_TRANS))*PL2*(
		2.0*AugerRate(Z,L2_L3L3_AUGER)+
		AugerRate(Z,L2_L3M1_AUGER)+
		AugerRate(Z,L2_L3M2_AUGER)+
		AugerRate(Z,L2_L3M3_AUGER)+
		AugerRate(Z,L2_L3M4_AUGER)+
		AugerRate(Z,L2_L3M5_AUGER)+
		AugerRate(Z,L2_M1L3_AUGER)+
		AugerRate(Z,L2_M2L3_AUGER)+
		AugerRate(Z,L2_M3L3_AUGER)+
		AugerRate(Z,L2_M4L3_AUGER)+
		AugerRate(Z,L2_M5L3_AUGER)
		)+CosKronTransProb(Z,FL23_TRANS)*PL2;


	return  rv;
}

float PL3_full_cascade_kissel(int Z, float E, float PK, float PL1, float PL2) {
	float rv;

	rv = CS_Photo_Partial(Z,L3_SHELL, E);

	//K-shell
	if (PK > 0.0)
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KL3_LINE)+
		(1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1L3_AUGER)+
		AugerRate(Z,K_L2L3_AUGER)+
		AugerRate(Z,K_L3L1_AUGER)+
		AugerRate(Z,K_L3L2_AUGER)+
		2.0*AugerRate(Z,K_L3L3_AUGER)+
		AugerRate(Z,K_L3M1_AUGER)+
		AugerRate(Z,K_L3M2_AUGER)+
		AugerRate(Z,K_L3M3_AUGER)+
		AugerRate(Z,K_L3M4_AUGER)+
		AugerRate(Z,K_L3M5_AUGER)+
		AugerRate(Z,K_M1L3_AUGER)+
		AugerRate(Z,K_M2L3_AUGER)+
		AugerRate(Z,K_M3L3_AUGER)+
		AugerRate(Z,K_M4L3_AUGER)+
		AugerRate(Z,K_M5L3_AUGER)
		);

	//L1-shell
	if (PL1 > 0.0)
		rv += (1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		AugerRate(Z,L1_L2L3_AUGER)+
		AugerRate(Z,L1_L3L2_AUGER)+
		2.0*AugerRate(Z,L1_L3L3_AUGER)+
		AugerRate(Z,L1_L3M1_AUGER)+
		AugerRate(Z,L1_L3M2_AUGER)+
		AugerRate(Z,L1_L3M3_AUGER)+
		AugerRate(Z,L1_L3M4_AUGER)+
		AugerRate(Z,L1_L3M5_AUGER)+
		AugerRate(Z,L1_M1L3_AUGER)+
		AugerRate(Z,L1_M2L3_AUGER)+
		AugerRate(Z,L1_M3L3_AUGER)+
		AugerRate(Z,L1_M4L3_AUGER)+
		AugerRate(Z,L1_M5L3_AUGER)
		)+CosKronTransProb(Z,FL13_TRANS)*PL1;

	//L2-shell
	if (PL2 > 0.0)
		rv += (1.0-FluorYield(Z,L2_SHELL)-CosKronTransProb(Z,FL23_TRANS))*PL2*(
		2.0*AugerRate(Z,L2_L3L3_AUGER)+
		AugerRate(Z,L2_L3M1_AUGER)+
		AugerRate(Z,L2_L3M2_AUGER)+
		AugerRate(Z,L2_L3M3_AUGER)+
		AugerRate(Z,L2_L3M4_AUGER)+
		AugerRate(Z,L2_L3M5_AUGER)+
		AugerRate(Z,L2_M1L3_AUGER)+
		AugerRate(Z,L2_M2L3_AUGER)+
		AugerRate(Z,L2_M3L3_AUGER)+
		AugerRate(Z,L2_M4L3_AUGER)+
		AugerRate(Z,L2_M5L3_AUGER)
		)+CosKronTransProb(Z,FL23_TRANS)*PL2;

	return rv;
}

float PM1_pure_kissel(int Z, float E) {
	return CS_Photo_Partial(Z, M1_SHELL, E);
}

float PM1_rad_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3) {
	float rv;

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

float PM1_auger_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3) {
	float rv;

	rv = CS_Photo_Partial(Z, M1_SHELL, E);

	//first whatever is due to K-shell excitations
	if (PK > 0.0)
		rv += (1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1M1_AUGER)+
		AugerRate(Z,K_L2M1_AUGER)+
		AugerRate(Z,K_L3M1_AUGER)+
		AugerRate(Z,K_M1L1_AUGER)+
		AugerRate(Z,K_M1L2_AUGER)+
		AugerRate(Z,K_M1L3_AUGER)+
		2.0*AugerRate(Z,K_M1M1_AUGER)+
		AugerRate(Z,K_M1M2_AUGER)+
		AugerRate(Z,K_M1M3_AUGER)+
		AugerRate(Z,K_M1M4_AUGER)+
		AugerRate(Z,K_M1M5_AUGER)+
		AugerRate(Z,K_M2M1_AUGER)+
		AugerRate(Z,K_M3M1_AUGER)+
		AugerRate(Z,K_M4M1_AUGER)+
		AugerRate(Z,K_M5M1_AUGER)
		);
	
	//followed by L1
	if (PL1 > 0.0)
		rv += (1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		AugerRate(Z,L1_L2M1_AUGER)+
		AugerRate(Z,L1_L3M1_AUGER)+
		AugerRate(Z,L1_M1L2_AUGER)+
		AugerRate(Z,L1_M1L3_AUGER)+
		2.0*AugerRate(Z,L1_M1M1_AUGER)+
		AugerRate(Z,L1_M1M2_AUGER)+
		AugerRate(Z,L1_M1M3_AUGER)+
		AugerRate(Z,L1_M1M4_AUGER)+
		AugerRate(Z,L1_M1M5_AUGER)+
		AugerRate(Z,L1_M2M1_AUGER)+
		AugerRate(Z,L1_M3M1_AUGER)+
		AugerRate(Z,L1_M4M1_AUGER)+
		AugerRate(Z,L1_M5M1_AUGER)
		);

	//next is L2
	if (PL2 > 0.0) 
		rv += (1.0-FluorYield(Z,L2_SHELL)-CosKronTransProb(Z,FL23_TRANS))*PL2*(
		AugerRate(Z,L2_L3M1_AUGER)+
		AugerRate(Z,L2_M1L3_AUGER)+
		2.0*AugerRate(Z,L2_M1M1_AUGER)+
		AugerRate(Z,L2_M1M2_AUGER)+
		AugerRate(Z,L2_M1M3_AUGER)+
		AugerRate(Z,L2_M1M4_AUGER)+
		AugerRate(Z,L2_M1M5_AUGER)+
		AugerRate(Z,L2_M2L3_AUGER)+
		AugerRate(Z,L2_M2M1_AUGER)+
		AugerRate(Z,L2_M3M1_AUGER)+
		AugerRate(Z,L2_M4M1_AUGER)+
		AugerRate(Z,L2_M5M1_AUGER)
		);
	
	//...L3
	if (PL3 > 0.0)
		rv += (1.0-FluorYield(Z,L3_SHELL))*PL3*(
		2.0*AugerRate(Z,L3_M1M1_AUGER)+
		AugerRate(Z,L3_M1M2_AUGER)+
		AugerRate(Z,L3_M1M3_AUGER)+
		AugerRate(Z,L3_M1M4_AUGER)+
		AugerRate(Z,L3_M1M5_AUGER)+
		AugerRate(Z,L3_M2M1_AUGER)+
		AugerRate(Z,L3_M3M1_AUGER)+
		AugerRate(Z,L3_M4M1_AUGER)+
		AugerRate(Z,L3_M5M1_AUGER)
		);
	return rv;
}

float PM1_full_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3) {
	float rv;

	rv = CS_Photo_Partial(Z, M1_SHELL, E);

	//first whatever is due to K-shell excitations
	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM1_LINE)+
		(1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1M1_AUGER)+
		AugerRate(Z,K_L2M1_AUGER)+
		AugerRate(Z,K_L3M1_AUGER)+
		AugerRate(Z,K_M1L1_AUGER)+
		AugerRate(Z,K_M1L2_AUGER)+
		AugerRate(Z,K_M1L3_AUGER)+
		2.0*AugerRate(Z,K_M1M1_AUGER)+
		AugerRate(Z,K_M1M2_AUGER)+
		AugerRate(Z,K_M1M3_AUGER)+
		AugerRate(Z,K_M1M4_AUGER)+
		AugerRate(Z,K_M1M5_AUGER)+
		AugerRate(Z,K_M2M1_AUGER)+
		AugerRate(Z,K_M3M1_AUGER)+
		AugerRate(Z,K_M4M1_AUGER)+
		AugerRate(Z,K_M5M1_AUGER)
		);
		

	//followed by L1
	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M1_LINE)+
		(1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		AugerRate(Z,L1_L2M1_AUGER)+
		AugerRate(Z,L1_L3M1_AUGER)+
		AugerRate(Z,L1_M1L2_AUGER)+
		AugerRate(Z,L1_M1L3_AUGER)+
		2.0*AugerRate(Z,L1_M1M1_AUGER)+
		AugerRate(Z,L1_M1M2_AUGER)+
		AugerRate(Z,L1_M1M3_AUGER)+
		AugerRate(Z,L1_M1M4_AUGER)+
		AugerRate(Z,L1_M1M5_AUGER)+
		AugerRate(Z,L1_M2M1_AUGER)+
		AugerRate(Z,L1_M3M1_AUGER)+
		AugerRate(Z,L1_M4M1_AUGER)+
		AugerRate(Z,L1_M5M1_AUGER)
		);
	
	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M1_LINE)+
		(1.0-FluorYield(Z,L2_SHELL)-CosKronTransProb(Z,FL23_TRANS))*PL2*(
		AugerRate(Z,L2_L3M1_AUGER)+
		AugerRate(Z,L2_M1L3_AUGER)+
		2.0*AugerRate(Z,L2_M1M1_AUGER)+
		AugerRate(Z,L2_M1M2_AUGER)+
		AugerRate(Z,L2_M1M3_AUGER)+
		AugerRate(Z,L2_M1M4_AUGER)+
		AugerRate(Z,L2_M1M5_AUGER)+
		AugerRate(Z,L2_M2L3_AUGER)+
		AugerRate(Z,L2_M2M1_AUGER)+
		AugerRate(Z,L2_M3M1_AUGER)+
		AugerRate(Z,L2_M4M1_AUGER)+
		AugerRate(Z,L2_M5M1_AUGER)
		);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M1_LINE)+
		(1.0-FluorYield(Z,L3_SHELL))*PL3*(
		2.0*AugerRate(Z,L3_M1M1_AUGER)+
		AugerRate(Z,L3_M1M2_AUGER)+
		AugerRate(Z,L3_M1M3_AUGER)+
		AugerRate(Z,L3_M1M4_AUGER)+
		AugerRate(Z,L3_M1M5_AUGER)+
		AugerRate(Z,L3_M2M1_AUGER)+
		AugerRate(Z,L3_M3M1_AUGER)+
		AugerRate(Z,L3_M4M1_AUGER)+
		AugerRate(Z,L3_M5M1_AUGER));


	return rv;
}


float PM2_pure_kissel(int Z, float E, float PM1) {
	float rv;

	rv = CS_Photo_Partial(Z, M2_SHELL, E);
	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM12_TRANS)*PM1;
		
	return rv; 
}

float PM2_rad_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1) {
	float rv;

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

float PM2_auger_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1) {
	float rv;

	rv = CS_Photo_Partial(Z, M2_SHELL, E);

	if (PK > 0.0)
		rv += (1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1M2_AUGER)+
		AugerRate(Z,K_L2M2_AUGER)+
		AugerRate(Z,K_L3M2_AUGER)+
		AugerRate(Z,K_M1M2_AUGER)+
		AugerRate(Z,K_M3M2_AUGER)+
		AugerRate(Z,K_M4M2_AUGER)+
		AugerRate(Z,K_M5M2_AUGER)+
		AugerRate(Z,K_M2L1_AUGER)+
		AugerRate(Z,K_M2L2_AUGER)+
		AugerRate(Z,K_M2L3_AUGER)+
		AugerRate(Z,K_M2M1_AUGER)+
		2.0*AugerRate(Z,K_M2M2_AUGER)+
		AugerRate(Z,K_M2M3_AUGER)+
		AugerRate(Z,K_M2M4_AUGER)+
		AugerRate(Z,K_M2M5_AUGER)
		);
	if (PL1 > 0.0)
		rv += (1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		AugerRate(Z,L1_L2M2_AUGER)+
		AugerRate(Z,L1_L3M2_AUGER)+
		AugerRate(Z,L1_M1M2_AUGER)+
		AugerRate(Z,L1_M2L2_AUGER)+
		AugerRate(Z,L1_M2L3_AUGER)+
		AugerRate(Z,L1_M2M1_AUGER)+
		2.0*AugerRate(Z,L1_M2M2_AUGER)+
		AugerRate(Z,L1_M2M3_AUGER)+
		AugerRate(Z,L1_M2M4_AUGER)+
		AugerRate(Z,L1_M2M5_AUGER)+
		AugerRate(Z,L1_M3M2_AUGER)+
		AugerRate(Z,L1_M4M2_AUGER)+
		AugerRate(Z,L1_M5M2_AUGER)
		);

	//L2
	if (PL2 > 0.0)
		rv += (1.0-FluorYield(Z,L2_SHELL)-CosKronTransProb(Z,FL23_TRANS))*PL2*(
		AugerRate(Z,L2_L3M2_AUGER)+
		AugerRate(Z,L2_M1M2_AUGER)+
		AugerRate(Z,L2_M2L3_AUGER)+
		AugerRate(Z,L2_M2M1_AUGER)+
		2.0*AugerRate(Z,L2_M2M2_AUGER)+
		AugerRate(Z,L2_M2M3_AUGER)+
		AugerRate(Z,L2_M2M4_AUGER)+
		AugerRate(Z,L2_M2M5_AUGER)+
		AugerRate(Z,L2_M3L3_AUGER)+
		AugerRate(Z,L2_M3M1_AUGER)+
		AugerRate(Z,L2_M3M2_AUGER)+
		AugerRate(Z,L2_M4M2_AUGER)+
		AugerRate(Z,L2_M5M2_AUGER)
		);
	//L3
	if (PL3 > 0.0)
		rv += (1.0-FluorYield(Z,L3_SHELL))*PL3*(
		2.0*AugerRate(Z,L3_M1M1_AUGER)+
		AugerRate(Z,L3_M1M2_AUGER)+
		AugerRate(Z,L3_M1M3_AUGER)+
		AugerRate(Z,L3_M1M4_AUGER)+
		AugerRate(Z,L3_M1M5_AUGER)+
		AugerRate(Z,L3_M2M1_AUGER)+
		AugerRate(Z,L3_M3M1_AUGER)+
		AugerRate(Z,L3_M4M1_AUGER)+
		AugerRate(Z,L3_M5M1_AUGER)
		);

	//M1
	if (PM1 > 0.0)
		rv += (1.0-FluorYield(Z,M1_SHELL)-CosKronTransProb(Z,FM12_TRANS)-CosKronTransProb(Z,FM13_TRANS)-CosKronTransProb(Z,FM14_TRANS)-CosKronTransProb(Z,FM15_TRANS))*PM1*(
		2.0*AugerRate(Z,M1_M2M2_AUGER)+
		AugerRate(Z,M1_M2M3_AUGER)+
		AugerRate(Z,M1_M2M4_AUGER)+
		AugerRate(Z,M1_M2M5_AUGER)+
		AugerRate(Z,M1_M3M2_AUGER)+
		AugerRate(Z,M1_M4M2_AUGER)+
		AugerRate(Z,M1_M5M2_AUGER)
		)
		+CosKronTransProb(Z,FM12_TRANS)*PM1;

	return rv;
}

float PM2_full_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1) {
	float rv;

	rv = CS_Photo_Partial(Z, M2_SHELL, E);

	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM2_LINE)+
		(1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1M2_AUGER)+
		AugerRate(Z,K_L2M2_AUGER)+
		AugerRate(Z,K_L3M2_AUGER)+
		AugerRate(Z,K_M1M2_AUGER)+
		AugerRate(Z,K_M3M2_AUGER)+
		AugerRate(Z,K_M4M2_AUGER)+
		AugerRate(Z,K_M5M2_AUGER)+
		AugerRate(Z,K_M2L1_AUGER)+
		AugerRate(Z,K_M2L2_AUGER)+
		AugerRate(Z,K_M2L3_AUGER)+
		AugerRate(Z,K_M2M1_AUGER)+
		2.0*AugerRate(Z,K_M2M2_AUGER)+
		AugerRate(Z,K_M2M3_AUGER)+
		AugerRate(Z,K_M2M4_AUGER)+
		AugerRate(Z,K_M2M5_AUGER)
		);

	if (PL1 > 0.0) 
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M2_LINE)+
		(1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		AugerRate(Z,L1_L2M2_AUGER)+
		AugerRate(Z,L1_L3M2_AUGER)+
		AugerRate(Z,L1_M1M2_AUGER)+
		AugerRate(Z,L1_M2L2_AUGER)+
		AugerRate(Z,L1_M2L3_AUGER)+
		AugerRate(Z,L1_M2M1_AUGER)+
		2.0*AugerRate(Z,L1_M2M2_AUGER)+
		AugerRate(Z,L1_M2M3_AUGER)+
		AugerRate(Z,L1_M2M4_AUGER)+
		AugerRate(Z,L1_M2M5_AUGER)+
		AugerRate(Z,L1_M3M2_AUGER)+
		AugerRate(Z,L1_M4M2_AUGER)+
		AugerRate(Z,L1_M5M2_AUGER)
		);
	
	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M2_LINE)+
		(1.0-FluorYield(Z,L2_SHELL)-CosKronTransProb(Z,FL23_TRANS))*PL2*(
		AugerRate(Z,L2_L3M2_AUGER)+
		AugerRate(Z,L2_M1M2_AUGER)+
		AugerRate(Z,L2_M2L3_AUGER)+
		AugerRate(Z,L2_M2M1_AUGER)+
		2.0*AugerRate(Z,L2_M2M2_AUGER)+
		AugerRate(Z,L2_M2M3_AUGER)+
		AugerRate(Z,L2_M2M4_AUGER)+
		AugerRate(Z,L2_M2M5_AUGER)+
		AugerRate(Z,L2_M3L3_AUGER)+
		AugerRate(Z,L2_M3M1_AUGER)+
		AugerRate(Z,L2_M3M2_AUGER)+
		AugerRate(Z,L2_M4M2_AUGER)+
		AugerRate(Z,L2_M5M2_AUGER)
		);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M2_LINE) +
		(1.0-FluorYield(Z,L3_SHELL))*PL3*(
		2.0*AugerRate(Z,L3_M1M1_AUGER)+
		AugerRate(Z,L3_M1M2_AUGER)+
		AugerRate(Z,L3_M1M3_AUGER)+
		AugerRate(Z,L3_M1M4_AUGER)+
		AugerRate(Z,L3_M1M5_AUGER)+
		AugerRate(Z,L3_M2M1_AUGER)+
		AugerRate(Z,L3_M3M1_AUGER)+
		AugerRate(Z,L3_M4M1_AUGER)+
		AugerRate(Z,L3_M5M1_AUGER)
		);

	if (PM1 > 0.0)
		rv += (1.0-FluorYield(Z,M1_SHELL)-CosKronTransProb(Z,FM12_TRANS)-CosKronTransProb(Z,FM13_TRANS)-CosKronTransProb(Z,FM14_TRANS)-CosKronTransProb(Z,FM15_TRANS))*PM1*(
		2.0*AugerRate(Z,M1_M2M2_AUGER)+
		AugerRate(Z,M1_M2M3_AUGER)+
		AugerRate(Z,M1_M2M4_AUGER)+
		AugerRate(Z,M1_M2M5_AUGER)+
		AugerRate(Z,M1_M3M2_AUGER)+
		AugerRate(Z,M1_M4M2_AUGER)+
		AugerRate(Z,M1_M5M2_AUGER)
		)+	
		CosKronTransProb(Z,FM12_TRANS)*PM1;

	return rv;
}

float PM3_pure_kissel(int Z, float E, float PM1, float PM2) {
	float rv;

	rv = CS_Photo_Partial(Z, M3_SHELL, E);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM13_TRANS)*PM1;

	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM23_TRANS)*PM2;

	return rv;
}

float PM3_rad_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2) {
	float rv;

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

float PM3_auger_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2) {
	float rv;

	rv = CS_Photo_Partial(Z, M3_SHELL, E);

	if (PK > 0.0)
		rv += (1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1M3_AUGER)+
		AugerRate(Z,K_L2M3_AUGER)+
		AugerRate(Z,K_L3M3_AUGER)+
		AugerRate(Z,K_M1M3_AUGER)+
		AugerRate(Z,K_M2M3_AUGER)+
		2.0*AugerRate(Z,K_M3M3_AUGER)+
		AugerRate(Z,K_M4M3_AUGER)+
		AugerRate(Z,K_M5M3_AUGER)+
		AugerRate(Z,K_M3L1_AUGER)+
		AugerRate(Z,K_M3L2_AUGER)+
		AugerRate(Z,K_M3L3_AUGER)+
		AugerRate(Z,K_M3M1_AUGER)+
		AugerRate(Z,K_M3M2_AUGER)+
		AugerRate(Z,K_M3M4_AUGER)+
		AugerRate(Z,K_M3M5_AUGER)
		);
	if (PL1 > 0.0)
		rv += (1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		AugerRate(Z,L1_L2M3_AUGER)+
		AugerRate(Z,L1_L3M3_AUGER)+
		AugerRate(Z,L1_M1M3_AUGER)+
		AugerRate(Z,L1_M2M3_AUGER)+
		AugerRate(Z,L1_M3L2_AUGER)+
		AugerRate(Z,L1_M3L3_AUGER)+
		AugerRate(Z,L1_M3M1_AUGER)+
		AugerRate(Z,L1_M3M2_AUGER)+
		2.0*AugerRate(Z,L1_M3M3_AUGER)+
		AugerRate(Z,L1_M3M4_AUGER)+
		AugerRate(Z,L1_M3M5_AUGER)+
		AugerRate(Z,L1_M4M3_AUGER)+
		AugerRate(Z,L1_M5M3_AUGER)
		);
	if (PL2 > 0.0)
		rv += (1.0-FluorYield(Z,L2_SHELL)-CosKronTransProb(Z,FL23_TRANS))*PL2*(
		AugerRate(Z,L2_L3M3_AUGER)+
		AugerRate(Z,L2_M1M3_AUGER)+
		AugerRate(Z,L2_M2M3_AUGER)+
		AugerRate(Z,L2_M3L3_AUGER)+
		AugerRate(Z,L2_M3M1_AUGER)+
		AugerRate(Z,L2_M3M2_AUGER)+
		2.0*AugerRate(Z,L2_M3M3_AUGER)+
		AugerRate(Z,L2_M3M4_AUGER)+
		AugerRate(Z,L2_M3M5_AUGER)+
		AugerRate(Z,L2_M4M3_AUGER)+
		AugerRate(Z,L2_M5M3_AUGER)
		);
	if (PL3 > 0.0) 
		rv += (1.0-FluorYield(Z,L3_SHELL))*PL3*(
		AugerRate(Z,L3_M1M3_AUGER)+
		AugerRate(Z,L3_M2M3_AUGER)+
		AugerRate(Z,L3_M3M1_AUGER)+
		AugerRate(Z,L3_M3M2_AUGER)+
		2.0*AugerRate(Z,L3_M3M3_AUGER)+
		AugerRate(Z,L3_M3M4_AUGER)+
		AugerRate(Z,L3_M3M5_AUGER)+
		AugerRate(Z,L3_M4M3_AUGER)+
		AugerRate(Z,L3_M5M3_AUGER)
		);
	if (PM1 > 0.0)
		rv += (1.0-FluorYield(Z,M1_SHELL)-CosKronTransProb(Z,FM12_TRANS)-CosKronTransProb(Z,FM13_TRANS)-CosKronTransProb(Z,FM14_TRANS)-CosKronTransProb(Z,FM15_TRANS))*PM1*(
		AugerRate(Z,M1_M2M3_AUGER)+
		AugerRate(Z,M1_M3M2_AUGER)+
		2.0*AugerRate(Z,M1_M3M3_AUGER)+
		AugerRate(Z,M1_M3M4_AUGER)+
		AugerRate(Z,M1_M3M5_AUGER)+
		AugerRate(Z,M1_M4M3_AUGER)+
		AugerRate(Z,M1_M5M3_AUGER)
		)
		+CosKronTransProb(Z,FM13_TRANS)*PM1;
	if (PM2 > 0.0)
		rv += (1.0-FluorYield(Z,M2_SHELL)-CosKronTransProb(Z,FM23_TRANS)-CosKronTransProb(Z,FM24_TRANS)-CosKronTransProb(Z,FM25_TRANS))*PM2*(
		2.0*AugerRate(Z,M2_M3M3_AUGER)+
		AugerRate(Z,M2_M3M4_AUGER)+
		AugerRate(Z,M2_M3M5_AUGER)+
		AugerRate(Z,M2_M4M3_AUGER)+
		AugerRate(Z,M2_M5M3_AUGER)
		) 
		+CosKronTransProb(Z,FM23_TRANS)*PM2;

	return rv;
}

float PM3_full_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2) {
	float rv;

	rv = CS_Photo_Partial(Z, M3_SHELL, E);

	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM3_LINE)+
		(1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1M3_AUGER)+
		AugerRate(Z,K_L2M3_AUGER)+
		AugerRate(Z,K_L3M3_AUGER)+
		AugerRate(Z,K_M1M3_AUGER)+
		AugerRate(Z,K_M2M3_AUGER)+
		2.0*AugerRate(Z,K_M3M3_AUGER)+
		AugerRate(Z,K_M4M3_AUGER)+
		AugerRate(Z,K_M5M3_AUGER)+
		AugerRate(Z,K_M3L1_AUGER)+
		AugerRate(Z,K_M3L2_AUGER)+
		AugerRate(Z,K_M3L3_AUGER)+
		AugerRate(Z,K_M3M1_AUGER)+
		AugerRate(Z,K_M3M2_AUGER)+
		AugerRate(Z,K_M3M4_AUGER)+
		AugerRate(Z,K_M3M5_AUGER)
		);

	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M3_LINE)+
		(1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		AugerRate(Z,L1_L2M3_AUGER)+
		AugerRate(Z,L1_L3M3_AUGER)+
		AugerRate(Z,L1_M1M3_AUGER)+
		AugerRate(Z,L1_M2M3_AUGER)+
		AugerRate(Z,L1_M3L2_AUGER)+
		AugerRate(Z,L1_M3L3_AUGER)+
		AugerRate(Z,L1_M3M1_AUGER)+
		AugerRate(Z,L1_M3M2_AUGER)+
		2.0*AugerRate(Z,L1_M3M3_AUGER)+
		AugerRate(Z,L1_M3M4_AUGER)+
		AugerRate(Z,L1_M3M5_AUGER)+
		AugerRate(Z,L1_M4M3_AUGER)+
		AugerRate(Z,L1_M5M3_AUGER)
		);

	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M3_LINE)+
		(1.0-FluorYield(Z,L2_SHELL)-CosKronTransProb(Z,FL23_TRANS))*PL2*(
		AugerRate(Z,L2_L3M3_AUGER)+
		AugerRate(Z,L2_M1M3_AUGER)+
		AugerRate(Z,L2_M2M3_AUGER)+
		AugerRate(Z,L2_M3L3_AUGER)+
		AugerRate(Z,L2_M3M1_AUGER)+
		AugerRate(Z,L2_M3M2_AUGER)+
		2.0*AugerRate(Z,L2_M3M3_AUGER)+
		AugerRate(Z,L2_M3M4_AUGER)+
		AugerRate(Z,L2_M3M5_AUGER)+
		AugerRate(Z,L2_M4M3_AUGER)+
		AugerRate(Z,L2_M5M3_AUGER)
		);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M3_LINE)+
		(1.0-FluorYield(Z,L3_SHELL))*PL3*(
		AugerRate(Z,L3_M1M3_AUGER)+
		AugerRate(Z,L3_M2M3_AUGER)+
		AugerRate(Z,L3_M3M1_AUGER)+
		AugerRate(Z,L3_M3M2_AUGER)+
		2.0*AugerRate(Z,L3_M3M3_AUGER)+
		AugerRate(Z,L3_M3M4_AUGER)+
		AugerRate(Z,L3_M3M5_AUGER)+
		AugerRate(Z,L3_M4M3_AUGER)+
		AugerRate(Z,L3_M5M3_AUGER)
		);

	if (PM1 > 0.0)
		rv += (1.0-FluorYield(Z,M1_SHELL)-CosKronTransProb(Z,FM12_TRANS)-CosKronTransProb(Z,FM13_TRANS)-CosKronTransProb(Z,FM14_TRANS)-CosKronTransProb(Z,FM15_TRANS))*PM1*(
		AugerRate(Z,M1_M2M3_AUGER)+
		AugerRate(Z,M1_M3M2_AUGER)+
		2.0*AugerRate(Z,M1_M3M3_AUGER)+
		AugerRate(Z,M1_M3M4_AUGER)+
		AugerRate(Z,M1_M3M5_AUGER)+
		AugerRate(Z,M1_M4M3_AUGER)+
		AugerRate(Z,M1_M5M3_AUGER)
		)
		+CosKronTransProb(Z,FM13_TRANS)*PM1;
	
	if (PM2 > 0.0)
		rv += (1.0-FluorYield(Z,M2_SHELL)-CosKronTransProb(Z,FM23_TRANS)-CosKronTransProb(Z,FM24_TRANS)-CosKronTransProb(Z,FM25_TRANS))*PM2*(
		2.0*AugerRate(Z,M2_M3M3_AUGER)+
		AugerRate(Z,M2_M3M4_AUGER)+
		AugerRate(Z,M2_M3M5_AUGER)+
		AugerRate(Z,M2_M4M3_AUGER)+
		AugerRate(Z,M2_M5M3_AUGER)
		) 
		+CosKronTransProb(Z,FM23_TRANS)*PM2;

	return rv;
}

float PM4_pure_kissel(int Z, float E, float PM1, float PM2, float PM3) {
	float rv;

	rv = CS_Photo_Partial(Z, M4_SHELL, E);

	if (PM1 > 0.0)
		rv += CosKronTransProb(Z,FM14_TRANS)*PM1;

	if (PM2 > 0.0)
		rv += CosKronTransProb(Z,FM24_TRANS)*PM2;

	if (PM3 > 0.0)
		rv += CosKronTransProb(Z,FM34_TRANS)*PM3;

	return rv;
}

float PM4_rad_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2, float PM3) {
	float rv;

	rv = CS_Photo_Partial(Z, M4_SHELL, E);

	//yes I know that KM4 lines are forbidden...
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

float PM4_auger_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2, float PM3) {
	float rv;

	rv = CS_Photo_Partial(Z, M4_SHELL, E);

	if (PK > 0.0) 
		rv += (1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1M4_AUGER)+
		AugerRate(Z,K_L2M4_AUGER)+
		AugerRate(Z,K_L3M4_AUGER)+
		AugerRate(Z,K_M1M4_AUGER)+
		AugerRate(Z,K_M2M4_AUGER)+
		AugerRate(Z,K_M3M4_AUGER)+
		2.0*AugerRate(Z,K_M4M4_AUGER)+
		AugerRate(Z,K_M5M4_AUGER)+
		AugerRate(Z,K_M4L1_AUGER)+
		AugerRate(Z,K_M4L2_AUGER)+
		AugerRate(Z,K_M4L3_AUGER)+
		AugerRate(Z,K_M4M1_AUGER)+
		AugerRate(Z,K_M4M2_AUGER)+
		AugerRate(Z,K_M4M3_AUGER)+
		AugerRate(Z,K_M4M5_AUGER)
		);
	if (PL1 > 0.0)
		rv += (1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		AugerRate(Z,L1_L2M4_AUGER)+
		AugerRate(Z,L1_L3M4_AUGER)+
		AugerRate(Z,L1_M1M4_AUGER)+
		AugerRate(Z,L1_M2M4_AUGER)+
		AugerRate(Z,L1_M3M4_AUGER)+
		2.0*AugerRate(Z,L1_M4M4_AUGER)+
		AugerRate(Z,L1_M5M4_AUGER)+
		AugerRate(Z,L1_M4L2_AUGER)+
		AugerRate(Z,L1_M4L3_AUGER)+
		AugerRate(Z,L1_M4M1_AUGER)+
		AugerRate(Z,L1_M4M2_AUGER)+
		AugerRate(Z,L1_M4M3_AUGER)+
		AugerRate(Z,L1_M4M5_AUGER)
		);
	if (PL2 > 0.0)
		rv += (1.0-FluorYield(Z,L2_SHELL)-CosKronTransProb(Z,FL23_TRANS))*PL2*(
		AugerRate(Z,L2_L3M4_AUGER)+
		AugerRate(Z,L2_M1M4_AUGER)+
		AugerRate(Z,L2_M2M4_AUGER)+
		AugerRate(Z,L2_M3M4_AUGER)+
		2.0*AugerRate(Z,L2_M4M4_AUGER)+
		AugerRate(Z,L2_M5M4_AUGER)+
		AugerRate(Z,L2_M4L3_AUGER)+
		AugerRate(Z,L2_M4M1_AUGER)+
		AugerRate(Z,L2_M4M2_AUGER)+
		AugerRate(Z,L2_M4M3_AUGER)+
		AugerRate(Z,L2_M4M5_AUGER)
		);
	if (PL3 > 0.0)
		rv += (1.0-FluorYield(Z,L3_SHELL))*PL3*(
		AugerRate(Z,L3_M1M4_AUGER)+
		AugerRate(Z,L3_M2M4_AUGER)+
		AugerRate(Z,L3_M3M4_AUGER)+
		2.0*AugerRate(Z,L3_M4M4_AUGER)+
		AugerRate(Z,L3_M5M4_AUGER)+
		AugerRate(Z,L3_M4M1_AUGER)+
		AugerRate(Z,L3_M4M2_AUGER)+
		AugerRate(Z,L3_M4M3_AUGER)+
		AugerRate(Z,L3_M4M5_AUGER)
		);
	if (PM1 > 0.0)
		rv += (1.0-FluorYield(Z,M1_SHELL)-CosKronTransProb(Z,FM12_TRANS)-CosKronTransProb(Z,FM13_TRANS)-CosKronTransProb(Z,FM14_TRANS)-CosKronTransProb(Z,FM15_TRANS))*PM1*(
		AugerRate(Z,M1_M2M4_AUGER)+
		AugerRate(Z,M1_M3M4_AUGER)+
		AugerRate(Z,M1_M4M2_AUGER)+
		AugerRate(Z,M1_M4M3_AUGER)+
		2.0*AugerRate(Z,M1_M4M4_AUGER)+
		AugerRate(Z,M1_M4M5_AUGER)+
		AugerRate(Z,M1_M5M4_AUGER)
		)
		+CosKronTransProb(Z,FM14_TRANS)*PM1;

	if (PM2 > 0.0)
		rv += (1.0-FluorYield(Z,M2_SHELL)-CosKronTransProb(Z,FM23_TRANS)-CosKronTransProb(Z,FM24_TRANS)-CosKronTransProb(Z,FM25_TRANS))*PM2*(
		AugerRate(Z,M2_M3M4_AUGER)+
		AugerRate(Z,M2_M3M5_AUGER)+
		AugerRate(Z,M2_M4M3_AUGER)+
		2.0*AugerRate(Z,M2_M4M4_AUGER)+
		AugerRate(Z,M2_M4M5_AUGER)+
		AugerRate(Z,M2_M5M3_AUGER)+
		AugerRate(Z,M2_M5M4_AUGER)
		)
		+CosKronTransProb(Z,FM24_TRANS)*PM2;

	if (PM3 > 0.0)	
		rv += (1.0-FluorYield(Z,M3_SHELL)-CosKronTransProb(Z,FM34_TRANS)-CosKronTransProb(Z,FM35_TRANS))*PM3*(
		2.0*AugerRate(Z,M3_M4M4_AUGER)+
		AugerRate(Z,M3_M4M5_AUGER)+
		AugerRate(Z,M3_M5M4_AUGER)
		)
		+CosKronTransProb(Z,FM34_TRANS)*PM3;

	return rv;
}

float PM4_full_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2, float PM3) {
	float rv;

	rv = CS_Photo_Partial(Z, M4_SHELL, E);

	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM4_LINE)+
		(1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1M4_AUGER)+
		AugerRate(Z,K_L2M4_AUGER)+
		AugerRate(Z,K_L3M4_AUGER)+
		AugerRate(Z,K_M1M4_AUGER)+
		AugerRate(Z,K_M2M4_AUGER)+
		AugerRate(Z,K_M3M4_AUGER)+
		2.0*AugerRate(Z,K_M4M4_AUGER)+
		AugerRate(Z,K_M5M4_AUGER)+
		AugerRate(Z,K_M4L1_AUGER)+
		AugerRate(Z,K_M4L2_AUGER)+
		AugerRate(Z,K_M4L3_AUGER)+
		AugerRate(Z,K_M4M1_AUGER)+
		AugerRate(Z,K_M4M2_AUGER)+
		AugerRate(Z,K_M4M3_AUGER)+
		AugerRate(Z,K_M4M5_AUGER)
		);

	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M4_LINE)+
		(1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		AugerRate(Z,L1_L2M4_AUGER)+
		AugerRate(Z,L1_L3M4_AUGER)+
		AugerRate(Z,L1_M1M4_AUGER)+
		AugerRate(Z,L1_M2M4_AUGER)+
		AugerRate(Z,L1_M3M4_AUGER)+
		2.0*AugerRate(Z,L1_M4M4_AUGER)+
		AugerRate(Z,L1_M5M4_AUGER)+
		AugerRate(Z,L1_M4L2_AUGER)+
		AugerRate(Z,L1_M4L3_AUGER)+
		AugerRate(Z,L1_M4M1_AUGER)+
		AugerRate(Z,L1_M4M2_AUGER)+
		AugerRate(Z,L1_M4M3_AUGER)+
		AugerRate(Z,L1_M4M5_AUGER)
		);

	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M4_LINE)+
		(1.0-FluorYield(Z,L2_SHELL)-CosKronTransProb(Z,FL23_TRANS))*PL2*(
		AugerRate(Z,L2_L3M4_AUGER)+
		AugerRate(Z,L2_M1M4_AUGER)+
		AugerRate(Z,L2_M2M4_AUGER)+
		AugerRate(Z,L2_M3M4_AUGER)+
		2.0*AugerRate(Z,L2_M4M4_AUGER)+
		AugerRate(Z,L2_M5M4_AUGER)+
		AugerRate(Z,L2_M4L3_AUGER)+
		AugerRate(Z,L2_M4M1_AUGER)+
		AugerRate(Z,L2_M4M2_AUGER)+
		AugerRate(Z,L2_M4M3_AUGER)+
		AugerRate(Z,L2_M4M5_AUGER)
		);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M4_LINE)+
		(1.0-FluorYield(Z,L3_SHELL))*PL3*(
		AugerRate(Z,L3_M1M4_AUGER)+
		AugerRate(Z,L3_M2M4_AUGER)+
		AugerRate(Z,L3_M3M4_AUGER)+
		2.0*AugerRate(Z,L3_M4M4_AUGER)+
		AugerRate(Z,L3_M5M4_AUGER)+
		AugerRate(Z,L3_M4M1_AUGER)+
		AugerRate(Z,L3_M4M2_AUGER)+
		AugerRate(Z,L3_M4M3_AUGER)+
		AugerRate(Z,L3_M4M5_AUGER)
		);

	if (PM1 > 0.0)
		rv += (1.0-FluorYield(Z,M1_SHELL)-CosKronTransProb(Z,FM12_TRANS)-CosKronTransProb(Z,FM13_TRANS)-CosKronTransProb(Z,FM14_TRANS)-CosKronTransProb(Z,FM15_TRANS))*PM1*(
		AugerRate(Z,M1_M2M4_AUGER)+
		AugerRate(Z,M1_M3M4_AUGER)+
		AugerRate(Z,M1_M4M2_AUGER)+
		AugerRate(Z,M1_M4M3_AUGER)+
		2.0*AugerRate(Z,M1_M4M4_AUGER)+
		AugerRate(Z,M1_M4M5_AUGER)+
		AugerRate(Z,M1_M5M4_AUGER)
		)
		+CosKronTransProb(Z,FM14_TRANS)*PM1;
	
	if (PM2 > 0.0)
		rv += (1.0-FluorYield(Z,M2_SHELL)-CosKronTransProb(Z,FM23_TRANS)-CosKronTransProb(Z,FM24_TRANS)-CosKronTransProb(Z,FM25_TRANS))*PM2*(
		AugerRate(Z,M2_M3M4_AUGER)+
		AugerRate(Z,M2_M3M5_AUGER)+
		AugerRate(Z,M2_M4M3_AUGER)+
		2.0*AugerRate(Z,M2_M4M4_AUGER)+
		AugerRate(Z,M2_M4M5_AUGER)+
		AugerRate(Z,M2_M5M3_AUGER)+
		AugerRate(Z,M2_M5M4_AUGER)
		)
		+CosKronTransProb(Z,FM24_TRANS)*PM2;

	if (PM3 > 0.0)
		rv += (1.0-FluorYield(Z,M3_SHELL)-CosKronTransProb(Z,FM34_TRANS)-CosKronTransProb(Z,FM35_TRANS))*PM3*(
		2.0*AugerRate(Z,M3_M4M4_AUGER)+
		AugerRate(Z,M3_M4M5_AUGER)+
		AugerRate(Z,M3_M5M4_AUGER)
		)
		+CosKronTransProb(Z,FM34_TRANS)*PM3;

	return rv;
}

float PM5_pure_kissel(int Z, float E, float PM1, float PM2, float PM3, float PM4) {
	float rv;

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

float PM5_rad_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2, float PM3, float PM4) {
	float rv;

	rv = CS_Photo_Partial(Z, M5_SHELL, E);

	//yes I know that KM5 lines are forbidden...
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

float PM5_auger_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2, float PM3, float PM4) {
	float rv;

	rv = CS_Photo_Partial(Z, M5_SHELL, E);

	if (PK > 0.0) 
		rv += (1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1M5_AUGER)+
		AugerRate(Z,K_L2M5_AUGER)+
		AugerRate(Z,K_L3M5_AUGER)+
		AugerRate(Z,K_M1M5_AUGER)+
		AugerRate(Z,K_M2M5_AUGER)+
		AugerRate(Z,K_M3M5_AUGER)+
		AugerRate(Z,K_M4M5_AUGER)+
		2.0*AugerRate(Z,K_M5M5_AUGER)+
		AugerRate(Z,K_M5L1_AUGER)+
		AugerRate(Z,K_M5L2_AUGER)+
		AugerRate(Z,K_M5L3_AUGER)+
		AugerRate(Z,K_M5M1_AUGER)+
		AugerRate(Z,K_M5M2_AUGER)+
		AugerRate(Z,K_M5M3_AUGER)+
		AugerRate(Z,K_M5M4_AUGER)
		);
	if (PL1 > 0.0)
		rv += (1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		AugerRate(Z,L1_L2M5_AUGER)+
		AugerRate(Z,L1_L3M5_AUGER)+
		AugerRate(Z,L1_M1M5_AUGER)+
		AugerRate(Z,L1_M2M5_AUGER)+
		AugerRate(Z,L1_M3M5_AUGER)+
		AugerRate(Z,L1_M4M5_AUGER)+
		2.0*AugerRate(Z,L1_M5M5_AUGER)+
		AugerRate(Z,L1_M5L2_AUGER)+
		AugerRate(Z,L1_M5L3_AUGER)+
		AugerRate(Z,L1_M5M1_AUGER)+
		AugerRate(Z,L1_M5M2_AUGER)+
		AugerRate(Z,L1_M5M3_AUGER)+
		AugerRate(Z,L1_M5M4_AUGER)
		);
	if (PL2 > 0.0)
		rv += (1.0-FluorYield(Z,L2_SHELL)-CosKronTransProb(Z,FL23_TRANS))*PL2*(
		AugerRate(Z,L2_L3M5_AUGER)+
		AugerRate(Z,L2_M1M5_AUGER)+
		AugerRate(Z,L2_M2M5_AUGER)+
		AugerRate(Z,L2_M3M5_AUGER)+
		AugerRate(Z,L2_M4M5_AUGER)+
		2.0*AugerRate(Z,L2_M5M5_AUGER)+
		AugerRate(Z,L2_M5L3_AUGER)+
		AugerRate(Z,L2_M5M1_AUGER)+
		AugerRate(Z,L2_M5M2_AUGER)+
		AugerRate(Z,L2_M5M3_AUGER)+
		AugerRate(Z,L2_M5M4_AUGER)
		);
	if (PL3 > 0.0)
		rv += (1.0-FluorYield(Z,L3_SHELL))*PL3*(
		AugerRate(Z,L3_M1M5_AUGER)+
		AugerRate(Z,L3_M2M5_AUGER)+
		AugerRate(Z,L3_M3M5_AUGER)+
		AugerRate(Z,L3_M4M5_AUGER)+
		2.0*AugerRate(Z,L3_M5M5_AUGER)+
		AugerRate(Z,L3_M5M1_AUGER)+
		AugerRate(Z,L3_M5M2_AUGER)+
		AugerRate(Z,L3_M5M3_AUGER)+
		AugerRate(Z,L3_M5M4_AUGER)
		);
	if (PM1 > 0.0)
		rv += (1.0-FluorYield(Z,M1_SHELL)-CosKronTransProb(Z,FM12_TRANS)-CosKronTransProb(Z,FM13_TRANS)-CosKronTransProb(Z,FM14_TRANS)-CosKronTransProb(Z,FM15_TRANS))*PM1*(
		AugerRate(Z,M1_M2M5_AUGER)+
		AugerRate(Z,M1_M3M5_AUGER)+
		AugerRate(Z,M1_M4M5_AUGER)+
		2.0*AugerRate(Z,M1_M5M5_AUGER)+
		AugerRate(Z,M1_M5M2_AUGER)+
		AugerRate(Z,M1_M5M3_AUGER)+
		AugerRate(Z,M1_M5M4_AUGER)
		)
		+CosKronTransProb(Z,FM15_TRANS)*PM1;
	if (PM2 > 0.0)
		rv += (1.0-FluorYield(Z,M2_SHELL)-CosKronTransProb(Z,FM23_TRANS)-CosKronTransProb(Z,FM24_TRANS)-CosKronTransProb(Z,FM25_TRANS))*PM2*(
		AugerRate(Z,M2_M3M5_AUGER)+
		AugerRate(Z,M2_M4M5_AUGER)+
		2.0*AugerRate(Z,M2_M5M5_AUGER)+
		AugerRate(Z,M2_M5M3_AUGER)+
		AugerRate(Z,M2_M5M4_AUGER)
		)
		+CosKronTransProb(Z,FM25_TRANS)*PM2;
	if (PM3 > 0.0)	
		rv += (1.0-FluorYield(Z,M3_SHELL)-CosKronTransProb(Z,FM34_TRANS)-CosKronTransProb(Z,FM35_TRANS))*PM3*(
		AugerRate(Z,M3_M4M5_AUGER)+
		AugerRate(Z,M3_M5M4_AUGER)+
		2.0*AugerRate(Z,M3_M5M5_AUGER)
		)
		+CosKronTransProb(Z,FM35_TRANS)*PM3;
	if (PM4 > 0.0)	
		rv += (1.0-FluorYield(Z,M4_SHELL)-CosKronTransProb(Z,FM45_TRANS))*PM4*(
		2.0*AugerRate(Z,M4_M5M5_AUGER)
		)
		+CosKronTransProb(Z,FM45_TRANS)*PM4;

	return rv;
}

float PM5_full_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2, float PM3, float PM4) {
	float rv;

	rv = CS_Photo_Partial(Z, M5_SHELL, E);

	if (PK > 0.0) 
		rv += FluorYield(Z,K_SHELL)*PK*RadRate(Z,KM5_LINE)+
		(1.0-FluorYield(Z,K_SHELL))*PK*(
		AugerRate(Z,K_L1M5_AUGER)+
		AugerRate(Z,K_L2M5_AUGER)+
		AugerRate(Z,K_L3M5_AUGER)+
		AugerRate(Z,K_M1M5_AUGER)+
		AugerRate(Z,K_M2M5_AUGER)+
		AugerRate(Z,K_M3M5_AUGER)+
		AugerRate(Z,K_M4M5_AUGER)+
		2.0*AugerRate(Z,K_M5M5_AUGER)+
		AugerRate(Z,K_M5L1_AUGER)+
		AugerRate(Z,K_M5L2_AUGER)+
		AugerRate(Z,K_M5L3_AUGER)+
		AugerRate(Z,K_M5M1_AUGER)+
		AugerRate(Z,K_M5M2_AUGER)+
		AugerRate(Z,K_M5M3_AUGER)+
		AugerRate(Z,K_M5M4_AUGER)
		);

	if (PL1 > 0.0)
		rv += FluorYield(Z,L1_SHELL)*PL1*RadRate(Z,L1M5_LINE)+
		(1.0-FluorYield(Z,L1_SHELL)-CosKronTransProb(Z,FL12_TRANS)-CosKronTransProb(Z,FL13_TRANS))*PL1*(
		AugerRate(Z,L1_L2M5_AUGER)+
		AugerRate(Z,L1_L3M5_AUGER)+
		AugerRate(Z,L1_M1M5_AUGER)+
		AugerRate(Z,L1_M2M5_AUGER)+
		AugerRate(Z,L1_M3M5_AUGER)+
		AugerRate(Z,L1_M4M5_AUGER)+
		2.0*AugerRate(Z,L1_M5M5_AUGER)+
		AugerRate(Z,L1_M5L2_AUGER)+
		AugerRate(Z,L1_M5L3_AUGER)+
		AugerRate(Z,L1_M5M1_AUGER)+
		AugerRate(Z,L1_M5M2_AUGER)+
		AugerRate(Z,L1_M5M3_AUGER)+
		AugerRate(Z,L1_M5M4_AUGER)
		);

	if (PL2 > 0.0)
		rv += FluorYield(Z,L2_SHELL)*PL2*RadRate(Z,L2M5_LINE)+
		(1.0-FluorYield(Z,L2_SHELL)-CosKronTransProb(Z,FL23_TRANS))*PL2*(
		AugerRate(Z,L2_L3M5_AUGER)+
		AugerRate(Z,L2_M1M5_AUGER)+
		AugerRate(Z,L2_M2M5_AUGER)+
		AugerRate(Z,L2_M3M5_AUGER)+
		AugerRate(Z,L2_M4M5_AUGER)+
		2.0*AugerRate(Z,L2_M5M5_AUGER)+
		AugerRate(Z,L2_M5L3_AUGER)+
		AugerRate(Z,L2_M5M1_AUGER)+
		AugerRate(Z,L2_M5M2_AUGER)+
		AugerRate(Z,L2_M5M3_AUGER)+
		AugerRate(Z,L2_M5M4_AUGER)
		);

	if (PL3 > 0.0)
		rv += FluorYield(Z,L3_SHELL)*PL3*RadRate(Z,L3M5_LINE)+
		(1.0-FluorYield(Z,L3_SHELL))*PL3*(
		AugerRate(Z,L3_M1M5_AUGER)+
		AugerRate(Z,L3_M2M5_AUGER)+
		AugerRate(Z,L3_M3M5_AUGER)+
		AugerRate(Z,L3_M4M5_AUGER)+
		2.0*AugerRate(Z,L3_M5M5_AUGER)+
		AugerRate(Z,L3_M5M1_AUGER)+
		AugerRate(Z,L3_M5M2_AUGER)+
		AugerRate(Z,L3_M5M3_AUGER)+
		AugerRate(Z,L3_M5M4_AUGER)
		);
	if (PM1 > 0.0)
		rv += (1.0-FluorYield(Z,M1_SHELL)-CosKronTransProb(Z,FM12_TRANS)-CosKronTransProb(Z,FM13_TRANS)-CosKronTransProb(Z,FM14_TRANS)-CosKronTransProb(Z,FM15_TRANS))*PM1*(
		AugerRate(Z,M1_M2M5_AUGER)+
		AugerRate(Z,M1_M3M5_AUGER)+
		AugerRate(Z,M1_M4M5_AUGER)+
		2.0*AugerRate(Z,M1_M5M5_AUGER)+
		AugerRate(Z,M1_M5M2_AUGER)+
		AugerRate(Z,M1_M5M3_AUGER)+
		AugerRate(Z,M1_M5M4_AUGER)
		)
		+CosKronTransProb(Z,FM15_TRANS)*PM1;
	
	if (PM2 > 0.0)
		rv += (1.0-FluorYield(Z,M2_SHELL)-CosKronTransProb(Z,FM23_TRANS)-CosKronTransProb(Z,FM24_TRANS)-CosKronTransProb(Z,FM25_TRANS))*PM2*(
		AugerRate(Z,M2_M3M5_AUGER)+
		AugerRate(Z,M2_M4M5_AUGER)+
		2.0*AugerRate(Z,M2_M5M5_AUGER)+
		AugerRate(Z,M2_M5M3_AUGER)+
		AugerRate(Z,M2_M5M4_AUGER)
		)
		+CosKronTransProb(Z,FM25_TRANS)*PM2;

	if (PM3 > 0.0)
		rv += (1.0-FluorYield(Z,M3_SHELL)-CosKronTransProb(Z,FM34_TRANS)-CosKronTransProb(Z,FM35_TRANS))*PM3*(
		AugerRate(Z,M3_M4M5_AUGER)+
		AugerRate(Z,M3_M5M4_AUGER)+
		2.0*AugerRate(Z,M3_M5M5_AUGER)
		)
		+CosKronTransProb(Z,FM35_TRANS)*PM3;

	if (PM4 > 0.0)
		rv += (1.0-FluorYield(Z,M4_SHELL)-CosKronTransProb(Z,FM45_TRANS))*PM4*(
		2.0*AugerRate(Z,M4_M5M5_AUGER)
		)
		+CosKronTransProb(Z,FM45_TRANS)*PM4;

	return rv;
}


