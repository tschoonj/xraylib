#include "xraylib.h"
#include "xrf_cross_sections_aux-private.h"
#include <stddef.h>

double PL1_get_cross_sections_constant_full(int Z, int shell) {
	if (shell == K_SHELL) {
		return (FluorYield(Z, K_SHELL, NULL) * RadRate(Z, KL1_LINE, NULL) +
		AugerYield(Z, K_SHELL, NULL) * (
		2 * AugerRate(Z, K_L1L1_AUGER, NULL)+
		AugerRate(Z, K_L1L2_AUGER, NULL)+
		AugerRate(Z, K_L1L3_AUGER, NULL)+
		AugerRate(Z, K_L1M1_AUGER, NULL)+
		AugerRate(Z, K_L1M2_AUGER, NULL)+
		AugerRate(Z, K_L1M3_AUGER, NULL)+
		AugerRate(Z, K_L1M4_AUGER, NULL)+
		AugerRate(Z, K_L1M5_AUGER, NULL)+
		AugerRate(Z, K_L1N1_AUGER, NULL)+
		AugerRate(Z, K_L1N2_AUGER, NULL)+
		AugerRate(Z, K_L1N3_AUGER, NULL)+
		AugerRate(Z, K_L1N4_AUGER, NULL)+
		AugerRate(Z, K_L1N5_AUGER, NULL)+
		AugerRate(Z, K_L1N6_AUGER, NULL)+
		AugerRate(Z, K_L1N7_AUGER, NULL)+
		AugerRate(Z, K_L1O1_AUGER, NULL)+
		AugerRate(Z, K_L1O2_AUGER, NULL)+
		AugerRate(Z, K_L1O3_AUGER, NULL)+
		AugerRate(Z, K_L1O4_AUGER, NULL)+
		AugerRate(Z, K_L1O5_AUGER, NULL)+
		AugerRate(Z, K_L1O6_AUGER, NULL)+
		AugerRate(Z, K_L1O7_AUGER, NULL)+
		AugerRate(Z, K_L1P1_AUGER, NULL)+
		AugerRate(Z, K_L1P2_AUGER, NULL)+
		AugerRate(Z, K_L1P3_AUGER, NULL)+
		AugerRate(Z, K_L1P4_AUGER, NULL)+
		AugerRate(Z, K_L1P5_AUGER, NULL)+
		AugerRate(Z, K_L1Q1_AUGER, NULL)+
		AugerRate(Z, K_L1Q2_AUGER, NULL)+
		AugerRate(Z, K_L1Q3_AUGER, NULL)+
		AugerRate(Z, K_L2L1_AUGER, NULL)+
		AugerRate(Z, K_L3L1_AUGER, NULL)+
		AugerRate(Z, K_M1L1_AUGER, NULL)+
		AugerRate(Z, K_M2L1_AUGER, NULL)+
		AugerRate(Z, K_M3L1_AUGER, NULL)+
		AugerRate(Z, K_M4L1_AUGER, NULL)+
		AugerRate(Z, K_M5L1_AUGER, NULL)
		));
	}
	return 0.0;
}

double PL1_get_cross_sections_constant_auger_only(int Z, int shell) {
	if (shell == K_SHELL) {
		return AugerYield(Z, K_SHELL, NULL) * (
		2 * AugerRate(Z, K_L1L1_AUGER, NULL)+
		AugerRate(Z, K_L1L2_AUGER, NULL)+
		AugerRate(Z, K_L1L3_AUGER, NULL)+
		AugerRate(Z, K_L1M1_AUGER, NULL)+
		AugerRate(Z, K_L1M2_AUGER, NULL)+
		AugerRate(Z, K_L1M3_AUGER, NULL)+
		AugerRate(Z, K_L1M4_AUGER, NULL)+
		AugerRate(Z, K_L1M5_AUGER, NULL)+
		AugerRate(Z, K_L1N1_AUGER, NULL)+
		AugerRate(Z, K_L1N2_AUGER, NULL)+
		AugerRate(Z, K_L1N3_AUGER, NULL)+
		AugerRate(Z, K_L1N4_AUGER, NULL)+
		AugerRate(Z, K_L1N5_AUGER, NULL)+
		AugerRate(Z, K_L1N6_AUGER, NULL)+
		AugerRate(Z, K_L1N7_AUGER, NULL)+
		AugerRate(Z, K_L1O1_AUGER, NULL)+
		AugerRate(Z, K_L1O2_AUGER, NULL)+
		AugerRate(Z, K_L1O3_AUGER, NULL)+
		AugerRate(Z, K_L1O4_AUGER, NULL)+
		AugerRate(Z, K_L1O5_AUGER, NULL)+
		AugerRate(Z, K_L1O6_AUGER, NULL)+
		AugerRate(Z, K_L1O7_AUGER, NULL)+
		AugerRate(Z, K_L1P1_AUGER, NULL)+
		AugerRate(Z, K_L1P2_AUGER, NULL)+
		AugerRate(Z, K_L1P3_AUGER, NULL)+
		AugerRate(Z, K_L1P4_AUGER, NULL)+
		AugerRate(Z, K_L1P5_AUGER, NULL)+
		AugerRate(Z, K_L1Q1_AUGER, NULL)+
		AugerRate(Z, K_L1Q2_AUGER, NULL)+
		AugerRate(Z, K_L1Q3_AUGER, NULL)+
		AugerRate(Z, K_L2L1_AUGER, NULL)+
		AugerRate(Z, K_L3L1_AUGER, NULL)+
		AugerRate(Z, K_M1L1_AUGER, NULL)+
		AugerRate(Z, K_M2L1_AUGER, NULL)+
		AugerRate(Z, K_M3L1_AUGER, NULL)+
		AugerRate(Z, K_M4L1_AUGER, NULL)+
		AugerRate(Z, K_M5L1_AUGER, NULL)
		);
	}
	return 0.0;
}

double PL2_get_cross_sections_constant_auger_only(int Z, int shell) {
	if (shell == K_SHELL) {
		return AugerYield(Z, K_SHELL, NULL) * (
		AugerRate(Z, K_L1L2_AUGER, NULL)+
		AugerRate(Z, K_L2L1_AUGER, NULL)+
		2 * AugerRate(Z, K_L2L2_AUGER, NULL)+
		AugerRate(Z, K_L2L3_AUGER, NULL)+
		AugerRate(Z, K_L2M1_AUGER, NULL)+
		AugerRate(Z, K_L2M2_AUGER, NULL)+
		AugerRate(Z, K_L2M3_AUGER, NULL)+
		AugerRate(Z, K_L2M4_AUGER, NULL)+
		AugerRate(Z, K_L2M5_AUGER, NULL)+
		AugerRate(Z, K_L2N1_AUGER, NULL)+
		AugerRate(Z, K_L2N2_AUGER, NULL)+
		AugerRate(Z, K_L2N3_AUGER, NULL)+
		AugerRate(Z, K_L2N4_AUGER, NULL)+
		AugerRate(Z, K_L2N5_AUGER, NULL)+
		AugerRate(Z, K_L2N6_AUGER, NULL)+
		AugerRate(Z, K_L2N7_AUGER, NULL)+
		AugerRate(Z, K_L2O1_AUGER, NULL)+
		AugerRate(Z, K_L2O2_AUGER, NULL)+
		AugerRate(Z, K_L2O3_AUGER, NULL)+
		AugerRate(Z, K_L2O4_AUGER, NULL)+
		AugerRate(Z, K_L2O5_AUGER, NULL)+
		AugerRate(Z, K_L2O6_AUGER, NULL)+
		AugerRate(Z, K_L2O7_AUGER, NULL)+
		AugerRate(Z, K_L2P1_AUGER, NULL)+
		AugerRate(Z, K_L2P2_AUGER, NULL)+
		AugerRate(Z, K_L2P3_AUGER, NULL)+
		AugerRate(Z, K_L2P4_AUGER, NULL)+
		AugerRate(Z, K_L2P5_AUGER, NULL)+
		AugerRate(Z, K_L2Q1_AUGER, NULL)+
		AugerRate(Z, K_L2Q2_AUGER, NULL)+
		AugerRate(Z, K_L2Q3_AUGER, NULL)+
		AugerRate(Z, K_L3L2_AUGER, NULL)+
		AugerRate(Z, K_M1L2_AUGER, NULL)+
		AugerRate(Z, K_M2L2_AUGER, NULL)+
		AugerRate(Z, K_M3L2_AUGER, NULL)+
		AugerRate(Z, K_M4L2_AUGER, NULL)+
		AugerRate(Z, K_M5L2_AUGER, NULL)
		);
	}
	return 0.0;
}

double PL2_get_cross_sections_constant_full(int Z, int shell) {
	if (shell == K_SHELL) {
		return (FluorYield(Z, K_SHELL, NULL) * RadRate(Z, KL2_LINE, NULL) +
		AugerYield(Z, K_SHELL, NULL) * (
		AugerRate(Z, K_L1L2_AUGER, NULL)+
		AugerRate(Z, K_L2L1_AUGER, NULL)+
		2 * AugerRate(Z, K_L2L2_AUGER, NULL)+
		AugerRate(Z, K_L2L3_AUGER, NULL)+
		AugerRate(Z, K_L2M1_AUGER, NULL)+
		AugerRate(Z, K_L2M2_AUGER, NULL)+
		AugerRate(Z, K_L2M3_AUGER, NULL)+
		AugerRate(Z, K_L2M4_AUGER, NULL)+
		AugerRate(Z, K_L2M5_AUGER, NULL)+
		AugerRate(Z, K_L2N1_AUGER, NULL)+
		AugerRate(Z, K_L2N2_AUGER, NULL)+
		AugerRate(Z, K_L2N3_AUGER, NULL)+
		AugerRate(Z, K_L2N4_AUGER, NULL)+
		AugerRate(Z, K_L2N5_AUGER, NULL)+
		AugerRate(Z, K_L2N6_AUGER, NULL)+
		AugerRate(Z, K_L2N7_AUGER, NULL)+
		AugerRate(Z, K_L2O1_AUGER, NULL)+
		AugerRate(Z, K_L2O2_AUGER, NULL)+
		AugerRate(Z, K_L2O3_AUGER, NULL)+
		AugerRate(Z, K_L2O4_AUGER, NULL)+
		AugerRate(Z, K_L2O5_AUGER, NULL)+
		AugerRate(Z, K_L2O6_AUGER, NULL)+
		AugerRate(Z, K_L2O7_AUGER, NULL)+
		AugerRate(Z, K_L2P1_AUGER, NULL)+
		AugerRate(Z, K_L2P2_AUGER, NULL)+
		AugerRate(Z, K_L2P3_AUGER, NULL)+
		AugerRate(Z, K_L2P4_AUGER, NULL)+
		AugerRate(Z, K_L2P5_AUGER, NULL)+
		AugerRate(Z, K_L2Q1_AUGER, NULL)+
		AugerRate(Z, K_L2Q2_AUGER, NULL)+
		AugerRate(Z, K_L2Q3_AUGER, NULL)+
		AugerRate(Z, K_L3L2_AUGER, NULL)+
		AugerRate(Z, K_M1L2_AUGER, NULL)+
		AugerRate(Z, K_M2L2_AUGER, NULL)+
		AugerRate(Z, K_M3L2_AUGER, NULL)+
		AugerRate(Z, K_M4L2_AUGER, NULL)+
		AugerRate(Z, K_M5L2_AUGER, NULL)
		));
	}
	return 0.0;
}

double PL3_get_cross_sections_constant_auger_only(int Z, int shell) {
	if (shell == K_SHELL) {
		return AugerYield(Z, K_SHELL, NULL) * (
		AugerRate(Z, K_L1L3_AUGER, NULL)+
		AugerRate(Z, K_L2L3_AUGER, NULL)+
		AugerRate(Z, K_L3L1_AUGER, NULL)+
		AugerRate(Z, K_L3L2_AUGER, NULL)+
		2 * AugerRate(Z, K_L3L3_AUGER, NULL)+
		AugerRate(Z, K_L3M1_AUGER, NULL)+
		AugerRate(Z, K_L3M2_AUGER, NULL)+
		AugerRate(Z, K_L3M3_AUGER, NULL)+
		AugerRate(Z, K_L3M4_AUGER, NULL)+
		AugerRate(Z, K_L3M5_AUGER, NULL)+
		AugerRate(Z, K_L3N1_AUGER, NULL)+
		AugerRate(Z, K_L3N2_AUGER, NULL)+
		AugerRate(Z, K_L3N3_AUGER, NULL)+
		AugerRate(Z, K_L3N4_AUGER, NULL)+
		AugerRate(Z, K_L3N5_AUGER, NULL)+
		AugerRate(Z, K_L3N6_AUGER, NULL)+
		AugerRate(Z, K_L3N7_AUGER, NULL)+
		AugerRate(Z, K_L3O1_AUGER, NULL)+
		AugerRate(Z, K_L3O2_AUGER, NULL)+
		AugerRate(Z, K_L3O3_AUGER, NULL)+
		AugerRate(Z, K_L3O4_AUGER, NULL)+
		AugerRate(Z, K_L3O5_AUGER, NULL)+
		AugerRate(Z, K_L3O6_AUGER, NULL)+
		AugerRate(Z, K_L3O7_AUGER, NULL)+
		AugerRate(Z, K_L3P1_AUGER, NULL)+
		AugerRate(Z, K_L3P2_AUGER, NULL)+
		AugerRate(Z, K_L3P3_AUGER, NULL)+
		AugerRate(Z, K_L3P4_AUGER, NULL)+
		AugerRate(Z, K_L3P5_AUGER, NULL)+
		AugerRate(Z, K_L3Q1_AUGER, NULL)+
		AugerRate(Z, K_L3Q2_AUGER, NULL)+
		AugerRate(Z, K_L3Q3_AUGER, NULL)+
		AugerRate(Z, K_M1L3_AUGER, NULL)+
		AugerRate(Z, K_M2L3_AUGER, NULL)+
		AugerRate(Z, K_M3L3_AUGER, NULL)+
		AugerRate(Z, K_M4L3_AUGER, NULL)+
		AugerRate(Z, K_M5L3_AUGER, NULL)
		);
	}
	return 0.0;
}

double PL3_get_cross_sections_constant_full(int Z, int shell) {
	if (shell == K_SHELL) {
		return (FluorYield(Z, K_SHELL, NULL) * RadRate(Z, KL3_LINE, NULL) +
		AugerYield(Z, K_SHELL, NULL) * (
		AugerRate(Z, K_L1L3_AUGER, NULL)+
		AugerRate(Z, K_L2L3_AUGER, NULL)+
		AugerRate(Z, K_L3L1_AUGER, NULL)+
		AugerRate(Z, K_L3L2_AUGER, NULL)+
		2 * AugerRate(Z, K_L3L3_AUGER, NULL)+
		AugerRate(Z, K_L3M1_AUGER, NULL)+
		AugerRate(Z, K_L3M2_AUGER, NULL)+
		AugerRate(Z, K_L3M3_AUGER, NULL)+
		AugerRate(Z, K_L3M4_AUGER, NULL)+
		AugerRate(Z, K_L3M5_AUGER, NULL)+
		AugerRate(Z, K_L3N1_AUGER, NULL)+
		AugerRate(Z, K_L3N2_AUGER, NULL)+
		AugerRate(Z, K_L3N3_AUGER, NULL)+
		AugerRate(Z, K_L3N4_AUGER, NULL)+
		AugerRate(Z, K_L3N5_AUGER, NULL)+
		AugerRate(Z, K_L3N6_AUGER, NULL)+
		AugerRate(Z, K_L3N7_AUGER, NULL)+
		AugerRate(Z, K_L3O1_AUGER, NULL)+
		AugerRate(Z, K_L3O2_AUGER, NULL)+
		AugerRate(Z, K_L3O3_AUGER, NULL)+
		AugerRate(Z, K_L3O4_AUGER, NULL)+
		AugerRate(Z, K_L3O5_AUGER, NULL)+
		AugerRate(Z, K_L3O6_AUGER, NULL)+
		AugerRate(Z, K_L3O7_AUGER, NULL)+
		AugerRate(Z, K_L3P1_AUGER, NULL)+
		AugerRate(Z, K_L3P2_AUGER, NULL)+
		AugerRate(Z, K_L3P3_AUGER, NULL)+
		AugerRate(Z, K_L3P4_AUGER, NULL)+
		AugerRate(Z, K_L3P5_AUGER, NULL)+
		AugerRate(Z, K_L3Q1_AUGER, NULL)+
		AugerRate(Z, K_L3Q2_AUGER, NULL)+
		AugerRate(Z, K_L3Q3_AUGER, NULL)+
		AugerRate(Z, K_M1L3_AUGER, NULL)+
		AugerRate(Z, K_M2L3_AUGER, NULL)+
		AugerRate(Z, K_M3L3_AUGER, NULL)+
		AugerRate(Z, K_M4L3_AUGER, NULL)+
		AugerRate(Z, K_M5L3_AUGER, NULL)
		));
	}
	return 0.0;
}

double PM1_get_cross_sections_constant_auger_only(int Z, int shell) {
	if (shell == K_SHELL) {
		return AugerYield(Z, K_SHELL, NULL) * (
		AugerRate(Z, K_L1M1_AUGER, NULL)+
		AugerRate(Z, K_L2M1_AUGER, NULL)+
		AugerRate(Z, K_L3M1_AUGER, NULL)+
		AugerRate(Z, K_M1L1_AUGER, NULL)+
		AugerRate(Z, K_M1L2_AUGER, NULL)+
		AugerRate(Z, K_M1L3_AUGER, NULL)+
		2 * AugerRate(Z, K_M1M1_AUGER, NULL)+
		AugerRate(Z, K_M1M2_AUGER, NULL)+
		AugerRate(Z, K_M1M3_AUGER, NULL)+
		AugerRate(Z, K_M1M4_AUGER, NULL)+
		AugerRate(Z, K_M1M5_AUGER, NULL)+
		AugerRate(Z, K_M2M1_AUGER, NULL)+
		AugerRate(Z, K_M3M1_AUGER, NULL)+
		AugerRate(Z, K_M4M1_AUGER, NULL)+
		AugerRate(Z, K_M5M1_AUGER, NULL)
		);
	}
	else if (shell == L1_SHELL) {
		return AugerYield(Z, L1_SHELL, NULL) * (
		2 * AugerRate(Z, L1_M1M1_AUGER, NULL)+
		AugerRate(Z, L1_M1M2_AUGER, NULL)+
		AugerRate(Z, L1_M1M3_AUGER, NULL)+
		AugerRate(Z, L1_M1M4_AUGER, NULL)+
		AugerRate(Z, L1_M1M5_AUGER, NULL)+
		AugerRate(Z, L1_M1N1_AUGER, NULL)+
		AugerRate(Z, L1_M1N2_AUGER, NULL)+
		AugerRate(Z, L1_M1N3_AUGER, NULL)+
		AugerRate(Z, L1_M1N4_AUGER, NULL)+
		AugerRate(Z, L1_M1N5_AUGER, NULL)+
		AugerRate(Z, L1_M1N6_AUGER, NULL)+
		AugerRate(Z, L1_M1N7_AUGER, NULL)+
		AugerRate(Z, L1_M1O1_AUGER, NULL)+
		AugerRate(Z, L1_M1O2_AUGER, NULL)+
		AugerRate(Z, L1_M1O3_AUGER, NULL)+
		AugerRate(Z, L1_M1O4_AUGER, NULL)+
		AugerRate(Z, L1_M1O5_AUGER, NULL)+
		AugerRate(Z, L1_M1O6_AUGER, NULL)+
		AugerRate(Z, L1_M1O7_AUGER, NULL)+
		AugerRate(Z, L1_M1P1_AUGER, NULL)+
		AugerRate(Z, L1_M1P2_AUGER, NULL)+
		AugerRate(Z, L1_M1P3_AUGER, NULL)+
		AugerRate(Z, L1_M1P4_AUGER, NULL)+
		AugerRate(Z, L1_M1P5_AUGER, NULL)+
		AugerRate(Z, L1_M1Q1_AUGER, NULL)+
		AugerRate(Z, L1_M1Q2_AUGER, NULL)+
		AugerRate(Z, L1_M1Q3_AUGER, NULL)+
		AugerRate(Z, L1_M2M1_AUGER, NULL)+
		AugerRate(Z, L1_M3M1_AUGER, NULL)+
		AugerRate(Z, L1_M4M1_AUGER, NULL)+
		AugerRate(Z, L1_M5M1_AUGER, NULL)
		);
	}
	else if (shell == L2_SHELL) {
		return AugerYield(Z, L2_SHELL, NULL) * (
		2 *AugerRate(Z, L2_M1M1_AUGER, NULL)+
		AugerRate(Z, L2_M1M2_AUGER, NULL)+
		AugerRate(Z, L2_M1M3_AUGER, NULL)+
		AugerRate(Z, L2_M1M4_AUGER, NULL)+
		AugerRate(Z, L2_M1M5_AUGER, NULL)+
		AugerRate(Z, L2_M1N1_AUGER, NULL)+
		AugerRate(Z, L2_M1N2_AUGER, NULL)+
		AugerRate(Z, L2_M1N3_AUGER, NULL)+
		AugerRate(Z, L2_M1N4_AUGER, NULL)+
		AugerRate(Z, L2_M1N5_AUGER, NULL)+
		AugerRate(Z, L2_M1N6_AUGER, NULL)+
		AugerRate(Z, L2_M1N7_AUGER, NULL)+
		AugerRate(Z, L2_M1O1_AUGER, NULL)+
		AugerRate(Z, L2_M1O2_AUGER, NULL)+
		AugerRate(Z, L2_M1O3_AUGER, NULL)+
		AugerRate(Z, L2_M1O4_AUGER, NULL)+
		AugerRate(Z, L2_M1O5_AUGER, NULL)+
		AugerRate(Z, L2_M1O6_AUGER, NULL)+
		AugerRate(Z, L2_M1O7_AUGER, NULL)+
		AugerRate(Z, L2_M1P1_AUGER, NULL)+
		AugerRate(Z, L2_M1P2_AUGER, NULL)+
		AugerRate(Z, L2_M1P3_AUGER, NULL)+
		AugerRate(Z, L2_M1P4_AUGER, NULL)+
		AugerRate(Z, L2_M1P5_AUGER, NULL)+
		AugerRate(Z, L2_M1Q1_AUGER, NULL)+
		AugerRate(Z, L2_M1Q2_AUGER, NULL)+
		AugerRate(Z, L2_M1Q3_AUGER, NULL)+
		AugerRate(Z, L2_M2M1_AUGER, NULL)+
		AugerRate(Z, L2_M3M1_AUGER, NULL)+
		AugerRate(Z, L2_M4M1_AUGER, NULL)+
		AugerRate(Z, L2_M5M1_AUGER, NULL)
		);
	}
	else if (shell == L3_SHELL) {
		return AugerYield(Z, L3_SHELL, NULL) * (
		2 * AugerRate(Z, L3_M1M1_AUGER, NULL)+
		AugerRate(Z, L3_M1M2_AUGER, NULL)+
		AugerRate(Z, L3_M1M3_AUGER, NULL)+
		AugerRate(Z, L3_M1M4_AUGER, NULL)+
		AugerRate(Z, L3_M1M5_AUGER, NULL)+
		AugerRate(Z, L3_M1N1_AUGER, NULL)+
		AugerRate(Z, L3_M1N2_AUGER, NULL)+
		AugerRate(Z, L3_M1N3_AUGER, NULL)+
		AugerRate(Z, L3_M1N4_AUGER, NULL)+
		AugerRate(Z, L3_M1N5_AUGER, NULL)+
		AugerRate(Z, L3_M1N6_AUGER, NULL)+
		AugerRate(Z, L3_M1N7_AUGER, NULL)+
		AugerRate(Z, L3_M1O1_AUGER, NULL)+
		AugerRate(Z, L3_M1O2_AUGER, NULL)+
		AugerRate(Z, L3_M1O3_AUGER, NULL)+
		AugerRate(Z, L3_M1O4_AUGER, NULL)+
		AugerRate(Z, L3_M1O5_AUGER, NULL)+
		AugerRate(Z, L3_M1O6_AUGER, NULL)+
		AugerRate(Z, L3_M1O7_AUGER, NULL)+
		AugerRate(Z, L3_M1P1_AUGER, NULL)+
		AugerRate(Z, L3_M1P2_AUGER, NULL)+
		AugerRate(Z, L3_M1P3_AUGER, NULL)+
		AugerRate(Z, L3_M1P4_AUGER, NULL)+
		AugerRate(Z, L3_M1P5_AUGER, NULL)+
		AugerRate(Z, L3_M1Q1_AUGER, NULL)+
		AugerRate(Z, L3_M1Q2_AUGER, NULL)+
		AugerRate(Z, L3_M1Q3_AUGER, NULL)+
		AugerRate(Z, L3_M2M1_AUGER, NULL)+
		AugerRate(Z, L3_M3M1_AUGER, NULL)+
		AugerRate(Z, L3_M4M1_AUGER, NULL)+
		AugerRate(Z, L3_M5M1_AUGER, NULL)
		);
	}
	return 0.0;
}

double PM1_get_cross_sections_constant_full(int Z, int shell) {
	if (shell == K_SHELL) {
		return (FluorYield(Z, K_SHELL, NULL) * RadRate(Z, KM1_LINE, NULL) +
		AugerYield(Z, K_SHELL, NULL) * (
		AugerRate(Z, K_L1M1_AUGER, NULL)+
		AugerRate(Z, K_L2M1_AUGER, NULL)+
		AugerRate(Z, K_L3M1_AUGER, NULL)+
		AugerRate(Z, K_M1L1_AUGER, NULL)+
		AugerRate(Z, K_M1L2_AUGER, NULL)+
		AugerRate(Z, K_M1L3_AUGER, NULL)+
		2 * AugerRate(Z, K_M1M1_AUGER, NULL)+
		AugerRate(Z, K_M1M2_AUGER, NULL)+
		AugerRate(Z, K_M1M3_AUGER, NULL)+
		AugerRate(Z, K_M1M4_AUGER, NULL)+
		AugerRate(Z, K_M1M5_AUGER, NULL)+
		AugerRate(Z, K_M2M1_AUGER, NULL)+
		AugerRate(Z, K_M3M1_AUGER, NULL)+
		AugerRate(Z, K_M4M1_AUGER, NULL)+
		AugerRate(Z, K_M5M1_AUGER, NULL)
		));
	}
	else if (shell == L1_SHELL) {
		return (FluorYield(Z, L1_SHELL, NULL) * RadRate(Z, L1M1_LINE, NULL) +
		AugerYield(Z,L1_SHELL, NULL) * (
		2 * AugerRate(Z, L1_M1M1_AUGER, NULL)+
		AugerRate(Z, L1_M1M2_AUGER, NULL)+
		AugerRate(Z, L1_M1M3_AUGER, NULL)+
		AugerRate(Z, L1_M1M4_AUGER, NULL)+
		AugerRate(Z, L1_M1M5_AUGER, NULL)+
		AugerRate(Z, L1_M1N1_AUGER, NULL)+
		AugerRate(Z, L1_M1N2_AUGER, NULL)+
		AugerRate(Z, L1_M1N3_AUGER, NULL)+
		AugerRate(Z, L1_M1N4_AUGER, NULL)+
		AugerRate(Z, L1_M1N5_AUGER, NULL)+
		AugerRate(Z, L1_M1N6_AUGER, NULL)+
		AugerRate(Z, L1_M1N7_AUGER, NULL)+
		AugerRate(Z, L1_M1O1_AUGER, NULL)+
		AugerRate(Z, L1_M1O2_AUGER, NULL)+
		AugerRate(Z, L1_M1O3_AUGER, NULL)+
		AugerRate(Z, L1_M1O4_AUGER, NULL)+
		AugerRate(Z, L1_M1O5_AUGER, NULL)+
		AugerRate(Z, L1_M1O6_AUGER, NULL)+
		AugerRate(Z, L1_M1O7_AUGER, NULL)+
		AugerRate(Z, L1_M1P1_AUGER, NULL)+
		AugerRate(Z, L1_M1P2_AUGER, NULL)+
		AugerRate(Z, L1_M1P3_AUGER, NULL)+
		AugerRate(Z, L1_M1P4_AUGER, NULL)+
		AugerRate(Z, L1_M1P5_AUGER, NULL)+
		AugerRate(Z, L1_M1Q1_AUGER, NULL)+
		AugerRate(Z, L1_M1Q2_AUGER, NULL)+
		AugerRate(Z, L1_M1Q3_AUGER, NULL)+
		AugerRate(Z, L1_M2M1_AUGER, NULL)+
		AugerRate(Z, L1_M3M1_AUGER, NULL)+
		AugerRate(Z, L1_M4M1_AUGER, NULL)+
		AugerRate(Z, L1_M5M1_AUGER, NULL)
		));
	}
	else if (shell == L2_SHELL) {
		return (FluorYield(Z, L2_SHELL, NULL) * RadRate(Z, L2M1_LINE, NULL)+
		AugerYield(Z, L2_SHELL, NULL) * (
		2 * AugerRate(Z, L2_M1M1_AUGER, NULL)+
		AugerRate(Z, L2_M1M2_AUGER, NULL)+
		AugerRate(Z, L2_M1M3_AUGER, NULL)+
		AugerRate(Z, L2_M1M4_AUGER, NULL)+
		AugerRate(Z, L2_M1M5_AUGER, NULL)+
		AugerRate(Z, L2_M1N1_AUGER, NULL)+
		AugerRate(Z, L2_M1N2_AUGER, NULL)+
		AugerRate(Z, L2_M1N3_AUGER, NULL)+
		AugerRate(Z, L2_M1N4_AUGER, NULL)+
		AugerRate(Z, L2_M1N5_AUGER, NULL)+
		AugerRate(Z, L2_M1N6_AUGER, NULL)+
		AugerRate(Z, L2_M1N7_AUGER, NULL)+
		AugerRate(Z, L2_M1O1_AUGER, NULL)+
		AugerRate(Z, L2_M1O2_AUGER, NULL)+
		AugerRate(Z, L2_M1O3_AUGER, NULL)+
		AugerRate(Z, L2_M1O4_AUGER, NULL)+
		AugerRate(Z, L2_M1O5_AUGER, NULL)+
		AugerRate(Z, L2_M1O6_AUGER, NULL)+
		AugerRate(Z, L2_M1O7_AUGER, NULL)+
		AugerRate(Z, L2_M1P1_AUGER, NULL)+
		AugerRate(Z, L2_M1P2_AUGER, NULL)+
		AugerRate(Z, L2_M1P3_AUGER, NULL)+
		AugerRate(Z, L2_M1P4_AUGER, NULL)+
		AugerRate(Z, L2_M1P5_AUGER, NULL)+
		AugerRate(Z, L2_M1Q1_AUGER, NULL)+
		AugerRate(Z, L2_M1Q2_AUGER, NULL)+
		AugerRate(Z, L2_M1Q3_AUGER, NULL)+
		AugerRate(Z, L2_M2M1_AUGER, NULL)+
		AugerRate(Z, L2_M3M1_AUGER, NULL)+
		AugerRate(Z, L2_M4M1_AUGER, NULL)+
		AugerRate(Z, L2_M5M1_AUGER, NULL)
		));
	}
	else if (shell == L3_SHELL) {
		return (FluorYield(Z, L3_SHELL, NULL) * RadRate(Z, L3M1_LINE, NULL) +
		AugerYield(Z, L3_SHELL, NULL) * (
		2 * AugerRate(Z, L3_M1M1_AUGER, NULL)+
		AugerRate(Z, L3_M1M2_AUGER, NULL)+
		AugerRate(Z, L3_M1M3_AUGER, NULL)+
		AugerRate(Z, L3_M1M4_AUGER, NULL)+
		AugerRate(Z, L3_M1M5_AUGER, NULL)+
		AugerRate(Z, L3_M1N1_AUGER, NULL)+
		AugerRate(Z, L3_M1N2_AUGER, NULL)+
		AugerRate(Z, L3_M1N3_AUGER, NULL)+
		AugerRate(Z, L3_M1N4_AUGER, NULL)+
		AugerRate(Z, L3_M1N5_AUGER, NULL)+
		AugerRate(Z, L3_M1N6_AUGER, NULL)+
		AugerRate(Z, L3_M1N7_AUGER, NULL)+
		AugerRate(Z, L3_M1O1_AUGER, NULL)+
		AugerRate(Z, L3_M1O2_AUGER, NULL)+
		AugerRate(Z, L3_M1O3_AUGER, NULL)+
		AugerRate(Z, L3_M1O4_AUGER, NULL)+
		AugerRate(Z, L3_M1O5_AUGER, NULL)+
		AugerRate(Z, L3_M1O6_AUGER, NULL)+
		AugerRate(Z, L3_M1O7_AUGER, NULL)+
		AugerRate(Z, L3_M1P1_AUGER, NULL)+
		AugerRate(Z, L3_M1P2_AUGER, NULL)+
		AugerRate(Z, L3_M1P3_AUGER, NULL)+
		AugerRate(Z, L3_M1P4_AUGER, NULL)+
		AugerRate(Z, L3_M1P5_AUGER, NULL)+
		AugerRate(Z, L3_M1Q1_AUGER, NULL)+
		AugerRate(Z, L3_M1Q2_AUGER, NULL)+
		AugerRate(Z, L3_M1Q3_AUGER, NULL)+
		AugerRate(Z, L3_M2M1_AUGER, NULL)+
		AugerRate(Z, L3_M3M1_AUGER, NULL)+
		AugerRate(Z, L3_M4M1_AUGER, NULL)+
		AugerRate(Z, L3_M5M1_AUGER, NULL)
		));
	}
	return 0.0;
}

double PM2_get_cross_sections_constant_auger_only(int Z, int shell) {
	if (shell == K_SHELL) {
		return AugerYield(Z, K_SHELL, NULL) * (
		AugerRate(Z, K_L1M2_AUGER, NULL)+
		AugerRate(Z, K_L2M2_AUGER, NULL)+
		AugerRate(Z, K_L3M2_AUGER, NULL)+
		AugerRate(Z, K_M1M2_AUGER, NULL)+
		AugerRate(Z, K_M2L1_AUGER, NULL)+
		AugerRate(Z, K_M2L2_AUGER, NULL)+
		AugerRate(Z, K_M2L3_AUGER, NULL)+
		AugerRate(Z, K_M2M1_AUGER, NULL)+
		2 * AugerRate(Z, K_M2M2_AUGER, NULL)+
		AugerRate(Z, K_M2M3_AUGER, NULL)+
		AugerRate(Z, K_M2M4_AUGER, NULL)+
		AugerRate(Z, K_M2M5_AUGER, NULL)+
		AugerRate(Z, K_M2N1_AUGER, NULL)+
		AugerRate(Z, K_M2N2_AUGER, NULL)+
		AugerRate(Z, K_M2N3_AUGER, NULL)+
		AugerRate(Z, K_M2N4_AUGER, NULL)+
		AugerRate(Z, K_M2N5_AUGER, NULL)+
		AugerRate(Z, K_M2N6_AUGER, NULL)+
		AugerRate(Z, K_M2N7_AUGER, NULL)+
		AugerRate(Z, K_M2O1_AUGER, NULL)+
		AugerRate(Z, K_M2O2_AUGER, NULL)+
		AugerRate(Z, K_M2O3_AUGER, NULL)+
		AugerRate(Z, K_M2O4_AUGER, NULL)+
		AugerRate(Z, K_M2O5_AUGER, NULL)+
		AugerRate(Z, K_M2O6_AUGER, NULL)+
		AugerRate(Z, K_M2O7_AUGER, NULL)+
		AugerRate(Z, K_M2P1_AUGER, NULL)+
		AugerRate(Z, K_M2P2_AUGER, NULL)+
		AugerRate(Z, K_M2P3_AUGER, NULL)+
		AugerRate(Z, K_M2P4_AUGER, NULL)+
		AugerRate(Z, K_M2P5_AUGER, NULL)+
		AugerRate(Z, K_M2Q1_AUGER, NULL)+
		AugerRate(Z, K_M2Q2_AUGER, NULL)+
		AugerRate(Z, K_M2Q3_AUGER, NULL)+
		AugerRate(Z, K_M3M2_AUGER, NULL)+
		AugerRate(Z, K_M4M2_AUGER, NULL)+
		AugerRate(Z, K_M5M2_AUGER, NULL)
		);
	}
	else if (shell == L1_SHELL) {
		return AugerYield(Z, L1_SHELL, NULL) * (
		AugerRate(Z, L1_M1M2_AUGER, NULL)+
		AugerRate(Z, L1_M2M1_AUGER, NULL)+
		2 * AugerRate(Z, L1_M2M2_AUGER, NULL)+
		AugerRate(Z, L1_M2M3_AUGER, NULL)+
		AugerRate(Z, L1_M2M4_AUGER, NULL)+
		AugerRate(Z, L1_M2M5_AUGER, NULL)+
		AugerRate(Z, L1_M2N1_AUGER, NULL)+
		AugerRate(Z, L1_M2N2_AUGER, NULL)+
		AugerRate(Z, L1_M2N3_AUGER, NULL)+
		AugerRate(Z, L1_M2N4_AUGER, NULL)+
		AugerRate(Z, L1_M2N5_AUGER, NULL)+
		AugerRate(Z, L1_M2N6_AUGER, NULL)+
		AugerRate(Z, L1_M2N7_AUGER, NULL)+
		AugerRate(Z, L1_M2O1_AUGER, NULL)+
		AugerRate(Z, L1_M2O2_AUGER, NULL)+
		AugerRate(Z, L1_M2O3_AUGER, NULL)+
		AugerRate(Z, L1_M2O4_AUGER, NULL)+
		AugerRate(Z, L1_M2O5_AUGER, NULL)+
		AugerRate(Z, L1_M2O6_AUGER, NULL)+
		AugerRate(Z, L1_M2O7_AUGER, NULL)+
		AugerRate(Z, L1_M2P1_AUGER, NULL)+
		AugerRate(Z, L1_M2P2_AUGER, NULL)+
		AugerRate(Z, L1_M2P3_AUGER, NULL)+
		AugerRate(Z, L1_M2P4_AUGER, NULL)+
		AugerRate(Z, L1_M2P5_AUGER, NULL)+
		AugerRate(Z, L1_M2Q1_AUGER, NULL)+
		AugerRate(Z, L1_M2Q2_AUGER, NULL)+
		AugerRate(Z, L1_M2Q3_AUGER, NULL)+
		AugerRate(Z, L1_M3M2_AUGER, NULL)+
		AugerRate(Z, L1_M4M2_AUGER, NULL)+
		AugerRate(Z, L1_M5M2_AUGER, NULL)
		);
	}
	else if (shell == L2_SHELL) {
		return AugerYield(Z, L2_SHELL, NULL) * (
		AugerRate(Z, L2_M1M2_AUGER, NULL)+
		AugerRate(Z, L2_M2M1_AUGER, NULL)+
		2 * AugerRate(Z, L2_M2M2_AUGER, NULL)+
		AugerRate(Z, L2_M2M3_AUGER, NULL)+
		AugerRate(Z, L2_M2M4_AUGER, NULL)+
		AugerRate(Z, L2_M2M5_AUGER, NULL)+
		AugerRate(Z, L2_M2N1_AUGER, NULL)+
		AugerRate(Z, L2_M2N2_AUGER, NULL)+
		AugerRate(Z, L2_M2N3_AUGER, NULL)+
		AugerRate(Z, L2_M2N4_AUGER, NULL)+
		AugerRate(Z, L2_M2N5_AUGER, NULL)+
		AugerRate(Z, L2_M2N6_AUGER, NULL)+
		AugerRate(Z, L2_M2N7_AUGER, NULL)+
		AugerRate(Z, L2_M2O1_AUGER, NULL)+
		AugerRate(Z, L2_M2O2_AUGER, NULL)+
		AugerRate(Z, L2_M2O3_AUGER, NULL)+
		AugerRate(Z, L2_M2O4_AUGER, NULL)+
		AugerRate(Z, L2_M2O5_AUGER, NULL)+
		AugerRate(Z, L2_M2O6_AUGER, NULL)+
		AugerRate(Z, L2_M2O7_AUGER, NULL)+
		AugerRate(Z, L2_M2P1_AUGER, NULL)+
		AugerRate(Z, L2_M2P2_AUGER, NULL)+
		AugerRate(Z, L2_M2P3_AUGER, NULL)+
		AugerRate(Z, L2_M2P4_AUGER, NULL)+
		AugerRate(Z, L2_M2P5_AUGER, NULL)+
		AugerRate(Z, L2_M2Q1_AUGER, NULL)+
		AugerRate(Z, L2_M2Q2_AUGER, NULL)+
		AugerRate(Z, L2_M2Q3_AUGER, NULL)+
		AugerRate(Z, L2_M3M2_AUGER, NULL)+
		AugerRate(Z, L2_M4M2_AUGER, NULL)+
		AugerRate(Z, L2_M5M2_AUGER, NULL)
		);
	}
	else if (shell == L3_SHELL) {
		return AugerYield(Z, L3_SHELL, NULL) * (
		AugerRate(Z, L3_M1M2_AUGER, NULL)+
		AugerRate(Z, L3_M2M1_AUGER, NULL)+
		2 * AugerRate(Z, L3_M2M2_AUGER, NULL)+
		AugerRate(Z, L3_M2M3_AUGER, NULL)+
		AugerRate(Z, L3_M2M4_AUGER, NULL)+
		AugerRate(Z, L3_M2M5_AUGER, NULL)+
		AugerRate(Z, L3_M2N1_AUGER, NULL)+
		AugerRate(Z, L3_M2N2_AUGER, NULL)+
		AugerRate(Z, L3_M2N3_AUGER, NULL)+
		AugerRate(Z, L3_M2N4_AUGER, NULL)+
		AugerRate(Z, L3_M2N5_AUGER, NULL)+
		AugerRate(Z, L3_M2N6_AUGER, NULL)+
		AugerRate(Z, L3_M2N7_AUGER, NULL)+
		AugerRate(Z, L3_M2O1_AUGER, NULL)+
		AugerRate(Z, L3_M2O2_AUGER, NULL)+
		AugerRate(Z, L3_M2O3_AUGER, NULL)+
		AugerRate(Z, L3_M2O4_AUGER, NULL)+
		AugerRate(Z, L3_M2O5_AUGER, NULL)+
		AugerRate(Z, L3_M2O6_AUGER, NULL)+
		AugerRate(Z, L3_M2O7_AUGER, NULL)+
		AugerRate(Z, L3_M2P1_AUGER, NULL)+
		AugerRate(Z, L3_M2P2_AUGER, NULL)+
		AugerRate(Z, L3_M2P3_AUGER, NULL)+
		AugerRate(Z, L3_M2P4_AUGER, NULL)+
		AugerRate(Z, L3_M2P5_AUGER, NULL)+
		AugerRate(Z, L3_M2Q1_AUGER, NULL)+
		AugerRate(Z, L3_M2Q2_AUGER, NULL)+
		AugerRate(Z, L3_M2Q3_AUGER, NULL)+
		AugerRate(Z, L3_M3M2_AUGER, NULL)+
		AugerRate(Z, L3_M4M2_AUGER, NULL)+
		AugerRate(Z, L3_M5M2_AUGER, NULL)
		);
	}
	return 0.0;
}

double PM2_get_cross_sections_constant_full(int Z, int shell) {
	if (shell == K_SHELL) {
		return (FluorYield(Z, K_SHELL, NULL) * RadRate(Z, KM2_LINE, NULL)+
		AugerYield(Z, K_SHELL, NULL) * (
		AugerRate(Z, K_L1M2_AUGER, NULL)+
		AugerRate(Z, K_L2M2_AUGER, NULL)+
		AugerRate(Z, K_L3M2_AUGER, NULL)+
		AugerRate(Z, K_M1M2_AUGER, NULL)+
		AugerRate(Z, K_M2L1_AUGER, NULL)+
		AugerRate(Z, K_M2L2_AUGER, NULL)+
		AugerRate(Z, K_M2L3_AUGER, NULL)+
		AugerRate(Z, K_M2M1_AUGER, NULL)+
		2 * AugerRate(Z, K_M2M2_AUGER, NULL)+
		AugerRate(Z, K_M2M3_AUGER, NULL)+
		AugerRate(Z, K_M2M4_AUGER, NULL)+
		AugerRate(Z, K_M2M5_AUGER, NULL)+
		AugerRate(Z, K_M2N1_AUGER, NULL)+
		AugerRate(Z, K_M2N2_AUGER, NULL)+
		AugerRate(Z, K_M2N3_AUGER, NULL)+
		AugerRate(Z, K_M2N4_AUGER, NULL)+
		AugerRate(Z, K_M2N5_AUGER, NULL)+
		AugerRate(Z, K_M2N6_AUGER, NULL)+
		AugerRate(Z, K_M2N7_AUGER, NULL)+
		AugerRate(Z, K_M2O1_AUGER, NULL)+
		AugerRate(Z, K_M2O2_AUGER, NULL)+
		AugerRate(Z, K_M2O3_AUGER, NULL)+
		AugerRate(Z, K_M2O4_AUGER, NULL)+
		AugerRate(Z, K_M2O5_AUGER, NULL)+
		AugerRate(Z, K_M2O6_AUGER, NULL)+
		AugerRate(Z, K_M2O7_AUGER, NULL)+
		AugerRate(Z, K_M2P1_AUGER, NULL)+
		AugerRate(Z, K_M2P2_AUGER, NULL)+
		AugerRate(Z, K_M2P3_AUGER, NULL)+
		AugerRate(Z, K_M2P4_AUGER, NULL)+
		AugerRate(Z, K_M2P5_AUGER, NULL)+
		AugerRate(Z, K_M2Q1_AUGER, NULL)+
		AugerRate(Z, K_M2Q2_AUGER, NULL)+
		AugerRate(Z, K_M2Q3_AUGER, NULL)+
		AugerRate(Z, K_M3M2_AUGER, NULL)+
		AugerRate(Z, K_M4M2_AUGER, NULL)+
		AugerRate(Z, K_M5M2_AUGER, NULL)
		));
	}
	else if (shell == L1_SHELL) {
		return (FluorYield(Z, L1_SHELL, NULL) * RadRate(Z, L1M2_LINE, NULL) +
		AugerYield(Z, L1_SHELL, NULL) * (
		AugerRate(Z, L1_M1M2_AUGER, NULL)+
		AugerRate(Z, L1_M2M1_AUGER, NULL)+
		2 * AugerRate(Z, L1_M2M2_AUGER, NULL)+
		AugerRate(Z, L1_M2M3_AUGER, NULL)+
		AugerRate(Z, L1_M2M4_AUGER, NULL)+
		AugerRate(Z, L1_M2M5_AUGER, NULL)+
		AugerRate(Z, L1_M2N1_AUGER, NULL)+
		AugerRate(Z, L1_M2N2_AUGER, NULL)+
		AugerRate(Z, L1_M2N3_AUGER, NULL)+
		AugerRate(Z, L1_M2N4_AUGER, NULL)+
		AugerRate(Z, L1_M2N5_AUGER, NULL)+
		AugerRate(Z, L1_M2N6_AUGER, NULL)+
		AugerRate(Z, L1_M2N7_AUGER, NULL)+
		AugerRate(Z, L1_M2O1_AUGER, NULL)+
		AugerRate(Z, L1_M2O2_AUGER, NULL)+
		AugerRate(Z, L1_M2O3_AUGER, NULL)+
		AugerRate(Z, L1_M2O4_AUGER, NULL)+
		AugerRate(Z, L1_M2O5_AUGER, NULL)+
		AugerRate(Z, L1_M2O6_AUGER, NULL)+
		AugerRate(Z, L1_M2O7_AUGER, NULL)+
		AugerRate(Z, L1_M2P1_AUGER, NULL)+
		AugerRate(Z, L1_M2P2_AUGER, NULL)+
		AugerRate(Z, L1_M2P3_AUGER, NULL)+
		AugerRate(Z, L1_M2P4_AUGER, NULL)+
		AugerRate(Z, L1_M2P5_AUGER, NULL)+
		AugerRate(Z, L1_M2Q1_AUGER, NULL)+
		AugerRate(Z, L1_M2Q2_AUGER, NULL)+
		AugerRate(Z, L1_M2Q3_AUGER, NULL)+
		AugerRate(Z, L1_M3M2_AUGER, NULL)+
		AugerRate(Z, L1_M4M2_AUGER, NULL)+
		AugerRate(Z, L1_M5M2_AUGER, NULL)
		));
	} 
	else if (shell == L2_SHELL) {
		return (FluorYield(Z, L2_SHELL, NULL) * RadRate(Z, L2M2_LINE, NULL) +
		AugerYield(Z, L2_SHELL, NULL) * (
		AugerRate(Z, L2_M1M2_AUGER, NULL)+
		AugerRate(Z, L2_M2M1_AUGER, NULL)+
		2 * AugerRate(Z, L2_M2M2_AUGER, NULL)+
		AugerRate(Z, L2_M2M3_AUGER, NULL)+
		AugerRate(Z, L2_M2M4_AUGER, NULL)+
		AugerRate(Z, L2_M2M5_AUGER, NULL)+
		AugerRate(Z, L2_M2N1_AUGER, NULL)+
		AugerRate(Z, L2_M2N2_AUGER, NULL)+
		AugerRate(Z, L2_M2N3_AUGER, NULL)+
		AugerRate(Z, L2_M2N4_AUGER, NULL)+
		AugerRate(Z, L2_M2N5_AUGER, NULL)+
		AugerRate(Z, L2_M2N6_AUGER, NULL)+
		AugerRate(Z, L2_M2N7_AUGER, NULL)+
		AugerRate(Z, L2_M2O1_AUGER, NULL)+
		AugerRate(Z, L2_M2O2_AUGER, NULL)+
		AugerRate(Z, L2_M2O3_AUGER, NULL)+
		AugerRate(Z, L2_M2O4_AUGER, NULL)+
		AugerRate(Z, L2_M2O5_AUGER, NULL)+
		AugerRate(Z, L2_M2O6_AUGER, NULL)+
		AugerRate(Z, L2_M2O7_AUGER, NULL)+
		AugerRate(Z, L2_M2P1_AUGER, NULL)+
		AugerRate(Z, L2_M2P2_AUGER, NULL)+
		AugerRate(Z, L2_M2P3_AUGER, NULL)+
		AugerRate(Z, L2_M2P4_AUGER, NULL)+
		AugerRate(Z, L2_M2P5_AUGER, NULL)+
		AugerRate(Z, L2_M2Q1_AUGER, NULL)+
		AugerRate(Z, L2_M2Q2_AUGER, NULL)+
		AugerRate(Z, L2_M2Q3_AUGER, NULL)+
		AugerRate(Z, L2_M3M2_AUGER, NULL)+
		AugerRate(Z, L2_M4M2_AUGER, NULL)+
		AugerRate(Z, L2_M5M2_AUGER, NULL)
		));
	}
	else if (shell == L3_SHELL) {
		return (FluorYield(Z, L3_SHELL, NULL) * RadRate(Z, L3M2_LINE, NULL) +
		AugerYield(Z, L3_SHELL, NULL) * (
		AugerRate(Z, L3_M1M2_AUGER, NULL)+
		AugerRate(Z, L3_M2M1_AUGER, NULL)+
		2 * AugerRate(Z, L3_M2M2_AUGER, NULL)+
		AugerRate(Z, L3_M2M3_AUGER, NULL)+
		AugerRate(Z, L3_M2M4_AUGER, NULL)+
		AugerRate(Z, L3_M2M5_AUGER, NULL)+
		AugerRate(Z, L3_M2N1_AUGER, NULL)+
		AugerRate(Z, L3_M2N2_AUGER, NULL)+
		AugerRate(Z, L3_M2N3_AUGER, NULL)+
		AugerRate(Z, L3_M2N4_AUGER, NULL)+
		AugerRate(Z, L3_M2N5_AUGER, NULL)+
		AugerRate(Z, L3_M2N6_AUGER, NULL)+
		AugerRate(Z, L3_M2N7_AUGER, NULL)+
		AugerRate(Z, L3_M2O1_AUGER, NULL)+
		AugerRate(Z, L3_M2O2_AUGER, NULL)+
		AugerRate(Z, L3_M2O3_AUGER, NULL)+
		AugerRate(Z, L3_M2O4_AUGER, NULL)+
		AugerRate(Z, L3_M2O5_AUGER, NULL)+
		AugerRate(Z, L3_M2O6_AUGER, NULL)+
		AugerRate(Z, L3_M2O7_AUGER, NULL)+
		AugerRate(Z, L3_M2P1_AUGER, NULL)+
		AugerRate(Z, L3_M2P2_AUGER, NULL)+
		AugerRate(Z, L3_M2P3_AUGER, NULL)+
		AugerRate(Z, L3_M2P4_AUGER, NULL)+
		AugerRate(Z, L3_M2P5_AUGER, NULL)+
		AugerRate(Z, L3_M2Q1_AUGER, NULL)+
		AugerRate(Z, L3_M2Q2_AUGER, NULL)+
		AugerRate(Z, L3_M2Q3_AUGER, NULL)+
		AugerRate(Z, L3_M3M2_AUGER, NULL)+
		AugerRate(Z, L3_M4M2_AUGER, NULL)+
		AugerRate(Z, L3_M5M2_AUGER, NULL)
		));
	}

	return 0.0;
}

double PM3_get_cross_sections_constant_auger_only(int Z, int shell) {
	if (shell == K_SHELL) {
		return AugerYield(Z, K_SHELL, NULL) * (
		AugerRate(Z, K_L1M3_AUGER, NULL)+
		AugerRate(Z, K_L2M3_AUGER, NULL)+
		AugerRate(Z, K_L3M3_AUGER, NULL)+
		AugerRate(Z, K_M1M3_AUGER, NULL)+
		AugerRate(Z, K_M2M3_AUGER, NULL)+
		AugerRate(Z, K_M3L1_AUGER, NULL)+
		AugerRate(Z, K_M3L2_AUGER, NULL)+
		AugerRate(Z, K_M3L3_AUGER, NULL)+
		AugerRate(Z, K_M3M1_AUGER, NULL)+
		AugerRate(Z, K_M3M2_AUGER, NULL)+
		2 * AugerRate(Z, K_M3M3_AUGER, NULL)+
		AugerRate(Z, K_M3M4_AUGER, NULL)+
		AugerRate(Z, K_M3M5_AUGER, NULL)+
		AugerRate(Z, K_M3N1_AUGER, NULL)+
		AugerRate(Z, K_M3N2_AUGER, NULL)+
		AugerRate(Z, K_M3N3_AUGER, NULL)+
		AugerRate(Z, K_M3N4_AUGER, NULL)+
		AugerRate(Z, K_M3N5_AUGER, NULL)+
		AugerRate(Z, K_M3N6_AUGER, NULL)+
		AugerRate(Z, K_M3N7_AUGER, NULL)+
		AugerRate(Z, K_M3O1_AUGER, NULL)+
		AugerRate(Z, K_M3O2_AUGER, NULL)+
		AugerRate(Z, K_M3O3_AUGER, NULL)+
		AugerRate(Z, K_M3O4_AUGER, NULL)+
		AugerRate(Z, K_M3O5_AUGER, NULL)+
		AugerRate(Z, K_M3O6_AUGER, NULL)+
		AugerRate(Z, K_M3O7_AUGER, NULL)+
		AugerRate(Z, K_M3P1_AUGER, NULL)+
		AugerRate(Z, K_M3P2_AUGER, NULL)+
		AugerRate(Z, K_M3P3_AUGER, NULL)+
		AugerRate(Z, K_M3P4_AUGER, NULL)+
		AugerRate(Z, K_M3P5_AUGER, NULL)+
		AugerRate(Z, K_M3Q1_AUGER, NULL)+
		AugerRate(Z, K_M3Q2_AUGER, NULL)+
		AugerRate(Z, K_M3Q3_AUGER, NULL)+
		AugerRate(Z, K_M4M3_AUGER, NULL)+
		AugerRate(Z, K_M5M3_AUGER, NULL)
		);
	}
	else if (shell == L1_SHELL) {
		return AugerYield(Z, L1_SHELL, NULL) * (
		AugerRate(Z, L1_M1M3_AUGER, NULL)+
		AugerRate(Z, L1_M2M3_AUGER, NULL)+
		AugerRate(Z, L1_M3M1_AUGER, NULL)+
		AugerRate(Z, L1_M3M2_AUGER, NULL)+
		2 * AugerRate(Z, L1_M3M3_AUGER, NULL)+
		AugerRate(Z, L1_M3M4_AUGER, NULL)+
		AugerRate(Z, L1_M3M5_AUGER, NULL)+
		AugerRate(Z, L1_M3N1_AUGER, NULL)+
		AugerRate(Z, L1_M3N2_AUGER, NULL)+
		AugerRate(Z, L1_M3N3_AUGER, NULL)+
		AugerRate(Z, L1_M3N4_AUGER, NULL)+
		AugerRate(Z, L1_M3N5_AUGER, NULL)+
		AugerRate(Z, L1_M3N6_AUGER, NULL)+
		AugerRate(Z, L1_M3N7_AUGER, NULL)+
		AugerRate(Z, L1_M3O1_AUGER, NULL)+
		AugerRate(Z, L1_M3O2_AUGER, NULL)+
		AugerRate(Z, L1_M3O3_AUGER, NULL)+
		AugerRate(Z, L1_M3O4_AUGER, NULL)+
		AugerRate(Z, L1_M3O5_AUGER, NULL)+
		AugerRate(Z, L1_M3O6_AUGER, NULL)+
		AugerRate(Z, L1_M3O7_AUGER, NULL)+
		AugerRate(Z, L1_M3P1_AUGER, NULL)+
		AugerRate(Z, L1_M3P2_AUGER, NULL)+
		AugerRate(Z, L1_M3P3_AUGER, NULL)+
		AugerRate(Z, L1_M3P4_AUGER, NULL)+
		AugerRate(Z, L1_M3P5_AUGER, NULL)+
		AugerRate(Z, L1_M3Q1_AUGER, NULL)+
		AugerRate(Z, L1_M3Q2_AUGER, NULL)+
		AugerRate(Z, L1_M3Q3_AUGER, NULL)+
		AugerRate(Z, L1_M4M3_AUGER, NULL)+
		AugerRate(Z, L1_M5M3_AUGER, NULL)
		);
	}
	else if (shell == L2_SHELL) {
		return AugerYield(Z, L2_SHELL, NULL) * (
		AugerRate(Z, L2_M1M3_AUGER, NULL)+
		AugerRate(Z, L2_M2M3_AUGER, NULL)+
		AugerRate(Z, L2_M3M1_AUGER, NULL)+
		AugerRate(Z, L2_M3M2_AUGER, NULL)+
		2 * AugerRate(Z, L2_M3M3_AUGER, NULL)+
		AugerRate(Z, L2_M3M4_AUGER, NULL)+
		AugerRate(Z, L2_M3M5_AUGER, NULL)+
		AugerRate(Z, L2_M3N1_AUGER, NULL)+
		AugerRate(Z, L2_M3N2_AUGER, NULL)+
		AugerRate(Z, L2_M3N3_AUGER, NULL)+
		AugerRate(Z, L2_M3N4_AUGER, NULL)+
		AugerRate(Z, L2_M3N5_AUGER, NULL)+
		AugerRate(Z, L2_M3N6_AUGER, NULL)+
		AugerRate(Z, L2_M3N7_AUGER, NULL)+
		AugerRate(Z, L2_M3O1_AUGER, NULL)+
		AugerRate(Z, L2_M3O2_AUGER, NULL)+
		AugerRate(Z, L2_M3O3_AUGER, NULL)+
		AugerRate(Z, L2_M3O4_AUGER, NULL)+
		AugerRate(Z, L2_M3O5_AUGER, NULL)+
		AugerRate(Z, L2_M3O6_AUGER, NULL)+
		AugerRate(Z, L2_M3O7_AUGER, NULL)+
		AugerRate(Z, L2_M3P1_AUGER, NULL)+
		AugerRate(Z, L2_M3P2_AUGER, NULL)+
		AugerRate(Z, L2_M3P3_AUGER, NULL)+
		AugerRate(Z, L2_M3P4_AUGER, NULL)+
		AugerRate(Z, L2_M3P5_AUGER, NULL)+
		AugerRate(Z, L2_M3Q1_AUGER, NULL)+
		AugerRate(Z, L2_M3Q2_AUGER, NULL)+
		AugerRate(Z, L2_M3Q3_AUGER, NULL)+
		AugerRate(Z, L2_M4M3_AUGER, NULL)+
		AugerRate(Z, L2_M5M3_AUGER, NULL)
		);
	}
	else if (shell == L3_SHELL) {
		return AugerYield(Z, L3_SHELL, NULL) * (
		AugerRate(Z, L3_M1M3_AUGER, NULL)+
		AugerRate(Z, L3_M2M3_AUGER, NULL)+
		AugerRate(Z, L3_M3M1_AUGER, NULL)+
		AugerRate(Z, L3_M3M2_AUGER, NULL)+
		2 * AugerRate(Z, L3_M3M3_AUGER, NULL)+
		AugerRate(Z, L3_M3M4_AUGER, NULL)+
		AugerRate(Z, L3_M3M5_AUGER, NULL)+
		AugerRate(Z, L3_M3N1_AUGER, NULL)+
		AugerRate(Z, L3_M3N2_AUGER, NULL)+
		AugerRate(Z, L3_M3N3_AUGER, NULL)+
		AugerRate(Z, L3_M3N4_AUGER, NULL)+
		AugerRate(Z, L3_M3N5_AUGER, NULL)+
		AugerRate(Z, L3_M3N6_AUGER, NULL)+
		AugerRate(Z, L3_M3N7_AUGER, NULL)+
		AugerRate(Z, L3_M3O1_AUGER, NULL)+
		AugerRate(Z, L3_M3O2_AUGER, NULL)+
		AugerRate(Z, L3_M3O3_AUGER, NULL)+
		AugerRate(Z, L3_M3O4_AUGER, NULL)+
		AugerRate(Z, L3_M3O5_AUGER, NULL)+
		AugerRate(Z, L3_M3O6_AUGER, NULL)+
		AugerRate(Z, L3_M3O7_AUGER, NULL)+
		AugerRate(Z, L3_M3P1_AUGER, NULL)+
		AugerRate(Z, L3_M3P2_AUGER, NULL)+
		AugerRate(Z, L3_M3P3_AUGER, NULL)+
		AugerRate(Z, L3_M3P4_AUGER, NULL)+
		AugerRate(Z, L3_M3P5_AUGER, NULL)+
		AugerRate(Z, L3_M3Q1_AUGER, NULL)+
		AugerRate(Z, L3_M3Q2_AUGER, NULL)+
		AugerRate(Z, L3_M3Q3_AUGER, NULL)+
		AugerRate(Z, L3_M4M3_AUGER, NULL)+
		AugerRate(Z, L3_M5M3_AUGER, NULL)
		);
	}
	return 0.0;
}

double PM3_get_cross_sections_constant_full(int Z, int shell) {
	if (shell == K_SHELL) {
		return (FluorYield(Z, K_SHELL, NULL) * RadRate(Z, KM3_LINE, NULL) +
		AugerYield(Z, K_SHELL, NULL) * (
		AugerRate(Z, K_L1M3_AUGER, NULL)+
		AugerRate(Z, K_L2M3_AUGER, NULL)+
		AugerRate(Z, K_L3M3_AUGER, NULL)+
		AugerRate(Z, K_M1M3_AUGER, NULL)+
		AugerRate(Z, K_M2M3_AUGER, NULL)+
		AugerRate(Z, K_M3L1_AUGER, NULL)+
		AugerRate(Z, K_M3L2_AUGER, NULL)+
		AugerRate(Z, K_M3L3_AUGER, NULL)+
		AugerRate(Z, K_M3M1_AUGER, NULL)+
		AugerRate(Z, K_M3M2_AUGER, NULL)+
		2 * AugerRate(Z, K_M3M3_AUGER, NULL)+
		AugerRate(Z, K_M3M4_AUGER, NULL)+
		AugerRate(Z, K_M3M5_AUGER, NULL)+
		AugerRate(Z, K_M3N1_AUGER, NULL)+
		AugerRate(Z, K_M3N2_AUGER, NULL)+
		AugerRate(Z, K_M3N3_AUGER, NULL)+
		AugerRate(Z, K_M3N4_AUGER, NULL)+
		AugerRate(Z, K_M3N5_AUGER, NULL)+
		AugerRate(Z, K_M3N6_AUGER, NULL)+
		AugerRate(Z, K_M3N7_AUGER, NULL)+
		AugerRate(Z, K_M3O1_AUGER, NULL)+
		AugerRate(Z, K_M3O2_AUGER, NULL)+
		AugerRate(Z, K_M3O3_AUGER, NULL)+
		AugerRate(Z, K_M3O4_AUGER, NULL)+
		AugerRate(Z, K_M3O5_AUGER, NULL)+
		AugerRate(Z, K_M3O6_AUGER, NULL)+
		AugerRate(Z, K_M3O7_AUGER, NULL)+
		AugerRate(Z, K_M3P1_AUGER, NULL)+
		AugerRate(Z, K_M3P2_AUGER, NULL)+
		AugerRate(Z, K_M3P3_AUGER, NULL)+
		AugerRate(Z, K_M3P4_AUGER, NULL)+
		AugerRate(Z, K_M3P5_AUGER, NULL)+
		AugerRate(Z, K_M3Q1_AUGER, NULL)+
		AugerRate(Z, K_M3Q2_AUGER, NULL)+
		AugerRate(Z, K_M3Q3_AUGER, NULL)+
		AugerRate(Z, K_M4M3_AUGER, NULL)+
		AugerRate(Z, K_M5M3_AUGER, NULL)
		));
	}
	else if (shell == L1_SHELL) {
		return (FluorYield(Z, L1_SHELL, NULL) * RadRate(Z, L1M3_LINE, NULL) +
		AugerYield(Z, L1_SHELL, NULL) * (
		AugerRate(Z, L1_M1M3_AUGER, NULL)+
		AugerRate(Z, L1_M2M3_AUGER, NULL)+
		AugerRate(Z, L1_M3M1_AUGER, NULL)+
		AugerRate(Z, L1_M3M2_AUGER, NULL)+
		2 * AugerRate(Z, L1_M3M3_AUGER, NULL)+
		AugerRate(Z, L1_M3M4_AUGER, NULL)+
		AugerRate(Z, L1_M3M5_AUGER, NULL)+
		AugerRate(Z, L1_M3N1_AUGER, NULL)+
		AugerRate(Z, L1_M3N2_AUGER, NULL)+
		AugerRate(Z, L1_M3N3_AUGER, NULL)+
		AugerRate(Z, L1_M3N4_AUGER, NULL)+
		AugerRate(Z, L1_M3N5_AUGER, NULL)+
		AugerRate(Z, L1_M3N6_AUGER, NULL)+
		AugerRate(Z, L1_M3N7_AUGER, NULL)+
		AugerRate(Z, L1_M3O1_AUGER, NULL)+
		AugerRate(Z, L1_M3O2_AUGER, NULL)+
		AugerRate(Z, L1_M3O3_AUGER, NULL)+
		AugerRate(Z, L1_M3O4_AUGER, NULL)+
		AugerRate(Z, L1_M3O5_AUGER, NULL)+
		AugerRate(Z, L1_M3O6_AUGER, NULL)+
		AugerRate(Z, L1_M3O7_AUGER, NULL)+
		AugerRate(Z, L1_M3P1_AUGER, NULL)+
		AugerRate(Z, L1_M3P2_AUGER, NULL)+
		AugerRate(Z, L1_M3P3_AUGER, NULL)+
		AugerRate(Z, L1_M3P4_AUGER, NULL)+
		AugerRate(Z, L1_M3P5_AUGER, NULL)+
		AugerRate(Z, L1_M3Q1_AUGER, NULL)+
		AugerRate(Z, L1_M3Q2_AUGER, NULL)+
		AugerRate(Z, L1_M3Q3_AUGER, NULL)+
		AugerRate(Z, L1_M4M3_AUGER, NULL)+
		AugerRate(Z, L1_M5M3_AUGER, NULL)
		));
	}
	else if (shell == L2_SHELL) {
		return (FluorYield(Z, L2_SHELL, NULL) * RadRate(Z, L2M3_LINE, NULL) +
		AugerYield(Z, L2_SHELL, NULL) * (
		AugerRate(Z, L2_M1M3_AUGER, NULL)+
		AugerRate(Z, L2_M2M3_AUGER, NULL)+
		AugerRate(Z, L2_M3M1_AUGER, NULL)+
		AugerRate(Z, L2_M3M2_AUGER, NULL)+
		2 * AugerRate(Z, L2_M3M3_AUGER, NULL)+
		AugerRate(Z, L2_M3M4_AUGER, NULL)+
		AugerRate(Z, L2_M3M5_AUGER, NULL)+
		AugerRate(Z, L2_M3N1_AUGER, NULL)+
		AugerRate(Z, L2_M3N2_AUGER, NULL)+
		AugerRate(Z, L2_M3N3_AUGER, NULL)+
		AugerRate(Z, L2_M3N4_AUGER, NULL)+
		AugerRate(Z, L2_M3N5_AUGER, NULL)+
		AugerRate(Z, L2_M3N6_AUGER, NULL)+
		AugerRate(Z, L2_M3N7_AUGER, NULL)+
		AugerRate(Z, L2_M3O1_AUGER, NULL)+
		AugerRate(Z, L2_M3O2_AUGER, NULL)+
		AugerRate(Z, L2_M3O3_AUGER, NULL)+
		AugerRate(Z, L2_M3O4_AUGER, NULL)+
		AugerRate(Z, L2_M3O5_AUGER, NULL)+
		AugerRate(Z, L2_M3O6_AUGER, NULL)+
		AugerRate(Z, L2_M3O7_AUGER, NULL)+
		AugerRate(Z, L2_M3P1_AUGER, NULL)+
		AugerRate(Z, L2_M3P2_AUGER, NULL)+
		AugerRate(Z, L2_M3P3_AUGER, NULL)+
		AugerRate(Z, L2_M3P4_AUGER, NULL)+
		AugerRate(Z, L2_M3P5_AUGER, NULL)+
		AugerRate(Z, L2_M3Q1_AUGER, NULL)+
		AugerRate(Z, L2_M3Q2_AUGER, NULL)+
		AugerRate(Z, L2_M3Q3_AUGER, NULL)+
		AugerRate(Z, L2_M4M3_AUGER, NULL)+
		AugerRate(Z, L2_M5M3_AUGER, NULL)
		));
	}
	else if (shell == L3_SHELL) {
		return (FluorYield(Z, L3_SHELL, NULL) * RadRate(Z, L3M3_LINE, NULL) +
		AugerYield(Z, L3_SHELL, NULL) * (
		AugerRate(Z, L3_M1M3_AUGER, NULL)+
		AugerRate(Z, L3_M2M3_AUGER, NULL)+
		AugerRate(Z, L3_M3M1_AUGER, NULL)+
		AugerRate(Z, L3_M3M2_AUGER, NULL)+
		2 * AugerRate(Z, L3_M3M3_AUGER, NULL)+
		AugerRate(Z, L3_M3M4_AUGER, NULL)+
		AugerRate(Z, L3_M3M5_AUGER, NULL)+
		AugerRate(Z, L3_M3N1_AUGER, NULL)+
		AugerRate(Z, L3_M3N2_AUGER, NULL)+
		AugerRate(Z, L3_M3N3_AUGER, NULL)+
		AugerRate(Z, L3_M3N4_AUGER, NULL)+
		AugerRate(Z, L3_M3N5_AUGER, NULL)+
		AugerRate(Z, L3_M3N6_AUGER, NULL)+
		AugerRate(Z, L3_M3N7_AUGER, NULL)+
		AugerRate(Z, L3_M3O1_AUGER, NULL)+
		AugerRate(Z, L3_M3O2_AUGER, NULL)+
		AugerRate(Z, L3_M3O3_AUGER, NULL)+
		AugerRate(Z, L3_M3O4_AUGER, NULL)+
		AugerRate(Z, L3_M3O5_AUGER, NULL)+
		AugerRate(Z, L3_M3O6_AUGER, NULL)+
		AugerRate(Z, L3_M3O7_AUGER, NULL)+
		AugerRate(Z, L3_M3P1_AUGER, NULL)+
		AugerRate(Z, L3_M3P2_AUGER, NULL)+
		AugerRate(Z, L3_M3P3_AUGER, NULL)+
		AugerRate(Z, L3_M3P4_AUGER, NULL)+
		AugerRate(Z, L3_M3P5_AUGER, NULL)+
		AugerRate(Z, L3_M3Q1_AUGER, NULL)+
		AugerRate(Z, L3_M3Q2_AUGER, NULL)+
		AugerRate(Z, L3_M3Q3_AUGER, NULL)+
		AugerRate(Z, L3_M4M3_AUGER, NULL)+
		AugerRate(Z, L3_M5M3_AUGER, NULL)
		));
	}
	return 0.0;
}

double PM4_get_cross_sections_constant_auger_only(int Z, int shell) {
	if (shell == K_SHELL) {
		return AugerYield(Z, K_SHELL, NULL)* (
		AugerRate(Z, K_L1M4_AUGER, NULL)+
		AugerRate(Z, K_L2M4_AUGER, NULL)+
		AugerRate(Z, K_L3M4_AUGER, NULL)+
		AugerRate(Z, K_M1M4_AUGER, NULL)+
		AugerRate(Z, K_M2M4_AUGER, NULL)+
		AugerRate(Z, K_M3M4_AUGER, NULL)+
		AugerRate(Z, K_M4L1_AUGER, NULL)+
		AugerRate(Z, K_M4L2_AUGER, NULL)+
		AugerRate(Z, K_M4L3_AUGER, NULL)+
		AugerRate(Z, K_M4M1_AUGER, NULL)+
		AugerRate(Z, K_M4M2_AUGER, NULL)+
		AugerRate(Z, K_M4M3_AUGER, NULL)+
		2 * AugerRate(Z, K_M4M4_AUGER, NULL)+
		AugerRate(Z, K_M4M5_AUGER, NULL)+
		AugerRate(Z, K_M4N1_AUGER, NULL)+
		AugerRate(Z, K_M4N2_AUGER, NULL)+
		AugerRate(Z, K_M4N3_AUGER, NULL)+
		AugerRate(Z, K_M4N4_AUGER, NULL)+
		AugerRate(Z, K_M4N5_AUGER, NULL)+
		AugerRate(Z, K_M4N6_AUGER, NULL)+
		AugerRate(Z, K_M4N7_AUGER, NULL)+
		AugerRate(Z, K_M4O1_AUGER, NULL)+
		AugerRate(Z, K_M4O2_AUGER, NULL)+
		AugerRate(Z, K_M4O3_AUGER, NULL)+
		AugerRate(Z, K_M4O4_AUGER, NULL)+
		AugerRate(Z, K_M4O5_AUGER, NULL)+
		AugerRate(Z, K_M4O6_AUGER, NULL)+
		AugerRate(Z, K_M4O7_AUGER, NULL)+
		AugerRate(Z, K_M4P1_AUGER, NULL)+
		AugerRate(Z, K_M4P2_AUGER, NULL)+
		AugerRate(Z, K_M4P3_AUGER, NULL)+
		AugerRate(Z, K_M4P4_AUGER, NULL)+
		AugerRate(Z, K_M4P5_AUGER, NULL)+
		AugerRate(Z, K_M4Q1_AUGER, NULL)+
		AugerRate(Z, K_M4Q2_AUGER, NULL)+
		AugerRate(Z, K_M4Q3_AUGER, NULL)+
		AugerRate(Z, K_M5M4_AUGER, NULL)
		);
	}
	else if (shell == L1_SHELL) {
		return AugerYield(Z, L1_SHELL, NULL) * (
		AugerRate(Z, L1_M1M4_AUGER, NULL)+
		AugerRate(Z, L1_M2M4_AUGER, NULL)+
		AugerRate(Z, L1_M3M4_AUGER, NULL)+
		AugerRate(Z, L1_M4M1_AUGER, NULL)+
		AugerRate(Z, L1_M4M2_AUGER, NULL)+
		AugerRate(Z, L1_M4M3_AUGER, NULL)+
		2 * AugerRate(Z, L1_M4M4_AUGER, NULL)+
		AugerRate(Z, L1_M4M5_AUGER, NULL)+
		AugerRate(Z, L1_M4N1_AUGER, NULL)+
		AugerRate(Z, L1_M4N2_AUGER, NULL)+
		AugerRate(Z, L1_M4N3_AUGER, NULL)+
		AugerRate(Z, L1_M4N4_AUGER, NULL)+
		AugerRate(Z, L1_M4N5_AUGER, NULL)+
		AugerRate(Z, L1_M4N6_AUGER, NULL)+
		AugerRate(Z, L1_M4N7_AUGER, NULL)+
		AugerRate(Z, L1_M4O1_AUGER, NULL)+
		AugerRate(Z, L1_M4O2_AUGER, NULL)+
		AugerRate(Z, L1_M4O3_AUGER, NULL)+
		AugerRate(Z, L1_M4O4_AUGER, NULL)+
		AugerRate(Z, L1_M4O5_AUGER, NULL)+
		AugerRate(Z, L1_M4O6_AUGER, NULL)+
		AugerRate(Z, L1_M4O7_AUGER, NULL)+
		AugerRate(Z, L1_M4P1_AUGER, NULL)+
		AugerRate(Z, L1_M4P2_AUGER, NULL)+
		AugerRate(Z, L1_M4P3_AUGER, NULL)+
		AugerRate(Z, L1_M4P4_AUGER, NULL)+
		AugerRate(Z, L1_M4P5_AUGER, NULL)+
		AugerRate(Z, L1_M4Q1_AUGER, NULL)+
		AugerRate(Z, L1_M4Q2_AUGER, NULL)+
		AugerRate(Z, L1_M4Q3_AUGER, NULL)+
		AugerRate(Z, L1_M5M4_AUGER, NULL)
		);
	}
	else if (shell == L2_SHELL) {
		return AugerYield(Z, L2_SHELL, NULL) * (
		AugerRate(Z, L2_M1M4_AUGER, NULL)+
		AugerRate(Z, L2_M2M4_AUGER, NULL)+
		AugerRate(Z, L2_M3M4_AUGER, NULL)+
		AugerRate(Z, L2_M4M1_AUGER, NULL)+
		AugerRate(Z, L2_M4M2_AUGER, NULL)+
		AugerRate(Z, L2_M4M3_AUGER, NULL)+
		2 * AugerRate(Z, L2_M4M4_AUGER, NULL)+
		AugerRate(Z, L2_M4M5_AUGER, NULL)+
		AugerRate(Z, L2_M4N1_AUGER, NULL)+
		AugerRate(Z, L2_M4N2_AUGER, NULL)+
		AugerRate(Z, L2_M4N3_AUGER, NULL)+
		AugerRate(Z, L2_M4N4_AUGER, NULL)+
		AugerRate(Z, L2_M4N5_AUGER, NULL)+
		AugerRate(Z, L2_M4N6_AUGER, NULL)+
		AugerRate(Z, L2_M4N7_AUGER, NULL)+
		AugerRate(Z, L2_M4O1_AUGER, NULL)+
		AugerRate(Z, L2_M4O2_AUGER, NULL)+
		AugerRate(Z, L2_M4O3_AUGER, NULL)+
		AugerRate(Z, L2_M4O4_AUGER, NULL)+
		AugerRate(Z, L2_M4O5_AUGER, NULL)+
		AugerRate(Z, L2_M4O6_AUGER, NULL)+
		AugerRate(Z, L2_M4O7_AUGER, NULL)+
		AugerRate(Z, L2_M4P1_AUGER, NULL)+
		AugerRate(Z, L2_M4P2_AUGER, NULL)+
		AugerRate(Z, L2_M4P3_AUGER, NULL)+
		AugerRate(Z, L2_M4P4_AUGER, NULL)+
		AugerRate(Z, L2_M4P5_AUGER, NULL)+
		AugerRate(Z, L2_M4Q1_AUGER, NULL)+
		AugerRate(Z, L2_M4Q2_AUGER, NULL)+
		AugerRate(Z, L2_M4Q3_AUGER, NULL)+
		AugerRate(Z, L2_M5M4_AUGER, NULL)
		);
	}
	else if (shell == L3_SHELL) {
		return AugerYield(Z, L3_SHELL, NULL) * (
		AugerRate(Z, L3_M1M4_AUGER, NULL)+
		AugerRate(Z, L3_M2M4_AUGER, NULL)+
		AugerRate(Z, L3_M3M4_AUGER, NULL)+
		AugerRate(Z, L3_M4M1_AUGER, NULL)+
		AugerRate(Z, L3_M4M2_AUGER, NULL)+
		AugerRate(Z, L3_M4M3_AUGER, NULL)+
		2 * AugerRate(Z, L3_M4M4_AUGER, NULL)+
		AugerRate(Z, L3_M4M5_AUGER, NULL)+
		AugerRate(Z, L3_M4N1_AUGER, NULL)+
		AugerRate(Z, L3_M4N2_AUGER, NULL)+
		AugerRate(Z, L3_M4N3_AUGER, NULL)+
		AugerRate(Z, L3_M4N4_AUGER, NULL)+
		AugerRate(Z, L3_M4N5_AUGER, NULL)+
		AugerRate(Z, L3_M4N6_AUGER, NULL)+
		AugerRate(Z, L3_M4N7_AUGER, NULL)+
		AugerRate(Z, L3_M4O1_AUGER, NULL)+
		AugerRate(Z, L3_M4O2_AUGER, NULL)+
		AugerRate(Z, L3_M4O3_AUGER, NULL)+
		AugerRate(Z, L3_M4O4_AUGER, NULL)+
		AugerRate(Z, L3_M4O5_AUGER, NULL)+
		AugerRate(Z, L3_M4O6_AUGER, NULL)+
		AugerRate(Z, L3_M4O7_AUGER, NULL)+
		AugerRate(Z, L3_M4P1_AUGER, NULL)+
		AugerRate(Z, L3_M4P2_AUGER, NULL)+
		AugerRate(Z, L3_M4P3_AUGER, NULL)+
		AugerRate(Z, L3_M4P4_AUGER, NULL)+
		AugerRate(Z, L3_M4P5_AUGER, NULL)+
		AugerRate(Z, L3_M4Q1_AUGER, NULL)+
		AugerRate(Z, L3_M4Q2_AUGER, NULL)+
		AugerRate(Z, L3_M4Q3_AUGER, NULL)+
		AugerRate(Z, L3_M5M4_AUGER, NULL)
		);
	}
	return 0.0;
}

double PM4_get_cross_sections_constant_full(int Z, int shell) {
	if (shell == K_SHELL) {
		return (FluorYield(Z, K_SHELL, NULL) * RadRate(Z, KM4_LINE, NULL) +
		AugerYield(Z, K_SHELL, NULL ) * (
		AugerRate(Z, K_L1M4_AUGER, NULL)+
		AugerRate(Z, K_L2M4_AUGER, NULL)+
		AugerRate(Z, K_L3M4_AUGER, NULL)+
		AugerRate(Z, K_M1M4_AUGER, NULL)+
		AugerRate(Z, K_M2M4_AUGER, NULL)+
		AugerRate(Z, K_M3M4_AUGER, NULL)+
		AugerRate(Z, K_M4L1_AUGER, NULL)+
		AugerRate(Z, K_M4L2_AUGER, NULL)+
		AugerRate(Z, K_M4L3_AUGER, NULL)+
		AugerRate(Z, K_M4M1_AUGER, NULL)+
		AugerRate(Z, K_M4M2_AUGER, NULL)+
		AugerRate(Z, K_M4M3_AUGER, NULL)+
		2 * AugerRate(Z, K_M4M4_AUGER, NULL)+
		AugerRate(Z, K_M4M5_AUGER, NULL)+
		AugerRate(Z, K_M4N1_AUGER, NULL)+
		AugerRate(Z, K_M4N2_AUGER, NULL)+
		AugerRate(Z, K_M4N3_AUGER, NULL)+
		AugerRate(Z, K_M4N4_AUGER, NULL)+
		AugerRate(Z, K_M4N5_AUGER, NULL)+
		AugerRate(Z, K_M4N6_AUGER, NULL)+
		AugerRate(Z, K_M4N7_AUGER, NULL)+
		AugerRate(Z, K_M4O1_AUGER, NULL)+
		AugerRate(Z, K_M4O2_AUGER, NULL)+
		AugerRate(Z, K_M4O3_AUGER, NULL)+
		AugerRate(Z, K_M4O4_AUGER, NULL)+
		AugerRate(Z, K_M4O5_AUGER, NULL)+
		AugerRate(Z, K_M4O6_AUGER, NULL)+
		AugerRate(Z, K_M4O7_AUGER, NULL)+
		AugerRate(Z, K_M4P1_AUGER, NULL)+
		AugerRate(Z, K_M4P2_AUGER, NULL)+
		AugerRate(Z, K_M4P3_AUGER, NULL)+
		AugerRate(Z, K_M4P4_AUGER, NULL)+
		AugerRate(Z, K_M4P5_AUGER, NULL)+
		AugerRate(Z, K_M4Q1_AUGER, NULL)+
		AugerRate(Z, K_M4Q2_AUGER, NULL)+
		AugerRate(Z, K_M4Q3_AUGER, NULL)+
		AugerRate(Z, K_M5M4_AUGER, NULL)
		));
	}
	else if (shell == L1_SHELL) {
		return (FluorYield(Z, L1_SHELL, NULL) * RadRate(Z, L1M4_LINE, NULL) +
		AugerYield(Z, L1_SHELL, NULL) * (
		AugerRate(Z, L1_M1M4_AUGER, NULL)+
		AugerRate(Z, L1_M2M4_AUGER, NULL)+
		AugerRate(Z, L1_M3M4_AUGER, NULL)+
		AugerRate(Z, L1_M4M1_AUGER, NULL)+
		AugerRate(Z, L1_M4M2_AUGER, NULL)+
		AugerRate(Z, L1_M4M3_AUGER, NULL)+
		2 * AugerRate(Z, L1_M4M4_AUGER, NULL)+
		AugerRate(Z, L1_M4M5_AUGER, NULL)+
		AugerRate(Z, L1_M4N1_AUGER, NULL)+
		AugerRate(Z, L1_M4N2_AUGER, NULL)+
		AugerRate(Z, L1_M4N3_AUGER, NULL)+
		AugerRate(Z, L1_M4N4_AUGER, NULL)+
		AugerRate(Z, L1_M4N5_AUGER, NULL)+
		AugerRate(Z, L1_M4N6_AUGER, NULL)+
		AugerRate(Z, L1_M4N7_AUGER, NULL)+
		AugerRate(Z, L1_M4O1_AUGER, NULL)+
		AugerRate(Z, L1_M4O2_AUGER, NULL)+
		AugerRate(Z, L1_M4O3_AUGER, NULL)+
		AugerRate(Z, L1_M4O4_AUGER, NULL)+
		AugerRate(Z, L1_M4O5_AUGER, NULL)+
		AugerRate(Z, L1_M4O6_AUGER, NULL)+
		AugerRate(Z, L1_M4O7_AUGER, NULL)+
		AugerRate(Z, L1_M4P1_AUGER, NULL)+
		AugerRate(Z, L1_M4P2_AUGER, NULL)+
		AugerRate(Z, L1_M4P3_AUGER, NULL)+
		AugerRate(Z, L1_M4P4_AUGER, NULL)+
		AugerRate(Z, L1_M4P5_AUGER, NULL)+
		AugerRate(Z, L1_M4Q1_AUGER, NULL)+
		AugerRate(Z, L1_M4Q2_AUGER, NULL)+
		AugerRate(Z, L1_M4Q3_AUGER, NULL)+
		AugerRate(Z, L1_M5M4_AUGER, NULL)
		));
	}
	else if (shell == L2_SHELL) {
		return (FluorYield(Z, L2_SHELL, NULL) * RadRate(Z, L2M4_LINE, NULL) +
		AugerYield(Z, L2_SHELL, NULL) * (
		AugerRate(Z, L2_M1M4_AUGER, NULL)+
		AugerRate(Z, L2_M2M4_AUGER, NULL)+
		AugerRate(Z, L2_M3M4_AUGER, NULL)+
		AugerRate(Z, L2_M4M1_AUGER, NULL)+
		AugerRate(Z, L2_M4M2_AUGER, NULL)+
		AugerRate(Z, L2_M4M3_AUGER, NULL)+
		2 * AugerRate(Z, L2_M4M4_AUGER, NULL)+
		AugerRate(Z, L2_M4M5_AUGER, NULL)+
		AugerRate(Z, L2_M4N1_AUGER, NULL)+
		AugerRate(Z, L2_M4N2_AUGER, NULL)+
		AugerRate(Z, L2_M4N3_AUGER, NULL)+
		AugerRate(Z, L2_M4N4_AUGER, NULL)+
		AugerRate(Z, L2_M4N5_AUGER, NULL)+
		AugerRate(Z, L2_M4N6_AUGER, NULL)+
		AugerRate(Z, L2_M4N7_AUGER, NULL)+
		AugerRate(Z, L2_M4O1_AUGER, NULL)+
		AugerRate(Z, L2_M4O2_AUGER, NULL)+
		AugerRate(Z, L2_M4O3_AUGER, NULL)+
		AugerRate(Z, L2_M4O4_AUGER, NULL)+
		AugerRate(Z, L2_M4O5_AUGER, NULL)+
		AugerRate(Z, L2_M4O6_AUGER, NULL)+
		AugerRate(Z, L2_M4O7_AUGER, NULL)+
		AugerRate(Z, L2_M4P1_AUGER, NULL)+
		AugerRate(Z, L2_M4P2_AUGER, NULL)+
		AugerRate(Z, L2_M4P3_AUGER, NULL)+
		AugerRate(Z, L2_M4P4_AUGER, NULL)+
		AugerRate(Z, L2_M4P5_AUGER, NULL)+
		AugerRate(Z, L2_M4Q1_AUGER, NULL)+
		AugerRate(Z, L2_M4Q2_AUGER, NULL)+
		AugerRate(Z, L2_M4Q3_AUGER, NULL)+
		AugerRate(Z, L2_M5M4_AUGER, NULL)
		));
	}
	else if (shell == L3_SHELL) {
		return (FluorYield(Z, L3_SHELL, NULL) * RadRate(Z, L3M4_LINE, NULL) +
		AugerYield(Z, L3_SHELL, NULL) * (
		AugerRate(Z, L3_M1M4_AUGER, NULL)+
		AugerRate(Z, L3_M2M4_AUGER, NULL)+
		AugerRate(Z, L3_M3M4_AUGER, NULL)+
		AugerRate(Z, L3_M4M1_AUGER, NULL)+
		AugerRate(Z, L3_M4M2_AUGER, NULL)+
		AugerRate(Z, L3_M4M3_AUGER, NULL)+
		2 * AugerRate(Z, L3_M4M4_AUGER, NULL)+
		AugerRate(Z, L3_M4M5_AUGER, NULL)+
		AugerRate(Z, L3_M4N1_AUGER, NULL)+
		AugerRate(Z, L3_M4N2_AUGER, NULL)+
		AugerRate(Z, L3_M4N3_AUGER, NULL)+
		AugerRate(Z, L3_M4N4_AUGER, NULL)+
		AugerRate(Z, L3_M4N5_AUGER, NULL)+
		AugerRate(Z, L3_M4N6_AUGER, NULL)+
		AugerRate(Z, L3_M4N7_AUGER, NULL)+
		AugerRate(Z, L3_M4O1_AUGER, NULL)+
		AugerRate(Z, L3_M4O2_AUGER, NULL)+
		AugerRate(Z, L3_M4O3_AUGER, NULL)+
		AugerRate(Z, L3_M4O4_AUGER, NULL)+
		AugerRate(Z, L3_M4O5_AUGER, NULL)+
		AugerRate(Z, L3_M4O6_AUGER, NULL)+
		AugerRate(Z, L3_M4O7_AUGER, NULL)+
		AugerRate(Z, L3_M4P1_AUGER, NULL)+
		AugerRate(Z, L3_M4P2_AUGER, NULL)+
		AugerRate(Z, L3_M4P3_AUGER, NULL)+
		AugerRate(Z, L3_M4P4_AUGER, NULL)+
		AugerRate(Z, L3_M4P5_AUGER, NULL)+
		AugerRate(Z, L3_M4Q1_AUGER, NULL)+
		AugerRate(Z, L3_M4Q2_AUGER, NULL)+
		AugerRate(Z, L3_M4Q3_AUGER, NULL)+
		AugerRate(Z, L3_M5M4_AUGER, NULL)
		));
	}
	return 0.0;
}

double PM5_get_cross_sections_constant_auger_only(int Z, int shell) {
	if (shell == K_SHELL) {
		return AugerYield(Z, K_SHELL, NULL) * (
		AugerRate(Z, K_L1M5_AUGER, NULL)+
		AugerRate(Z, K_L2M5_AUGER, NULL)+
		AugerRate(Z, K_L3M5_AUGER, NULL)+
		AugerRate(Z, K_M1M5_AUGER, NULL)+
		AugerRate(Z, K_M2M5_AUGER, NULL)+
		AugerRate(Z, K_M3M5_AUGER, NULL)+
		AugerRate(Z, K_M4M5_AUGER, NULL)+
		AugerRate(Z, K_M5L1_AUGER, NULL)+
		AugerRate(Z, K_M5L2_AUGER, NULL)+
		AugerRate(Z, K_M5L3_AUGER, NULL)+
		AugerRate(Z, K_M5M1_AUGER, NULL)+
		AugerRate(Z, K_M5M2_AUGER, NULL)+
		AugerRate(Z, K_M5M3_AUGER, NULL)+
		AugerRate(Z, K_M5M4_AUGER, NULL)+
		2 * AugerRate(Z, K_M5M5_AUGER, NULL)+
		AugerRate(Z, K_M5N1_AUGER, NULL)+
		AugerRate(Z, K_M5N2_AUGER, NULL)+
		AugerRate(Z, K_M5N3_AUGER, NULL)+
		AugerRate(Z, K_M5N4_AUGER, NULL)+
		AugerRate(Z, K_M5N5_AUGER, NULL)+
		AugerRate(Z, K_M5N6_AUGER, NULL)+
		AugerRate(Z, K_M5N7_AUGER, NULL)+
		AugerRate(Z, K_M5O1_AUGER, NULL)+
		AugerRate(Z, K_M5O2_AUGER, NULL)+
		AugerRate(Z, K_M5O3_AUGER, NULL)+
		AugerRate(Z, K_M5O4_AUGER, NULL)+
		AugerRate(Z, K_M5O5_AUGER, NULL)+
		AugerRate(Z, K_M5O6_AUGER, NULL)+
		AugerRate(Z, K_M5O7_AUGER, NULL)+
		AugerRate(Z, K_M5P1_AUGER, NULL)+
		AugerRate(Z, K_M5P2_AUGER, NULL)+
		AugerRate(Z, K_M5P3_AUGER, NULL)+
		AugerRate(Z, K_M5P4_AUGER, NULL)+
		AugerRate(Z, K_M5P5_AUGER, NULL)+
		AugerRate(Z, K_M5Q1_AUGER, NULL)+
		AugerRate(Z, K_M5Q2_AUGER, NULL)+
		AugerRate(Z, K_M5Q3_AUGER, NULL)
		);
	}
	else if (shell == L1_SHELL) {
		return AugerYield(Z, L1_SHELL, NULL) * (
		AugerRate(Z, L1_M1M5_AUGER, NULL)+
		AugerRate(Z, L1_M2M5_AUGER, NULL)+
		AugerRate(Z, L1_M3M5_AUGER, NULL)+
		AugerRate(Z, L1_M4M5_AUGER, NULL)+
		AugerRate(Z, L1_M5M1_AUGER, NULL)+
		AugerRate(Z, L1_M5M2_AUGER, NULL)+
		AugerRate(Z, L1_M5M3_AUGER, NULL)+
		AugerRate(Z, L1_M5M4_AUGER, NULL)+
		2 * AugerRate(Z, L1_M5M5_AUGER, NULL)+
		AugerRate(Z, L1_M5N1_AUGER, NULL)+
		AugerRate(Z, L1_M5N2_AUGER, NULL)+
		AugerRate(Z, L1_M5N3_AUGER, NULL)+
		AugerRate(Z, L1_M5N4_AUGER, NULL)+
		AugerRate(Z, L1_M5N5_AUGER, NULL)+
		AugerRate(Z, L1_M5N6_AUGER, NULL)+
		AugerRate(Z, L1_M5N7_AUGER, NULL)+
		AugerRate(Z, L1_M5O1_AUGER, NULL)+
		AugerRate(Z, L1_M5O2_AUGER, NULL)+
		AugerRate(Z, L1_M5O3_AUGER, NULL)+
		AugerRate(Z, L1_M5O4_AUGER, NULL)+
		AugerRate(Z, L1_M5O5_AUGER, NULL)+
		AugerRate(Z, L1_M5O6_AUGER, NULL)+
		AugerRate(Z, L1_M5O7_AUGER, NULL)+
		AugerRate(Z, L1_M5P1_AUGER, NULL)+
		AugerRate(Z, L1_M5P2_AUGER, NULL)+
		AugerRate(Z, L1_M5P3_AUGER, NULL)+
		AugerRate(Z, L1_M5P4_AUGER, NULL)+
		AugerRate(Z, L1_M5P5_AUGER, NULL)+
		AugerRate(Z, L1_M5Q1_AUGER, NULL)+
		AugerRate(Z, L1_M5Q2_AUGER, NULL)+
		AugerRate(Z, L1_M5Q3_AUGER, NULL)
		);
	}
	else if (shell == L2_SHELL) {
		return AugerYield(Z, L2_SHELL, NULL) * (
		AugerRate(Z, L2_M1M5_AUGER, NULL)+
		AugerRate(Z, L2_M2M5_AUGER, NULL)+
		AugerRate(Z, L2_M3M5_AUGER, NULL)+
		AugerRate(Z, L2_M4M5_AUGER, NULL)+
		AugerRate(Z, L2_M5M1_AUGER, NULL)+
		AugerRate(Z, L2_M5M2_AUGER, NULL)+
		AugerRate(Z, L2_M5M3_AUGER, NULL)+
		AugerRate(Z, L2_M5M4_AUGER, NULL)+
		2 * AugerRate(Z, L2_M5M5_AUGER, NULL)+
		AugerRate(Z, L2_M5N1_AUGER, NULL)+
		AugerRate(Z, L2_M5N2_AUGER, NULL)+
		AugerRate(Z, L2_M5N3_AUGER, NULL)+
		AugerRate(Z, L2_M5N4_AUGER, NULL)+
		AugerRate(Z, L2_M5N5_AUGER, NULL)+
		AugerRate(Z, L2_M5N6_AUGER, NULL)+
		AugerRate(Z, L2_M5N7_AUGER, NULL)+
		AugerRate(Z, L2_M5O1_AUGER, NULL)+
		AugerRate(Z, L2_M5O2_AUGER, NULL)+
		AugerRate(Z, L2_M5O3_AUGER, NULL)+
		AugerRate(Z, L2_M5O4_AUGER, NULL)+
		AugerRate(Z, L2_M5O5_AUGER, NULL)+
		AugerRate(Z, L2_M5O6_AUGER, NULL)+
		AugerRate(Z, L2_M5O7_AUGER, NULL)+
		AugerRate(Z, L2_M5P1_AUGER, NULL)+
		AugerRate(Z, L2_M5P2_AUGER, NULL)+
		AugerRate(Z, L2_M5P3_AUGER, NULL)+
		AugerRate(Z, L2_M5P4_AUGER, NULL)+
		AugerRate(Z, L2_M5P5_AUGER, NULL)+
		AugerRate(Z, L2_M5Q1_AUGER, NULL)+
		AugerRate(Z, L2_M5Q2_AUGER, NULL)+
		AugerRate(Z, L2_M5Q3_AUGER, NULL)
		);
	}
	else if (shell == L3_SHELL) {
		return AugerYield(Z, L3_SHELL, NULL) * (
		AugerRate(Z, L3_M1M5_AUGER, NULL)+
		AugerRate(Z, L3_M2M5_AUGER, NULL)+
		AugerRate(Z, L3_M3M5_AUGER, NULL)+
		AugerRate(Z, L3_M4M5_AUGER, NULL)+
		AugerRate(Z, L3_M5M1_AUGER, NULL)+
		AugerRate(Z, L3_M5M2_AUGER, NULL)+
		AugerRate(Z, L3_M5M3_AUGER, NULL)+
		AugerRate(Z, L3_M5M4_AUGER, NULL)+
		2 * AugerRate(Z, L3_M5M5_AUGER, NULL)+
		AugerRate(Z, L3_M5N1_AUGER, NULL)+
		AugerRate(Z, L3_M5N2_AUGER, NULL)+
		AugerRate(Z, L3_M5N3_AUGER, NULL)+
		AugerRate(Z, L3_M5N4_AUGER, NULL)+
		AugerRate(Z, L3_M5N5_AUGER, NULL)+
		AugerRate(Z, L3_M5N6_AUGER, NULL)+
		AugerRate(Z, L3_M5N7_AUGER, NULL)+
		AugerRate(Z, L3_M5O1_AUGER, NULL)+
		AugerRate(Z, L3_M5O2_AUGER, NULL)+
		AugerRate(Z, L3_M5O3_AUGER, NULL)+
		AugerRate(Z, L3_M5O4_AUGER, NULL)+
		AugerRate(Z, L3_M5O5_AUGER, NULL)+
		AugerRate(Z, L3_M5O6_AUGER, NULL)+
		AugerRate(Z, L3_M5O7_AUGER, NULL)+
		AugerRate(Z, L3_M5P1_AUGER, NULL)+
		AugerRate(Z, L3_M5P2_AUGER, NULL)+
		AugerRate(Z, L3_M5P3_AUGER, NULL)+
		AugerRate(Z, L3_M5P4_AUGER, NULL)+
		AugerRate(Z, L3_M5P5_AUGER, NULL)+
		AugerRate(Z, L3_M5Q1_AUGER, NULL)+
		AugerRate(Z, L3_M5Q2_AUGER, NULL)+
		AugerRate(Z, L3_M5Q3_AUGER, NULL)
		);
	}
	return 0.0;
}

double PM5_get_cross_sections_constant_full(int Z, int shell) {
	if (shell == K_SHELL) {
		return (FluorYield(Z, K_SHELL, NULL) * RadRate(Z, KM5_LINE, NULL) +
		AugerYield(Z, K_SHELL, NULL) * (
		AugerRate(Z, K_L1M5_AUGER, NULL)+
		AugerRate(Z, K_L2M5_AUGER, NULL)+
		AugerRate(Z, K_L3M5_AUGER, NULL)+
		AugerRate(Z, K_M1M5_AUGER, NULL)+
		AugerRate(Z, K_M2M5_AUGER, NULL)+
		AugerRate(Z, K_M3M5_AUGER, NULL)+
		AugerRate(Z, K_M4M5_AUGER, NULL)+
		AugerRate(Z, K_M5L1_AUGER, NULL)+
		AugerRate(Z, K_M5L2_AUGER, NULL)+
		AugerRate(Z, K_M5L3_AUGER, NULL)+
		AugerRate(Z, K_M5M1_AUGER, NULL)+
		AugerRate(Z, K_M5M2_AUGER, NULL)+
		AugerRate(Z, K_M5M3_AUGER, NULL)+
		AugerRate(Z, K_M5M4_AUGER, NULL)+
		2 * AugerRate(Z, K_M5M5_AUGER, NULL)+
		AugerRate(Z, K_M5N1_AUGER, NULL)+
		AugerRate(Z, K_M5N2_AUGER, NULL)+
		AugerRate(Z, K_M5N3_AUGER, NULL)+
		AugerRate(Z, K_M5N4_AUGER, NULL)+
		AugerRate(Z, K_M5N5_AUGER, NULL)+
		AugerRate(Z, K_M5N6_AUGER, NULL)+
		AugerRate(Z, K_M5N7_AUGER, NULL)+
		AugerRate(Z, K_M5O1_AUGER, NULL)+
		AugerRate(Z, K_M5O2_AUGER, NULL)+
		AugerRate(Z, K_M5O3_AUGER, NULL)+
		AugerRate(Z, K_M5O4_AUGER, NULL)+
		AugerRate(Z, K_M5O5_AUGER, NULL)+
		AugerRate(Z, K_M5O6_AUGER, NULL)+
		AugerRate(Z, K_M5O7_AUGER, NULL)+
		AugerRate(Z, K_M5P1_AUGER, NULL)+
		AugerRate(Z, K_M5P2_AUGER, NULL)+
		AugerRate(Z, K_M5P3_AUGER, NULL)+
		AugerRate(Z, K_M5P4_AUGER, NULL)+
		AugerRate(Z, K_M5P5_AUGER, NULL)+
		AugerRate(Z, K_M5Q1_AUGER, NULL)+
		AugerRate(Z, K_M5Q2_AUGER, NULL)+
		AugerRate(Z, K_M5Q3_AUGER, NULL)
		));
	}
	else if (shell == L1_SHELL) {
		return (FluorYield(Z, L1_SHELL, NULL) * RadRate(Z, L1M5_LINE, NULL) +
		AugerYield(Z, L1_SHELL, NULL) * (
		AugerRate(Z, L1_M1M5_AUGER, NULL)+
		AugerRate(Z, L1_M2M5_AUGER, NULL)+
		AugerRate(Z, L1_M3M5_AUGER, NULL)+
		AugerRate(Z, L1_M4M5_AUGER, NULL)+
		AugerRate(Z, L1_M5M1_AUGER, NULL)+
		AugerRate(Z, L1_M5M2_AUGER, NULL)+
		AugerRate(Z, L1_M5M3_AUGER, NULL)+
		AugerRate(Z, L1_M5M4_AUGER, NULL)+
		2 * AugerRate(Z, L1_M5M5_AUGER, NULL)+
		AugerRate(Z, L1_M5N1_AUGER, NULL)+
		AugerRate(Z, L1_M5N2_AUGER, NULL)+
		AugerRate(Z, L1_M5N3_AUGER, NULL)+
		AugerRate(Z, L1_M5N4_AUGER, NULL)+
		AugerRate(Z, L1_M5N5_AUGER, NULL)+
		AugerRate(Z, L1_M5N6_AUGER, NULL)+
		AugerRate(Z, L1_M5N7_AUGER, NULL)+
		AugerRate(Z, L1_M5O1_AUGER, NULL)+
		AugerRate(Z, L1_M5O2_AUGER, NULL)+
		AugerRate(Z, L1_M5O3_AUGER, NULL)+
		AugerRate(Z, L1_M5O4_AUGER, NULL)+
		AugerRate(Z, L1_M5O5_AUGER, NULL)+
		AugerRate(Z, L1_M5O6_AUGER, NULL)+
		AugerRate(Z, L1_M5O7_AUGER, NULL)+
		AugerRate(Z, L1_M5P1_AUGER, NULL)+
		AugerRate(Z, L1_M5P2_AUGER, NULL)+
		AugerRate(Z, L1_M5P3_AUGER, NULL)+
		AugerRate(Z, L1_M5P4_AUGER, NULL)+
		AugerRate(Z, L1_M5P5_AUGER, NULL)+
		AugerRate(Z, L1_M5Q1_AUGER, NULL)+
		AugerRate(Z, L1_M5Q2_AUGER, NULL)+
		AugerRate(Z, L1_M5Q3_AUGER, NULL)
		));
	}
	else if (shell == L2_SHELL) {
		return (FluorYield(Z, L2_SHELL, NULL) * RadRate(Z, L2M5_LINE, NULL) +
		AugerYield(Z, L2_SHELL, NULL) * (
		AugerRate(Z, L2_M1M5_AUGER, NULL)+
		AugerRate(Z, L2_M2M5_AUGER, NULL)+
		AugerRate(Z, L2_M3M5_AUGER, NULL)+
		AugerRate(Z, L2_M4M5_AUGER, NULL)+
		AugerRate(Z, L2_M5M1_AUGER, NULL)+
		AugerRate(Z, L2_M5M2_AUGER, NULL)+
		AugerRate(Z, L2_M5M3_AUGER, NULL)+
		AugerRate(Z, L2_M5M4_AUGER, NULL)+
		2 * AugerRate(Z, L2_M5M5_AUGER, NULL)+
		AugerRate(Z, L2_M5N1_AUGER, NULL)+
		AugerRate(Z, L2_M5N2_AUGER, NULL)+
		AugerRate(Z, L2_M5N3_AUGER, NULL)+
		AugerRate(Z, L2_M5N4_AUGER, NULL)+
		AugerRate(Z, L2_M5N5_AUGER, NULL)+
		AugerRate(Z, L2_M5N6_AUGER, NULL)+
		AugerRate(Z, L2_M5N7_AUGER, NULL)+
		AugerRate(Z, L2_M5O1_AUGER, NULL)+
		AugerRate(Z, L2_M5O2_AUGER, NULL)+
		AugerRate(Z, L2_M5O3_AUGER, NULL)+
		AugerRate(Z, L2_M5O4_AUGER, NULL)+
		AugerRate(Z, L2_M5O5_AUGER, NULL)+
		AugerRate(Z, L2_M5O6_AUGER, NULL)+
		AugerRate(Z, L2_M5O7_AUGER, NULL)+
		AugerRate(Z, L2_M5P1_AUGER, NULL)+
		AugerRate(Z, L2_M5P2_AUGER, NULL)+
		AugerRate(Z, L2_M5P3_AUGER, NULL)+
		AugerRate(Z, L2_M5P4_AUGER, NULL)+
		AugerRate(Z, L2_M5P5_AUGER, NULL)+
		AugerRate(Z, L2_M5Q1_AUGER, NULL)+
		AugerRate(Z, L2_M5Q2_AUGER, NULL)+
		AugerRate(Z, L2_M5Q3_AUGER, NULL)
		));
	}
	else if (shell == L3_SHELL) {
		return (FluorYield(Z, L3_SHELL, NULL) * RadRate(Z, L3M5_LINE, NULL) +
		AugerYield(Z, L3_SHELL, NULL) * (
		AugerRate(Z, L3_M1M5_AUGER, NULL)+
		AugerRate(Z, L3_M2M5_AUGER, NULL)+
		AugerRate(Z, L3_M3M5_AUGER, NULL)+
		AugerRate(Z, L3_M4M5_AUGER, NULL)+
		AugerRate(Z, L3_M5M1_AUGER, NULL)+
		AugerRate(Z, L3_M5M2_AUGER, NULL)+
		AugerRate(Z, L3_M5M3_AUGER, NULL)+
		AugerRate(Z, L3_M5M4_AUGER, NULL)+
		2 * AugerRate(Z, L3_M5M5_AUGER, NULL)+
		AugerRate(Z, L3_M5N1_AUGER, NULL)+
		AugerRate(Z, L3_M5N2_AUGER, NULL)+
		AugerRate(Z, L3_M5N3_AUGER, NULL)+
		AugerRate(Z, L3_M5N4_AUGER, NULL)+
		AugerRate(Z, L3_M5N5_AUGER, NULL)+
		AugerRate(Z, L3_M5N6_AUGER, NULL)+
		AugerRate(Z, L3_M5N7_AUGER, NULL)+
		AugerRate(Z, L3_M5O1_AUGER, NULL)+
		AugerRate(Z, L3_M5O2_AUGER, NULL)+
		AugerRate(Z, L3_M5O3_AUGER, NULL)+
		AugerRate(Z, L3_M5O4_AUGER, NULL)+
		AugerRate(Z, L3_M5O5_AUGER, NULL)+
		AugerRate(Z, L3_M5O6_AUGER, NULL)+
		AugerRate(Z, L3_M5O7_AUGER, NULL)+
		AugerRate(Z, L3_M5P1_AUGER, NULL)+
		AugerRate(Z, L3_M5P2_AUGER, NULL)+
		AugerRate(Z, L3_M5P3_AUGER, NULL)+
		AugerRate(Z, L3_M5P4_AUGER, NULL)+
		AugerRate(Z, L3_M5P5_AUGER, NULL)+
		AugerRate(Z, L3_M5Q1_AUGER, NULL)+
		AugerRate(Z, L3_M5Q2_AUGER, NULL)+
		AugerRate(Z, L3_M5Q3_AUGER, NULL)
		));
	}
	return 0.0;
}
