#ifndef XRAYLIB_XRF_CROSS_SECTION_AUX_PRIVATE_H
#define XRAYLIB_XRF_CROSS_SECTION_AUX_PRIVATE_H

/*
 * The following methods are not visible outside the library!
 */
double PL1_get_cross_sections_constant_auger_only(int Z, int shell);
double PL1_get_cross_sections_constant_full(int Z, int shell);

double PL2_get_cross_sections_constant_auger_only(int Z, int shell);
double PL2_get_cross_sections_constant_full(int Z, int shell);

double PL3_get_cross_sections_constant_auger_only(int Z, int shell);
double PL3_get_cross_sections_constant_full(int Z, int shell);

double PM1_get_cross_sections_constant_auger_only(int Z, int shell);
double PM1_get_cross_sections_constant_full(int Z, int shell);

double PM2_get_cross_sections_constant_auger_only(int Z, int shell);
double PM2_get_cross_sections_constant_full(int Z, int shell);

double PM3_get_cross_sections_constant_auger_only(int Z, int shell);
double PM3_get_cross_sections_constant_full(int Z, int shell);

double PM4_get_cross_sections_constant_auger_only(int Z, int shell);
double PM4_get_cross_sections_constant_full(int Z, int shell);

double PM5_get_cross_sections_constant_auger_only(int Z, int shell);
double PM5_get_cross_sections_constant_full(int Z, int shell);

#endif
