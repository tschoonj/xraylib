#include "xraylib.h"

float PL1_pure_kissel(int Z, float E);
float PL1_rad_cascade_kissel(int Z, float E, float PK);
float PL1_auger_cascade_kissel(int Z, float E, float PK);
float PL1_full_cascade_kissel(int Z, float E, float PK);

float PL2_pure_kissel(int Z, float E, float PL1);
float PL2_rad_cascade_kissel(int Z, float E, float PK, float PL1);
float PL2_auger_cascade_kissel(int Z, float E, float PK, float PL1);
float PL2_full_cascade_kissel(int Z, float E, float PK, float PL1);

float PL3_pure_kissel(int Z, float E, float PL1, float PL2);
float PL3_rad_cascade_kissel(int Z, float E, float PK, float PL1, float PL2);
float PL3_auger_cascade_kissel(int Z, float E, float PK, float PL1, float PL2);
float PL3_full_cascade_kissel(int Z, float E, float PK, float PL1, float PL2);

float PM1_pure_kissel(int Z, float E);
float PM1_rad_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3);
float PM1_auger_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3);
float PM1_full_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3);

float PM2_pure_kissel(int Z, float E, float PM1);
float PM2_rad_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1);
float PM2_auger_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1);
float PM2_full_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1);

float PM3_pure_kissel(int Z, float E, float PM1, float PM2);
float PM3_rad_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2);
float PM3_auger_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2);
float PM3_full_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2);

float PM4_pure_kissel(int Z, float E, float PM1, float PM2, float PM3);
float PM4_rad_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2, float PM3);
float PM4_auger_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2, float PM3);
float PM4_full_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2, float PM3);


float PM5_pure_kissel(int Z, float E, float PM1, float PM2, float PM3, float PM4);
float PM5_rad_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2, float PM3, float PM4);
float PM5_auger_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2, float PM3, float PM4);
float PM5_full_cascade_kissel(int Z, float E, float PK, float PL1, float PL2, float PL3, float PM1, float PM2, float PM3, float PM4);
