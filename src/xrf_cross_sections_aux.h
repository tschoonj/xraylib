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

#include "xraylib.h"

double PL1_pure_kissel(int Z, double E);
double PL1_rad_cascade_kissel(int Z, double E, double PK);
double PL1_auger_cascade_kissel(int Z, double E, double PK);
double PL1_full_cascade_kissel(int Z, double E, double PK);

double PL2_pure_kissel(int Z, double E, double PL1);
double PL2_rad_cascade_kissel(int Z, double E, double PK, double PL1);
double PL2_auger_cascade_kissel(int Z, double E, double PK, double PL1);
double PL2_full_cascade_kissel(int Z, double E, double PK, double PL1);

double PL3_pure_kissel(int Z, double E, double PL1, double PL2);
double PL3_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2);
double PL3_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2);
double PL3_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2);

double PM1_pure_kissel(int Z, double E);
double PM1_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3);
double PM1_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3);
double PM1_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3);

double PM2_pure_kissel(int Z, double E, double PM1);
double PM2_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1);
double PM2_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1);
double PM2_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1);

double PM3_pure_kissel(int Z, double E, double PM1, double PM2);
double PM3_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2);
double PM3_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2);
double PM3_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2);

double PM4_pure_kissel(int Z, double E, double PM1, double PM2, double PM3);
double PM4_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3);
double PM4_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3);
double PM4_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3);


double PM5_pure_kissel(int Z, double E, double PM1, double PM2, double PM3, double PM4);
double PM5_rad_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4);
double PM5_auger_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4);
double PM5_full_cascade_kissel(int Z, double E, double PK, double PL1, double PL2, double PL3, double PM1, double PM2, double PM3, double PM4);
