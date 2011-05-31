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
