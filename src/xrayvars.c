/*
Copyright (c) 2009-2010, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define VARSH
#include <stdio.h>
#include <stdlib.h>
#include "xrayvars.h"
#include "xrayglob.h"

//////////////////////////////////////////////////////////////////////
/////            Variables                                       /////
//////////////////////////////////////////////////////////////////////

int HardExit = 0;
int ExitStatus = 0;
char XRayLibDir[MAXFILENAMESIZE];

char ShellName[][5] = {
"K",    "L1",   "L2",   "L3",   "M1",   "M2",   "M3",   "M4",   "M5",   "N1",
"N2",   "N3",   "N4",   "N5",   "N6",   "N7",   "O1",   "O2",   "O3",   "O4",
"O5",   "O6",   "O7",   "P1",   "P2",   "P3",   "P4",   "P5"
};

//must match with contents of lines.h !!!! -> contains 383 linenames
char LineName[][6] = {
"KL1", "KL2", "KL3", "KM1", "KM2", "KM3", "KM4", "KM5", "KN1", "KN2",
"KN3", "KN4", "KN5", "KN6", "KN7", "KO", "KO1", "KO2", "KO3", "KO4",
"KO5", "KO6", "KO7", "KP", "KP1", "KP2", "KP3", "KP4", "KP5", "L1L2",
"L1L3", "L1M1", "L1M2", "L1M3", "L1M4", "L1M5", "L1N1", "L1N2", "L1N3", "L1N4",
"L1N5", "L1N6", "L1N67", "L1N7", "L1O1", "L1O2", "L1O3", "L1O4", "L1O45", "L1O5",
"L1O6", "L1O7", "L1P1", "L1P2", "L1P23", "L1P3", "L1P4", "L1P5", "L2L3", "L2M1",
"L2M2", "L2M3", "L2M4", "L2M5", "L2N1", "L2N2", "L2N3", "L2N4", "L2N5", "L2N6",
"L2N7", "L2O1", "L2O2", "L2O3", "L2O4", "L2O5", "L2O6", "L2O7", "L2P1", "L2P2",
"L2P23", "L2P3", "L2P4", "L2P5", "L2Q1", "L3M1", "L3M2", "L3M3", "L3M4", "L3M5",
"L3N1", "L3N2", "L3N3", "L3N4", "L3N5", "L3N6", "L3N7", "L3O1", "L3O2", "L3O3",
"L3O4", "L3O45", "L3O5", "L3O6", "L3O7", "L3P1", "L3P2", "L3P23", "L3P3", "L3P4",
"L3P45", "L3P5", "L3Q1", "M1M2", "M1M3", "M1M4", "M1M5", "M1N1", "M1N2", "M1N3",
"M1N4", "M1N5", "M1N6", "M1N7", "M1O1", "M1O2", "M1O3", "M1O4", "M1O5", "M1O6",
"M1O7", "M1P1", "M1P2", "M1P3", "M1P4", "M1P5", "M2M3", "M2M4", "M2M5", "M2N1",
"M2N2", "M2N3", "M2N4", "M2N5", "M2N6", "M2N7", "M2O1", "M2O2", "M2O3", "M2O4",
"M2O5", "M2O6", "M2O7", "M2P1", "M2P2", "M2P3", "M2P4", "M2P5", "M3M4", "M3M5",
"M3N1", "M3N2", "M3N3", "M3N4", "M3N5", "M3N6", "M3N7", "M3O1", "M3O2", "M3O3",
"M3O4", "M3O5", "M3O6", "M3O7", "M3P1", "M3P2", "M3P3", "M3P4", "M3P5", "M3Q1",
"M4M5", "M4N1", "M4N2", "M4N3", "M4N4", "M4N5", "M4N6", "M4N7", "M4O1", "M4O2",
"M4O3", "M4O4", "M4O5", "M4O6", "M4O7", "M4P1", "M4P2", "M4P3", "M4P4", "M4P5",
"M5N1", "M5N2", "M5N3", "M5N4", "M5N5", "M5N6", "M5N7", "M5O1", "M5O2", "M5O3",
"M5O4", "M5O5", "M5O6", "M5O7", "M5P1", "M5P2", "M5P3", "M5P4", "M5P5", "N1N2",
"N1N3", "N1N4", "N1N5", "N1N6", "N1N7", "N1O1", "N1O2", "N1O3", "N1O4", "N1O5",
"N1O6", "N1O7", "N1P1", "N1P2", "N1P3", "N1P4", "N1P5", "N2N3", "N2N4", "N2N5",
"N2N6", "N2N7", "N2O1", "N2O2", "N2O3", "N2O4", "N2O5", "N2O6", "N2O7", "N2P1",
"N2P2", "N2P3", "N2P4", "N2P5", "N3N4", "N3N5", "N3N6", "N3N7", "N3O1", "N3O2",
"N3O3", "N3O4", "N3O5", "N3O6", "N3O7", "N3P1", "N3P2", "N3P3", "N3P4", "N3P5",
"N4N5", "N4N6", "N4N7", "N4O1", "N4O2", "N4O3", "N4O4", "N4O5", "N4O6", "N4O7",
"N4P1", "N4P2", "N4P3", "N4P4", "N4P5", "N5N6", "N5N7", "N5O1", "N5O2", "N5O3",
"N5O4", "N5O5", "N5O6", "N5O7", "N5P1", "N5P2", "N5P3", "N5P4", "N5P5", "N6N7",
"N6O1", "N6O2", "N6O3", "N6O4", "N6O5", "N6O6", "N6O7", "N6P1", "N6P2", "N6P3",
"N6P4", "N6P5", "N7O1", "N7O2", "N7O3", "N7O4", "N7O5", "N7O6", "N7O7", "N7P1",
"N7P2", "N7P3", "N7P4", "N7P5", "O1O2", "O1O3", "O1O4", "O1O5", "O1O6", "O1O7",
"O1P1", "O1P2", "O1P3", "O1P4", "O1P5", "O2O3", "O2O4", "O2O5", "O2O6", "O2O7",
"O2P1", "O2P2", "O2P3", "O2P4", "O2P5", "O3O4", "O3O5", "O3O6", "O3O7", "O3P1",
"O3P2", "O3P3", "O3P4", "O3P5", "O4O5", "O4O6", "O4O7", "O4P1", "O4P2", "O4P3",
"O4P4", "O4P5", "O5O6", "O5O7", "O5P1", "O5P2", "O5P3", "O5P4", "O5P5", "O6O7",
"O6P4", "O6P5", "O7P4", "O7P5", "P1P2", "P1P3", "P1P4", "P1P5", "P2P3", "P2P4", 
"P2P5", "P3P4", "P3P5"
};


char TransName[][5] = {"F1","F12","F13","FP13","F23"};

//////////////////////////////////////////////////////////////////////
/////            Functions                                       /////
//////////////////////////////////////////////////////////////////////

void ErrorExit(char *error_message)
{
  printf("%s\n", error_message);
  ExitStatus = 1;
  if (HardExit != 0) exit(EXIT_FAILURE);
}

void SetHardExit(int hard_exit)
{
  HardExit = hard_exit;
}

void SetExitStatus(int exit_status)
{
  ExitStatus = exit_status;
}

int GetExitStatus()
{
  return ExitStatus;
}
