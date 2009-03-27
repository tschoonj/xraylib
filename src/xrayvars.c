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

char LineName[][5] = {
"KL1",	"KL2",	"KL3",	"KM1",	"KM2",	"KM3",	"KM4",	"KM5",	"KN1",	"KN2",
"KN3",	"KN4",	"KN5",	"L1M1",	"L1M2",	"L1M3",	"L1M4",	"L1M5",	"L1N1",	"L1N2",
"L1N3",	"L1N4",	"L1N5",	"L1N6",	"L1N7",	"L2M1",	"L2M2",	"L2M3",	"L2M4",	"L2M5",
"L2N1",	"L2N2",	"L2N3",	"L2N4",	"L2N5",	"L2N6",	"L2N7",	"L3M1",	"L3M2",	"L3M3",
"L3M4",	"L3M5",	"L3N1",	"L3N2",	"L3N3",	"L3N4",	"L3N5",	"L3N6",	"L3N7"
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
