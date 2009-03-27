#include "xrayvars.h"


/*
#define ZMAX 120
#define MAXFILENAMESIZE 1000
#define SHELLNUM 28
#define LINENUM 50
#define TRANSNUM 5

//////////////////////////////////////////////////////////////////////
/////            Functions                                       /////
//////////////////////////////////////////////////////////////////////
void XRayInit(void);
void ErrorExit(char *error_message);


//////////////////////////////////////////////////////////////////////
/////            Variables                                       /////
//////////////////////////////////////////////////////////////////////
*/
#ifndef GLOBH
#define GLOBH
/*
extern int HardExit;
extern int ExitStatus;
extern char XRayLibDir[];

extern char ShellName[][5];
extern char LineName[][5];
extern char TransName[][5];
*/
extern float AtomicWeight_arr[ZMAX+1];
extern float EdgeEnergy_arr[ZMAX+1][SHELLNUM];
extern float LineEnergy_arr[ZMAX+1][LINENUM];
extern float FluorYield_arr[ZMAX+1][SHELLNUM];
extern float JumpFactor_arr[ZMAX+1][SHELLNUM];
extern float CosKron_arr[ZMAX+1][TRANSNUM];
extern float RadRate_arr[ZMAX+1][LINENUM];

extern int NE_Photo[ZMAX+1];
extern float *E_Photo_arr[ZMAX+1];
extern float *CS_Photo_arr[ZMAX+1];
extern float *CS_Photo_arr2[ZMAX+1];

extern int NE_Rayl[ZMAX+1];
extern float *E_Rayl_arr[ZMAX+1];
extern float *CS_Rayl_arr[ZMAX+1];
extern float *CS_Rayl_arr2[ZMAX+1];

extern int NE_Compt[ZMAX+1];
extern float *E_Compt_arr[ZMAX+1];
extern float *CS_Compt_arr[ZMAX+1];
extern float *CS_Compt_arr2[ZMAX+1];

extern int Nq_Rayl[ZMAX+1];
extern float *q_Rayl_arr[ZMAX+1];
extern float *FF_Rayl_arr[ZMAX+1];
extern float *FF_Rayl_arr2[ZMAX+1];

extern int Nq_Compt[ZMAX+1];
extern float *q_Compt_arr[ZMAX+1];
extern float *SF_Compt_arr[ZMAX+1];
extern float *SF_Compt_arr2[ZMAX+1];

extern int NE_Fi[ZMAX+1];
extern float *E_Fi_arr[ZMAX+1];
extern float *Fi_arr[ZMAX+1];
extern float *Fi_arr2[ZMAX+1];

extern int NE_Fii[ZMAX+1];
extern float *E_Fii_arr[ZMAX+1];
extern float *Fii_arr[ZMAX+1];
extern float *Fii_arr2[ZMAX+1];

extern int NE_Photo_Total_Kissel[ZMAX+1];
extern double *E_Photo_Total_Kissel[ZMAX+1];
extern double *Photo_Total_Kissel[ZMAX+1];
extern double *Photo_Total_Kissel2[ZMAX+1];

extern float Electron_Config_Kissel[ZMAX+1][SHELLNUM_K];

extern int NE_Photo_Partial_Kissel[ZMAX+1][SHELLNUM_K];
extern double *E_Photo_Partial_Kissel[ZMAX+1][SHELLNUM_K];
extern double *Photo_Partial_Kissel[ZMAX+1][SHELLNUM_K];
extern double *Photo_Partial_Kissel2[ZMAX+1][SHELLNUM_K];




#endif


















