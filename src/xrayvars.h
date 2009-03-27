#define ZMAX 120
#define MAXFILENAMESIZE 1000
#define SHELLNUM 28
#define SHELLNUM_K SHELLNUM+3
#define LINENUM 50
#define TRANSNUM 5

#ifndef VARSH
#define VARSH

//////////////////////////////////////////////////////////////////////
/////            Functions                                       /////
//////////////////////////////////////////////////////////////////////
void XRayInit(void);
void ErrorExit(char *error_message);


//////////////////////////////////////////////////////////////////////
/////            Variables                                       /////
//////////////////////////////////////////////////////////////////////

extern int HardExit;
extern int ExitStatus;
extern char XRayLibDir[];

extern char ShellName[][5];
extern char LineName[][5];
extern char TransName[][5];

#endif
