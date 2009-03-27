#include <stdio.h>
#include "export.h"
#include "xraylib.h"
#include <stdbool.h>


extern int HardExit, ExitStatus;


#ifdef __i386__
bool arch64=false;
#elif defined(__x86_64__)
bool arch64=true;
#endif

extern void IDL_CDECL IDL_XRayInit(int argc, IDL_VPTR argv[]);
extern void IDL_CDECL IDL_SetHardExit(int argc, IDL_VPTR argv[]);
extern void IDL_CDECL IDL_SetExitStatus(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_GetExitStatus(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_AtomicWeight(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CS_Total(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CS_Photo(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CS_Rayl(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CS_Compt(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CSb_Total(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CSb_Photo(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CSb_Rayl(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CSb_Compt(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CS_KN(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_DCS_Thoms(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_DCS_KN(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_DCS_Rayl(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_DCS_Compt(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_DCSb_Rayl(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_DCSb_Compt(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_DCSP_Thoms(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_DCSP_KN(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_DCSP_Rayl(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_DCSP_Compt(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_DCSPb_Rayl(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_DCSPb_Compt(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_FF_Rayl(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_SF_Compt(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_MomentTransf(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_LineEnergy(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_FluorYield(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CosKronTransProb(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_EdgeEnergy(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_JumpFactor(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CS_FluorLine(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CSb_FluorLine(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_RadRate(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_ComptonEnergy(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_Fi(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_Fii(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CSb_Photo_Total(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CS_Photo_Total(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CSb_Photo_Partial(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CS_Photo_Partial(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CS_FluorLine_Kissel(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CSb_FluorLine_Kissel(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CS_Total_Kissel(int argc, IDL_VPTR argv[]);
extern IDL_VPTR IDL_CDECL IDL_CSb_Total_Kissel(int argc, IDL_VPTR argv[]);


static IDL_SYSFUN_DEF2 xrl_functions[] = {
	{IDL_GetExitStatus,"GETEXITSTATUS", 0 , 0 , 0 , 0},
	{IDL_AtomicWeight,"ATOMICWEIGHT", 1 , 1 , 0 , 0},
	{IDL_CS_Total,"CS_TOTAL", 2 , 2 , 0 , 0},
	{IDL_CS_Photo,"CS_PHOTO", 2 , 2 , 0 , 0},
	{IDL_CS_Rayl,"CS_RAYL", 2 , 2 , 0 , 0},
	{IDL_CS_Compt,"CS_COMPT", 2 , 2 , 0 , 0},
	{IDL_CSb_Total,"CSB_TOTAL", 2 , 2 , 0 , 0},
	{IDL_CSb_Photo,"CSB_PHOTO", 2 , 2 , 0 , 0},
	{IDL_CSb_Rayl,"CSB_RAYL", 2 , 2 , 0 , 0},
	{IDL_CSb_Compt,"CSB_COMPT", 2 , 2 , 0 , 0},
	{IDL_CS_KN,"CS_KN", 1 , 1 , 0 , 0},
	{IDL_DCS_Thoms,"DCS_THOMS", 1 , 1 , 0 , 0},
	{IDL_DCS_KN,"DCS_KN", 2 , 2 , 0 , 0},
	{IDL_DCS_Rayl,"DCS_RAYL", 3 , 3 , 0 , 0},
	{IDL_DCS_Compt,"DCS_COMPT", 3 , 3 , 0 , 0},
	{IDL_DCSb_Rayl,"DCSB_RAYL", 3 , 3 , 0 , 0},
	{IDL_DCSb_Compt,"DCSB_COMPT", 3 , 3 , 0 , 0},
	{IDL_DCSP_Thoms,"DCSP_THOMS", 2 , 2 , 0 , 0},
	{IDL_DCSP_KN,"DCSP_KN", 3 , 3 , 0 , 0},
	{IDL_DCSP_Rayl,"DCSP_RAYL", 4 , 4 , 0 , 0},
	{IDL_DCSP_Compt,"DCSP_COMPT", 4 , 4 , 0 , 0},
	{IDL_DCSPb_Rayl,"DCSPB_RAYL", 4 , 4 , 0 , 0},
	{IDL_DCSPb_Compt,"DCSPB_COMPT", 4 , 4 , 0 , 0},
	{IDL_FF_Rayl,"FF_RAYL", 2 , 2 , 0 , 0},
	{IDL_SF_Compt,"SF_COMPT", 2 , 2 , 0 , 0},
	{IDL_MomentTransf,"MOMENTTRANSF", 2 , 2 , 0 , 0},
	{IDL_LineEnergy,"LINEENERGY", 2 , 2 , 0 , 0},
	{IDL_FluorYield,"FLUORYIELD", 2 , 2 , 0 , 0},
	{IDL_CosKronTransProb,"COSKRONTRANSPROB", 2 , 2 , 0 , 0},
	{IDL_EdgeEnergy,"EDGEENERGY", 2 , 2 , 0 , 0},
	{IDL_JumpFactor,"JUMPFACTOR", 2 , 2 , 0 , 0},
	{IDL_CS_FluorLine,"CS_FLUORLINE", 3 , 3 , 0 , 0},
	{IDL_CSb_FluorLine,"CSB_FLUORLINE", 3 , 3 , 0 , 0},
	{IDL_RadRate,"RADRATE", 2 , 2 , 0 , 0},
	{IDL_ComptonEnergy,"COMPTONENERGY", 2 , 2 , 0 , 0},
	{IDL_Fi,"FI", 2 , 2 , 0 , 0},
	{IDL_Fii,"FII", 2 , 2 , 0 , 0},
	{IDL_CSb_Photo_Total,"CSB_PHOTO_TOTAL", 2 , 2 , 0 , 0},
	{IDL_CS_Photo_Total,"CS_PHOTO_TOTAL", 2 , 2 , 0 , 0},
	{IDL_CSb_Photo_Partial,"CSB_PHOTO_PARTIAL", 3 , 3 , 0 , 0},
	{IDL_CS_Photo_Partial,"CS_PHOTO_PARTIAL", 3 , 3 , 0 , 0},
	{IDL_CS_FluorLine_Kissel,"CS_FLUORLINE_KISSEL", 3 , 3 , 0 , 0},
	{IDL_CSb_FluorLine_Kissel,"CSB_FLUORLINE_KISSEL", 3 , 3 , 0 , 0},
	{IDL_CS_Total_Kissel,"CS_TOTAL_KISSEL", 2 , 2 , 0 , 0},
	{IDL_CSb_Total_Kissel,"CSB_TOTAL_KISSEL", 2 , 2 , 0 , 0},
};
static IDL_SYSFUN_DEF2 xrl_procedures[] = {
	{(IDL_SYSRTN_GENERIC) IDL_XRayInit,"XRAYINIT", 0 , 0 , 0 , 0},
	{(IDL_SYSRTN_GENERIC) IDL_SetHardExit,"SETHARDEXIT", 1 , 1 , 0 , 0},
	{(IDL_SYSRTN_GENERIC) IDL_SetExitStatus,"SETEXITSTATUS", 1 , 1 , 0 , 0},
};


// Error Handling
void IDL_CDECL IDL_SetHardExit(int argc, IDL_VPTR argv[])
{
  HardExit = IDL_LongScalar(argv[0]);
}

void IDL_CDECL IDL_SetExitStatus(int argc, IDL_VPTR argv[])
{
  ExitStatus = IDL_LongScalar(argv[0]);
}

IDL_VPTR IDL_CDECL IDL_GetExitStatus(int argc, IDL_VPTR argv[])
{
  return IDL_GettmpInt((IDL_INT) ExitStatus);
}

void IDL_CDECL IDL_XRayInit(int argc, IDL_VPTR argv[])
{
  XRayInit();
}

//macros for the IDL functions

//1 argument: 1 int
#define XRL_1I(name) IDL_VPTR IDL_CDECL IDL_ ## name(int argc, IDL_VPTR argv[])\
{\
  int Z;\
  IDL_VPTR out_var;\
  \
  Z = IDL_LongScalar(argv[0]);\
  \
  out_var = IDL_Gettmp();\
  out_var->type = IDL_TYP_FLOAT;\
  out_var->value.f = name(Z);\
  \
  return out_var;\
}

//1 argument: 1 float
#define XRL_1F(name) IDL_VPTR IDL_CDECL IDL_ ## name(int argc, IDL_VPTR argv[])\
{\
  float E;\
  IDL_VPTR out_var;\
  \
  E = (float) IDL_DoubleScalar(argv[0]);\
  \
  out_var = IDL_Gettmp();\
  out_var->type = IDL_TYP_FLOAT;\
  out_var->value.f = name(E);\
  \
  return out_var;\
}


//2 arguments: 1 int, 1 float
#define XRL_2IF(name) IDL_VPTR IDL_CDECL IDL_ ## name(int argc, IDL_VPTR argv[])\
{\
  int Z;\
  float E;\
  IDL_VPTR out_var;\
  \
  Z = IDL_LongScalar(argv[0]);\
  E = (float)IDL_DoubleScalar(argv[1]);\
  \
  out_var = IDL_Gettmp();\
  out_var->type = IDL_TYP_FLOAT;\
  out_var->value.f = name(Z, E);\
  \
  return out_var;\
}

//2 arguments: 2 int
#define XRL_2II(name) IDL_VPTR IDL_CDECL IDL_ ## name(int argc, IDL_VPTR argv[])\
{\
  int Z;\
  int Z2;\
  IDL_VPTR out_var;\
  \
  Z = IDL_LongScalar(argv[0]);\
  Z2 = IDL_LongScalar(argv[1]);\
  \
  out_var = IDL_Gettmp();\
  out_var->type = IDL_TYP_FLOAT;\
  out_var->value.f = name(Z, Z2);\
  \
  return out_var;\
}

//2 arguments: 2 float
#define XRL_2FF(name) IDL_VPTR IDL_CDECL IDL_ ## name(int argc, IDL_VPTR argv[])\
{\
  float Z;\
  float Z2;\
  IDL_VPTR out_var;\
  \
  Z = (float) IDL_DoubleScalar(argv[0]);\
  Z2 = (float) IDL_DoubleScalar(argv[1]);\
  \
  out_var = IDL_Gettmp();\
  out_var->type = IDL_TYP_FLOAT;\
  out_var->value.f = name(Z, Z2);\
  \
  return out_var;\
}

//3 arguments: int, float, float
#define XRL_3IFF(name) IDL_VPTR IDL_CDECL IDL_ ## name(int argc, IDL_VPTR argv[])\
{\
  int Z;\
  float Z2;\
  float Z3;\
  IDL_VPTR out_var;\
  \
  Z = IDL_LongScalar(argv[0]);\
  Z2 = (float) IDL_DoubleScalar(argv[1]);\
  Z3 = (float) IDL_DoubleScalar(argv[2]);\
  \
  out_var = IDL_Gettmp();\
  out_var->type = IDL_TYP_FLOAT;\
  out_var->value.f = name(Z, Z2, Z3);\
  \
  return out_var;\
}

//3 arguments: float, float, float
#define XRL_3FFF(name) IDL_VPTR IDL_CDECL IDL_ ## name(int argc, IDL_VPTR argv[])\
{\
  float Z;\
  float Z2;\
  float Z3;\
  IDL_VPTR out_var;\
  \
  Z = (float) IDL_DoubleScalar(argv[0]);\
  Z2 = (float) IDL_DoubleScalar(argv[1]);\
  Z3 = (float) IDL_DoubleScalar(argv[2]);\
  \
  out_var = IDL_Gettmp();\
  out_var->type = IDL_TYP_FLOAT;\
  out_var->value.f = name(Z, Z2, Z3);\
  \
  return out_var;\
}

//3 arguments: int, int, float
#define XRL_3IIF(name) IDL_VPTR IDL_CDECL IDL_ ## name(int argc, IDL_VPTR argv[])\
{\
  int Z;\
  int Z2;\
  float Z3;\
  IDL_VPTR out_var;\
  \
  Z = IDL_LongScalar(argv[0]);\
  Z2 = IDL_LongScalar(argv[1]);\
  Z3 = (float) IDL_DoubleScalar(argv[2]);\
  \
  out_var = IDL_Gettmp();\
  out_var->type = IDL_TYP_FLOAT;\
  out_var->value.f = name(Z, Z2, Z3);\
  \
  return out_var;\
}

//4 arguments: int, float, float, float
#define XRL_4IFFF(name) IDL_VPTR IDL_CDECL IDL_ ## name(int argc, IDL_VPTR argv[])\
{\
  int Z;\
  float Z2;\
  float Z3;\
  float Z4;\
  IDL_VPTR out_var;\
  \
  Z = IDL_LongScalar(argv[0]);\
  Z2 = (float) IDL_DoubleScalar(argv[1]);\
  Z3 = (float) IDL_DoubleScalar(argv[2]);\
  Z4 = (float) IDL_DoubleScalar(argv[3]);\
  \
  out_var = IDL_Gettmp();\
  out_var->type = IDL_TYP_FLOAT;\
  out_var->value.f = name(Z, Z2, Z3, Z4);\
  \
  return out_var;\
}


XRL_1I(AtomicWeight)
XRL_1F(CS_KN)
XRL_1F(DCS_Thoms)
XRL_2IF(CS_Total)
XRL_2IF(CS_Photo)
XRL_2IF(CS_Rayl)
XRL_2IF(CS_Compt)
XRL_2IF(CSb_Total)
XRL_2IF(CSb_Photo)
XRL_2IF(CSb_Rayl)
XRL_2IF(CSb_Compt)
XRL_2IF(FF_Rayl)
XRL_2IF(SF_Compt)
XRL_2IF(Fi)
XRL_2IF(Fii)
XRL_2IF(CS_Photo_Total)
XRL_2IF(CSb_Photo_Total)
XRL_2IF(CS_Total_Kissel)
XRL_2IF(CSb_Total_Kissel)
XRL_2II(LineEnergy)
XRL_2II(FluorYield)
XRL_2II(CosKronTransProb)
XRL_2II(EdgeEnergy)
XRL_2II(JumpFactor)
XRL_2II(RadRate)
XRL_2FF(DCS_KN)
XRL_2FF(DCSP_Thoms)
XRL_2FF(MomentTransf)
XRL_2FF(ComptonEnergy)
XRL_3IFF(DCS_Rayl)
XRL_3IFF(DCS_Compt)
XRL_3IFF(DCSb_Rayl)
XRL_3IFF(DCSb_Compt)
XRL_3FFF(DCSP_KN)
XRL_3IIF(CS_FluorLine)
XRL_3IIF(CSb_FluorLine)
XRL_3IIF(CS_Photo_Partial)
XRL_3IIF(CSb_Photo_Partial)
XRL_3IIF(CS_FluorLine_Kissel)
XRL_3IIF(CSb_FluorLine_Kissel)
XRL_4IFFF(DCSP_Rayl)
XRL_4IFFF(DCSP_Compt)
XRL_4IFFF(DCSPb_Rayl)
XRL_4IFFF(DCSPb_Compt)

int IDL_Load (void) 
{
	//register the routines
	return IDL_SysRtnAdd(xrl_functions, TRUE, IDL_CARRAY_ELTS(xrl_functions)) &&
                IDL_SysRtnAdd(xrl_procedures, FALSE, IDL_CARRAY_ELTS(xrl_procedures));
}
