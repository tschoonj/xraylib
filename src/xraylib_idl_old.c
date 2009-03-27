#include <stdio.h>
#include "export.h"
#include "xraylib.h"

extern int HardExit, ExitStatus;

// Error Handling
void IDL_SetHardExit(int argc, IDL_VPTR argv[])
{
  HardExit = (int)IDL_LongScalar(argv[0]);
}

void IDL_SetExitStatus(int argc, IDL_VPTR argv[])
{
  ExitStatus = (int)IDL_LongScalar(argv[0]);
}

IDL_VPTR IDL_GetExitStatus()
{
  IDL_VPTR out_var;

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_INT;
  out_var->value.i = ExitStatus;

  return out_var;
}


// Atomic weights
IDL_VPTR IDL_AtomicWeight(int argc, IDL_VPTR argv[])
{
  int Z;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = AtomicWeight(Z);

  return out_var;
}

// Cross sections (cm2/g)
IDL_VPTR IDL_CS_Total(int argc, IDL_VPTR argv[])
{
  int Z;
  float E;
  IDL_VPTR out_var;

   Z = (int)IDL_LongScalar(argv[0]);
   E = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CS_Total(Z, E);

  return out_var;
}

IDL_VPTR IDL_CS_Photo(int argc, IDL_VPTR argv[])
{
  int Z;
  float E;
  IDL_VPTR out_var;

   Z = (int)IDL_LongScalar(argv[0]);
   E = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CS_Photo(Z, E);

  return out_var;
}

IDL_VPTR IDL_CS_Rayl(int argc, IDL_VPTR argv[])
{
  int Z;
  float E;
  IDL_VPTR out_var;

   Z = (int)IDL_LongScalar(argv[0]);
   E = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CS_Rayl(Z, E);

  return out_var;
}

IDL_VPTR IDL_CS_Compt(int argc, IDL_VPTR argv[])
{
  int Z;
  float E;
  IDL_VPTR out_var;

   Z = (int)IDL_LongScalar(argv[0]);
   E = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CS_Compt(Z, E);

  return out_var;
}

// barn/atom
IDL_VPTR IDL_CSb_Total(int argc, IDL_VPTR argv[])
{
  int Z;
  float E;
  IDL_VPTR out_var;

   Z = (int)IDL_LongScalar(argv[0]);
   E = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CSb_Total(Z, E);

  return out_var;
}

IDL_VPTR IDL_CSb_Photo(int argc, IDL_VPTR argv[])
{
  int Z;
  float E;
  IDL_VPTR out_var;

   Z = (int)IDL_LongScalar(argv[0]);
   E = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CSb_Photo(Z, E);

  return out_var;
}

IDL_VPTR IDL_CSb_Rayl(int argc, IDL_VPTR argv[])
{
  int Z;
  float E;
  IDL_VPTR out_var;

   Z = (int)IDL_LongScalar(argv[0]);
   E = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CSb_Rayl(Z, E);

  return out_var;
}

IDL_VPTR IDL_CSb_Compt(int argc, IDL_VPTR argv[])
{
  int Z;
  float E;
  IDL_VPTR out_var;

   Z = (int)IDL_LongScalar(argv[0]);
   E = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CSb_Compt(Z, E);

  return out_var;
}

IDL_VPTR IDL_CS_KN(int argc, IDL_VPTR argv[])
{
  float E;
  IDL_VPTR out_var;

  E = (float)IDL_DoubleScalar(argv[0]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CS_KN(E);

  return out_var;
}

// Unpolarized differential scattering cross sections
IDL_VPTR IDL_DCS_Thoms(int argc, IDL_VPTR argv[])
{
  float theta;
  IDL_VPTR out_var;

  theta = (float)IDL_DoubleScalar(argv[0]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = DCS_Thoms(theta);;

  return out_var;
}

IDL_VPTR IDL_DCS_KN(int argc, IDL_VPTR argv[])
{
  float E, theta;
  IDL_VPTR out_var;

  E = (float)IDL_DoubleScalar(argv[0]);
  theta = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = DCS_KN(E, theta);

  return out_var;
}

IDL_VPTR IDL_DCS_Rayl(int argc, IDL_VPTR argv[])
{
  int Z;
  float E, theta;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  E = (float)IDL_DoubleScalar(argv[1]);
  theta = (float)IDL_DoubleScalar(argv[2]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = DCS_Rayl(Z, E, theta);

  return out_var;
}

IDL_VPTR IDL_DCS_Compt(int argc, IDL_VPTR argv[])
{
  int Z;
  float E, theta;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  E = (float)IDL_DoubleScalar(argv[1]);
  theta = (float)IDL_DoubleScalar(argv[2]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = DCS_Compt(Z, E, theta);

  return out_var;
}

IDL_VPTR IDL_DCSb_Rayl(int argc, IDL_VPTR argv[])
{
  int Z;
  float E, theta;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  E = (float)IDL_DoubleScalar(argv[1]);
  theta = (float)IDL_DoubleScalar(argv[2]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = DCSb_Rayl(Z, E, theta);

  return out_var;
}

IDL_VPTR IDL_DCSb_Compt(int argc, IDL_VPTR argv[])
{
  int Z;
  float E, theta;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  E = (float)IDL_DoubleScalar(argv[1]);
  theta = (float)IDL_DoubleScalar(argv[2]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = DCSb_Compt(Z, E, theta);

  return out_var;
}

 
// Polarized differential scattering cross sections

IDL_VPTR IDL_DCSP_Thoms(int argc, IDL_VPTR argv[])
{
  float theta, phi;
  IDL_VPTR out_var;

  theta = (float)IDL_DoubleScalar(argv[0]);
  phi = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = DCSP_Thoms(theta, phi);

  return out_var;
}

IDL_VPTR IDL_DCSP_KN(int argc, IDL_VPTR argv[])
{
  float E, theta, phi;
  IDL_VPTR out_var;

  E = (float)IDL_DoubleScalar(argv[0]);
  theta = (float)IDL_DoubleScalar(argv[1]);
  phi = (float)IDL_DoubleScalar(argv[2]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = DCSP_KN(E, theta, phi);

  return out_var;
}

IDL_VPTR IDL_DCSP_Rayl(int argc, IDL_VPTR argv[])
{
  int Z;
  float E, theta, phi;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  E = (float)IDL_DoubleScalar(argv[1]);
  theta = (float)IDL_DoubleScalar(argv[2]);
  phi = (float)IDL_DoubleScalar(argv[3]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = DCSP_Rayl(Z, E, theta, phi);

  return out_var;
}

IDL_VPTR IDL_DCSP_Compt(int argc, IDL_VPTR argv[])
{
  int Z;
  float E, theta, phi;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  E = (float)IDL_DoubleScalar(argv[1]);
  theta = (float)IDL_DoubleScalar(argv[2]);
  phi = (float)IDL_DoubleScalar(argv[3]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = DCSP_Compt(Z, E, theta, phi);

  return out_var;
}
  
IDL_VPTR IDL_DCSPb_Rayl(int argc, IDL_VPTR argv[])
{
  int Z;
  float E, theta, phi;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  E = (float)IDL_DoubleScalar(argv[1]);
  theta = (float)IDL_DoubleScalar(argv[2]);
  phi = (float)IDL_DoubleScalar(argv[3]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = DCSPb_Rayl(Z, E, theta, phi);

  return out_var;
}

IDL_VPTR IDL_DCSPb_Compt(int argc, IDL_VPTR argv[])
{
  int Z;
  float E, theta, phi;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  E = (float)IDL_DoubleScalar(argv[1]);
  theta = (float)IDL_DoubleScalar(argv[2]);
  phi = (float)IDL_DoubleScalar(argv[3]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = DCSPb_Compt(Z, E, theta, phi);

  return out_var;
}

// Scattering factors
IDL_VPTR IDL_FF_Rayl(int argc, IDL_VPTR argv[])
{
  int Z;
  float q;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  q = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = FF_Rayl(Z, q);

  return out_var;
}

IDL_VPTR IDL_SF_Compt(int argc, IDL_VPTR argv[])
{
  int Z;
  float q;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  q = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = SF_Compt(Z, q);

  return out_var;
}

IDL_VPTR IDL_MomentTransf(int argc, IDL_VPTR argv[])
{
  float E, theta;
  IDL_VPTR out_var;

  E = (float)IDL_DoubleScalar(argv[0]);
  theta = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = MomentTransf(E, theta);

  return out_var;
}

// X-ray fluorescent line energy
IDL_VPTR IDL_LineEnergy(int argc, IDL_VPTR argv[])
{
  int Z, line;
  IDL_VPTR out_var;

   Z = (int)IDL_LongScalar(argv[0]);
   line = (int)IDL_LongScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = LineEnergy(Z, line);

  return out_var;
}

// Fluorescence yield 
IDL_VPTR IDL_FluorYield(int argc, IDL_VPTR argv[])
{
  int Z, shell;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  shell = (int)IDL_LongScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = FluorYield(Z, shell);

  return out_var;
}

// Coster-Kronig transition Probability
IDL_VPTR IDL_CosKronTransProb(int argc, IDL_VPTR argv[])
{
  int Z, trans;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  trans = (int)IDL_LongScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CosKronTransProb(Z, trans);

  return out_var;
}

// Absorption-edge energies
IDL_VPTR IDL_EdgeEnergy(int argc, IDL_VPTR argv[])
{
  int Z, shell;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  shell = (int)IDL_LongScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = EdgeEnergy(Z, shell);

  return out_var;
}

// Jump ratio
IDL_VPTR IDL_JumpFactor(int argc, IDL_VPTR argv[])
{
  int Z, shell;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  shell = (int)IDL_LongScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = JumpFactor(Z, shell);

  return out_var;
}

// Fluorescent-lines cross sections
IDL_VPTR IDL_CS_FluorLine(int argc, IDL_VPTR argv[])
{
  int Z;
  float line, E;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  line = (float)IDL_DoubleScalar(argv[1]);
  E = (float)IDL_DoubleScalar(argv[2]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CS_FluorLine(Z, line, E);

  return out_var;
}

IDL_VPTR IDL_CSb_FluorLine(int argc, IDL_VPTR argv[])
{
  int Z;
  float line, E;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  line = (float)IDL_DoubleScalar(argv[1]);
  E = (float)IDL_DoubleScalar(argv[2]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CSb_FluorLine(Z, line, E);

  return out_var;
}

// Fractional radiative rate
IDL_VPTR IDL_RadRate(int argc, IDL_VPTR argv[])
{
  int Z, line;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  line = (int)IDL_LongScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = RadRate(Z, line);

  return out_var;
}

// Photon energy after Compton scattering
IDL_VPTR IDL_ComptonEnergy(int argc, IDL_VPTR argv[])
{
  float E0, theta;
  IDL_VPTR out_var;

  E0 = (float)IDL_DoubleScalar(argv[0]);
  theta = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = ComptonEnergy(E0, theta);

  return out_var;
}

//Anomalous Scattering Factors
IDL_VPTR IDL_Fi (int argc, IDL_VPTR argv[])
{
  int Z;
  float E;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  E = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = Fi(Z, E);

  return out_var;

}

IDL_VPTR IDL_Fii (int argc, IDL_VPTR argv[])
{
  int Z;
  float E;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  E = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = Fii(Z, E);

  return out_var;

}

IDL_VPTR IDL_CS_Photo_Total (int argc, IDL_VPTR argv[])
{
  int Z;
  float E;
  IDL_VPTR out_var;

  Z = (int)IDL_LongScalar(argv[0]);
  E = (float)IDL_DoubleScalar(argv[1]);

  out_var = IDL_Gettmp();
  out_var->type = IDL_TYP_FLOAT;
  out_var->value.f = CS_Photo_Total(Z, E);

  return out_var;

}
