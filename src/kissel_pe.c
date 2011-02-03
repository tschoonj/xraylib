/*
Copyright (c) 2009, 2010, 2011, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <math.h>
#include "splint.h"
#include "xrayglob.h"
#include "xraylib.h"
#include "xrf_cross_sections_aux.h"


///////////////////////////////////////////////////////////
//                                                       //
//        Photoelectric cross section  (barns/atom)      //
//                  Using the Kissel data                //
//                                                       //
//    Z : atomic number                                  //
//    E : energy (keV)                                   //
//                                                       //
///////////////////////////////////////////////////////////
float CSb_Photo_Total(int Z, float E) {
  double ln_E, ln_sigma, sigma;
  int shell;
  float rv = 0.0;

  if (Z<1 || Z>ZMAX || NE_Photo_Total_Kissel[Z]<0) {
    ErrorExit("Z out of range in function CSb_Photo_Total");
    return 0.0;
  }
  if (E <= 0.) {
    ErrorExit("Energy <=0 in function CSb_Photo_Total");
    return 0.0;
  }
/*  ln_E = log((double) E);
  splintd(E_Photo_Total_Kissel[Z]-1, Photo_Total_Kissel[Z]-1, Photo_Total_Kissel2[Z]-1,NE_Photo_Total_Kissel[Z], ln_E, &ln_sigma);

  sigma = exp(ln_sigma);

  return (float) sigma; 
*/
  for (shell = K_SHELL ; shell <= Q3_SHELL ; shell++) {
    if (Electron_Config_Kissel[Z][shell] > 1.0E-06 && E >= EdgeEnergy_arr[Z][shell] ) {
  	rv += CSb_Photo_Partial(Z,shell,E)*Electron_Config_Kissel[Z][shell];
    }
  }
  return rv;
}

///////////////////////////////////////////////////////////
//                                                       //
//        Photoelectric cross section  (cm2/g)           //
//                  Using the Kissel data                //
//                                                       //
//    Z : atomic number                                  //
//    E : energy (keV)                                   //
//                                                       //
///////////////////////////////////////////////////////////

float CS_Photo_Total(int Z, float E) {
  return CSb_Photo_Total(Z, E)*AVOGNUM/AtomicWeight_arr[Z];
}


///////////////////////////////////////////////////////////
//                                                       //
//   Partial Photoelectric cross section  (barns/elec)   //
//                  Using the Kissel data                //
//                                                       //
//    Z : atomic number                                  //
//    shell : shell                                      //
//    E : energy (keV)                                   //
//                                                       //
///////////////////////////////////////////////////////////

float CSb_Photo_Partial(int Z, int shell, float E) {
  double ln_E, ln_sigma, sigma;
  double x0, x1, y0, y1;

  if (Z < 1 || Z > ZMAX) {
    ErrorExit("Z out of range in function CSb_Photo_Partial");
    return 0.0;
  }
  if (shell < 0 || shell >= SHELLNUM_K) {
    ErrorExit("shell out of range in function CSb_Photo_Partial");
    return 0.0;
  }
  if (E <= 0.0) {
    ErrorExit("Energy <= 0.0 in function CSb_Photo_Partial");
    return 0.0;
  }
  if (Electron_Config_Kissel[Z][shell] < 1.0E-06){
    ErrorExit("selected orbital is unoccupied");
    return 0.0;
  } 
  
  if (EdgeEnergy_arr[Z][shell] > E) {
    ErrorExit("selected energy cannot excite the orbital: energy must be greater than the absorption edge energy");
    return 0.0;
  } 
  else {
    ln_E = log((double) E);
    if (EdgeEnergy_Kissel[Z][shell] > EdgeEnergy_arr[Z][shell] && E < EdgeEnergy_Kissel[Z][shell]) {
   	//use log-log extrapolation 
	x0 = E_Photo_Partial_Kissel[Z][shell][0];
	x1 = E_Photo_Partial_Kissel[Z][shell][1];
	y0 = Photo_Partial_Kissel[Z][shell][0];
	y1 = Photo_Partial_Kissel[Z][shell][1];
	ln_sigma = y0+(y1-y0)*(ln_E-x0)/(x1-x0);
    }
    else {
    	splintd(E_Photo_Partial_Kissel[Z][shell]-1, Photo_Partial_Kissel[Z][shell]-1, Photo_Partial_Kissel2[Z][shell]-1,NE_Photo_Partial_Kissel[Z][shell], ln_E, &ln_sigma);
   }
 sigma = exp(ln_sigma);


    return (float) sigma; 

  }
}

///////////////////////////////////////////////////////////
//                                                       //
//   Partial Photoelectric cross section  (cm2/g)        //
//                  Using the Kissel data                //
//                                                       //
//    Z : atomic number                                  //
//    shell : shell                                      //
//    E : energy (keV)                                   //
//                                                       //
///////////////////////////////////////////////////////////


float CS_Photo_Partial(int Z, int shell, float E) {
  return CSb_Photo_Partial(Z, shell, E)*Electron_Config_Kissel[Z][shell]*AVOGNUM/AtomicWeight_arr[Z];
}


//////////////////////////////////////////////////////////////////////
//                                                                  //
//                    Fluorescent line cross section (cm2/g)        //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//          line :                                                  //
//            KA_LINE 0                                             //
//            KB_LINE 1                                             //
//            LA_LINE 2                                             //
//            LB_LINE 3                                             //
//                                                                  //
//////////////////////////////////////////////////////////////////////

/*float CS_FluorLine_Kissel(int Z, int line, float E) {

  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function CS_FluorLine_Kissel");
    return 0.0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function CS_FluorLine_Kissel");
    return 0.0;
  }

  if (line>=KN5_LINE && line<=KB_LINE) {
    //K lines
    return CS_Photo_Partial(Z, K_SHELL, E)*FluorYield(Z, K_SHELL)*RadRate(Z,line);
  }
  else if (line>=L1P5_LINE && line<=L1M1_LINE) {
    //L1 lines
    return CS_Photo_Partial(Z, L1_SHELL, E)*FluorYield(Z, L1_SHELL)*RadRate(Z,line);
  }
  else if (line>=L2Q1_LINE && line<=L2M1_LINE) {
    //L2 lines
    return (FluorYield(Z, L2_SHELL)*RadRate(Z,line))*
		(CS_Photo_Partial(Z, L2_SHELL, E)+
		(CS_Photo_Partial(Z, L1_SHELL, E)*CosKronTransProb(Z,F12_TRANS)));
  }
  else if (line>=L3Q1_LINE && line<=L3M1_LINE) {
    //L3 lines
    return (FluorYield(Z, L3_SHELL)*RadRate(Z,line))*
		(CS_Photo_Partial(Z, L3_SHELL, E)+
		(CS_Photo_Partial(Z, L2_SHELL, E)*CosKronTransProb(Z,F23_TRANS))+
		(CS_Photo_Partial(Z, L1_SHELL, E)*(CosKronTransProb(Z,F13_TRANS) + CosKronTransProb(Z,FP13_TRANS) + CosKronTransProb(Z,F12_TRANS) * CosKronTransProb(Z,F23_TRANS)))
		);

  }
  else if (line == LA_LINE) {
    return (CS_FluorLine_Kissel(Z,L3M4_LINE,E)+CS_FluorLine_Kissel(Z,L3M5_LINE,E)); 
  }
  else if (line == LB_LINE) {
    return (CS_FluorLine_Kissel(Z,L2M4_LINE,E)+
    	CS_FluorLine_Kissel(Z,L2M3_LINE,E)+
        CS_FluorLine_Kissel(Z,L3N5_LINE,E)+
        CS_FluorLine_Kissel(Z,L3O4_LINE,E)+
	CS_FluorLine_Kissel(Z,L3O5_LINE,E)+
	CS_FluorLine_Kissel(Z,L3O45_LINE,E)+
	CS_FluorLine_Kissel(Z,L3N1_LINE,E)+
	CS_FluorLine_Kissel(Z,L3O1_LINE,E)+
	CS_FluorLine_Kissel(Z,L3N6_LINE,E)+
	CS_FluorLine_Kissel(Z,L3N7_LINE,E)+
	CS_FluorLine_Kissel(Z,L3N4_LINE,E)+
	CS_FluorLine_Kissel(Z,L1M3_LINE,E)+
	CS_FluorLine_Kissel(Z,L1M2_LINE,E)+
	CS_FluorLine_Kissel(Z,L1M5_LINE,E)+
	CS_FluorLine_Kissel(Z,L1M4_LINE,E)
    );
  }
#define FM12 CosKronTransProb(Z,FM12_TRANS)
#define FM23 CosKronTransProb(Z,FM23_TRANS)
#define FM34 CosKronTransProb(Z,FM34_TRANS)
#define FM45 CosKronTransProb(Z,FM45_TRANS)
#define FM13 CosKronTransProb(Z,FM13_TRANS)
#define FM24 CosKronTransProb(Z,FM24_TRANS)
#define FM14 CosKronTransProb(Z,FM14_TRANS)
#define FM35 CosKronTransProb(Z,FM35_TRANS)
#define FM25 CosKronTransProb(Z,FM25_TRANS)
#define FM15 CosKronTransProb(Z,FM15_TRANS)
  else if (line>=M1P5_LINE && line<=M1N1_LINE) {
    //M1 lines
    return CS_Photo_Partial(Z, M1_SHELL, E)*FluorYield(Z, M1_SHELL)*RadRate(Z,line);
  }
  else if (line>=M2P5_LINE && line<=M2N1_LINE) {
    //M2 lines
    return (FluorYield(Z, M2_SHELL)*RadRate(Z,line))*
		(CS_Photo_Partial(Z, M2_SHELL, E)+
		(CS_Photo_Partial(Z, M1_SHELL, E)*FM12));
  }
  else if (line>=M3Q1_LINE && line<=M3N1_LINE) {
    //M3 lines
    return (FluorYield(Z, M3_SHELL)*RadRate(Z,line))*
		(CS_Photo_Partial(Z, M3_SHELL, E)+
		(CS_Photo_Partial(Z, M2_SHELL, E)*FM23)+
		(CS_Photo_Partial(Z, M1_SHELL, E)*(FM13 + FM12 * FM23))
		);

  }
  else if (line>=M4P5_LINE && line<=M4N1_LINE) {
    //M4 lines
    return (FluorYield(Z, M4_SHELL)*RadRate(Z,line))*
		(CS_Photo_Partial(Z, M4_SHELL, E)+
		(CS_Photo_Partial(Z, M3_SHELL, E)*FM34)+
		(CS_Photo_Partial(Z, M2_SHELL, E)*(FM24 + FM34*FM23))+
		(CS_Photo_Partial(Z, M1_SHELL, E)*(FM12*FM23*FM34 + FM13*FM34 + FM12*FM24 + FM14)    )
		);

  }
  else if (line>=M5P5_LINE && line<=M5N1_LINE) {
 
    //M5 lines
    return (FluorYield(Z, M5_SHELL)*RadRate(Z,line))*
		(CS_Photo_Partial(Z, M5_SHELL, E)+
		(CS_Photo_Partial(Z, M4_SHELL, E)*FM45)+
		(CS_Photo_Partial(Z, M3_SHELL, E)*(FM35 + FM34 * FM45))+
		(CS_Photo_Partial(Z, M2_SHELL, E)*(FM23 * FM34 * FM45 + FM24 * FM45 + FM23 * FM35 + FM25)    )+
		(CS_Photo_Partial(Z, M1_SHELL, E)*(FM12*FM23*FM34*FM45 + FM13*FM34*FM45 + FM12*FM24*FM45 + FM14*FM45 + FM12*FM23*FM35 + FM13*FM35 + FM12*FM25 + FM15)
		));

  }
  else {
    ErrorExit("Line not allowed in function CS_FluorLine_Kissel");
    return 0.0;
  }  
}*/

//////////////////////////////////////////////////////////////////////
//                                                                  //
//                    Fluorescent line cross section (barns/atom)   //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//          line :                                                  //
//            KA_LINE 0                                             //
//            KB_LINE 1                                             //
//            LA_LINE 2                                             //
//            LB_LINE 3                                             //
//                                                                  //
//////////////////////////////////////////////////////////////////////

/*float CSb_FluorLine_Kissel(int Z, int line, float E) {
  return CS_FluorLine_Kissel(Z, line, E)*AtomicWeight_arr[Z]/AVOGNUM;
}*/

//////////////////////////////////////////////////////////////////////
//                                                                  //
//                  Total cross section  (cm2/g)                    //
//         (Photoelectric (Kissel) + Compton + Rayleigh)            //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
//////////////////////////////////////////////////////////////////////
float CS_Total_Kissel(int Z, float E) { 

  if (Z<1 || Z>ZMAX || NE_Photo_Total_Kissel[Z]<0 || NE_Rayl[Z]<0 || NE_Compt[Z]<0) {
    ErrorExit("Z out of range in function CS_Total_Kissel");
    return 0.0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function CS_Total_Kissel");
    return 0.0;
  }

  return CS_Photo_Total(Z, E) + CS_Rayl(Z, E) + CS_Compt(Z, E);

}

//////////////////////////////////////////////////////////////////////
//                                                                  //
//                  Total cross section  (barn/atom)                //
//         (Photoelectric (Kissel) + Compton + Rayleigh)            //
//                                                                  //
//          Z : atomic number                                       //
//          E : energy (keV)                                        //
//                                                                  //
//////////////////////////////////////////////////////////////////////

float CSb_Total_Kissel(int Z, float E) {

  return CS_Total_Kissel(Z,E)*AtomicWeight_arr[Z]/AVOGNUM;
}


float ElectronConfig(int Z, int shell) {

  if (Z<1 || Z>ZMAX  ) {
    ErrorExit("Z out of range in function ElectronConfig");
    return 0.0;
  }

  if (shell < 0 || shell >= SHELLNUM_K ) {
    ErrorExit("shell out of range in function ElectronConfig");
    return 0.0;
  }

  return Electron_Config_Kissel[Z][shell]; 

}

//new functions...
float CS_FluorLine_Kissel_no_Cascade(int Z, int line, float E) {
  float PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, PM5;

  PK = PL1 = PL2 = PL3 = PM1 = PM2 = PM3 = PM4 = PM5 = 0.0;


  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function CS_FluorLine_Kissel_no_Cascade");
    return 0.0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function CS_FluorLine_Kissel_no_Cascade");
    return 0.0;
  }

  if (line>=KN5_LINE && line<=KB_LINE) {
    //K lines -> never cascade effect!
    return CS_Photo_Partial(Z, K_SHELL, E)*FluorYield(Z, K_SHELL)*RadRate(Z,line);
  }
  else if (line>=L1P5_LINE && line<=L1M1_LINE) {
    //L1 lines
    return PL1_pure_kissel(Z,E)*FluorYield(Z, L1_SHELL)*RadRate(Z,line);
  }
  else if (line>=L2Q1_LINE && line<=L2M1_LINE) {
    //L2 lines
    PL1 = PL1_pure_kissel(Z,E);
    return (FluorYield(Z, L2_SHELL)*RadRate(Z,line))*
		PL2_pure_kissel(Z, E, PL1);
  }
  else if (line>=L3Q1_LINE && line<=L3M1_LINE) {
    //L3 lines
    PL1 = PL1_pure_kissel(Z,E);
    PL2 = PL2_pure_kissel(Z, E, PL1);
    return (FluorYield(Z, L3_SHELL)*RadRate(Z,line))*PL3_pure_kissel(Z, E, PL1, PL2);
  }
  else if (line == LA_LINE) {
    return (CS_FluorLine_Kissel_no_Cascade(Z,L3M4_LINE,E)+CS_FluorLine_Kissel_no_Cascade(Z,L3M5_LINE,E)); 
  }
  else if (line == LB_LINE) {
    return (CS_FluorLine_Kissel_no_Cascade(Z,L2M4_LINE,E)+
    	CS_FluorLine_Kissel_no_Cascade(Z,L2M3_LINE,E)+
        CS_FluorLine_Kissel_no_Cascade(Z,L3N5_LINE,E)+
        CS_FluorLine_Kissel_no_Cascade(Z,L3O4_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade(Z,L3O5_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade(Z,L3O45_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade(Z,L3N1_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade(Z,L3O1_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade(Z,L3N6_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade(Z,L3N7_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade(Z,L3N4_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade(Z,L1M3_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade(Z,L1M2_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade(Z,L1M5_LINE,E)+
	CS_FluorLine_Kissel_no_Cascade(Z,L1M4_LINE,E)
    );
  }
  else if (line>=M1P5_LINE && line<=M1N1_LINE) {
    //M1 lines
    return PM1_pure_kissel(Z, E)*FluorYield(Z, M1_SHELL)*RadRate(Z,line);
  }
  else if (line>=M2P5_LINE && line<=M2N1_LINE) {
    //M2 lines
    PM1 = PM1_pure_kissel(Z, E);
    return (FluorYield(Z, M2_SHELL)*RadRate(Z,line))*
		PM2_pure_kissel(Z, E, PM1);
  }
  else if (line>=M3Q1_LINE && line<=M3N1_LINE) {
    //M3 lines
    PM1 = PM1_pure_kissel(Z, E);
    PM2 = PM2_pure_kissel(Z, E, PM1);
    return (FluorYield(Z, M3_SHELL)*RadRate(Z,line))*
		PM3_pure_kissel(Z, E, PM1, PM2);
  }
  else if (line>=M4P5_LINE && line<=M4N1_LINE) {
    //M4 lines
    PM1 = PM1_pure_kissel(Z, E);
    PM2 = PM2_pure_kissel(Z, E, PM1);
    PM3 = PM3_pure_kissel(Z, E, PM1, PM2);
    return (FluorYield(Z, M4_SHELL)*RadRate(Z,line))*
		PM4_pure_kissel(Z, E, PM1, PM2, PM3);
  }
  else if (line>=M5P5_LINE && line<=M5N1_LINE) {
    //M5 lines
    PM1 = PM1_pure_kissel(Z, E);
    PM2 = PM2_pure_kissel(Z, E, PM1);
    PM3 = PM3_pure_kissel(Z, E, PM1, PM2);
    PM4 = PM4_pure_kissel(Z, E, PM1, PM2, PM3);
    return (FluorYield(Z, M5_SHELL)*RadRate(Z,line))*
		PM5_pure_kissel(Z, E, PM1, PM2, PM3, PM4);

  }
  else {
    ErrorExit("Line not allowed in function CS_FluorLine_Kissel_no_Cascade");
    return 0.0;
  }  
}

float CS_FluorLine_Kissel_Radiative_Cascade(int Z, int line, float E) {
  float PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, PM5;

  PK = PL1 = PL2 = PL3 = PM1 = PM2 = PM3 = PM4 = PM5 = 0.0;


  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function CS_FluorLine_Kissel_Radiative_Cascade");
    return 0.0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function CS_FluorLine_Kissel_Radiative_Cascade");
    return 0.0;
  }

  if (line>=KN5_LINE && line<=KB_LINE) {
    //K lines -> never cascade effect!
    return CS_Photo_Partial(Z, K_SHELL, E)*FluorYield(Z, K_SHELL)*RadRate(Z,line);
  }
  else if (line>=L1P5_LINE && line<=L1M1_LINE) {
    //L1 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    return PL1_rad_cascade_kissel(Z, E, PK)*FluorYield(Z, L1_SHELL)*RadRate(Z,line);
  }
  else if (line>=L2Q1_LINE && line<=L2M1_LINE) {
    //L2 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel(Z,E, PK);
    return (FluorYield(Z, L2_SHELL)*RadRate(Z,line))*
		PL2_rad_cascade_kissel(Z, E, PK, PL1);
  }
  else if (line>=L3Q1_LINE && line<=L3M1_LINE) {
    //L3 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel(Z, E, PK);
    PL2 = PL2_rad_cascade_kissel(Z, E, PK, PL1);
    return (FluorYield(Z, L3_SHELL)*RadRate(Z,line))*PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2);
  }
  else if (line == LA_LINE) {
    return (CS_FluorLine_Kissel_Radiative_Cascade(Z,L3M4_LINE,E)+CS_FluorLine_Kissel_Radiative_Cascade(Z,L3M5_LINE,E)); 
  }
  else if (line == LB_LINE) {
    return (CS_FluorLine_Kissel_Radiative_Cascade(Z,L2M4_LINE,E)+
    	CS_FluorLine_Kissel_Radiative_Cascade(Z,L2M3_LINE,E)+
        CS_FluorLine_Kissel_Radiative_Cascade(Z,L3N5_LINE,E)+
        CS_FluorLine_Kissel_Radiative_Cascade(Z,L3O4_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade(Z,L3O5_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade(Z,L3O45_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade(Z,L3N1_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade(Z,L3O1_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade(Z,L3N6_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade(Z,L3N7_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade(Z,L3N4_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade(Z,L1M3_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade(Z,L1M2_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade(Z,L1M5_LINE,E)+
	CS_FluorLine_Kissel_Radiative_Cascade(Z,L1M4_LINE,E)
    );
  }
  else if (line>=M1P5_LINE && line<=M1N1_LINE) {
    //M1 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel(Z, E, PK);
    PL2 = PL2_rad_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2);
    return PM1_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3)*FluorYield(Z, M1_SHELL)*RadRate(Z,line);
  }
  else if (line>=M2P5_LINE && line<=M2N1_LINE) {
    //M2 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel(Z, E, PK);
    PL2 = PL2_rad_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2);
    PM1 = PM1_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    return (FluorYield(Z, M2_SHELL)*RadRate(Z,line))*
		PM2_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
  }
  else if (line>=M3Q1_LINE && line<=M3N1_LINE) {
    //M3 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel(Z, E, PK);
    PL2 = PL2_rad_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2);
    PM1 = PM1_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
    return (FluorYield(Z, M3_SHELL)*RadRate(Z,line))*
		PM3_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
  }
  else if (line>=M4P5_LINE && line<=M4N1_LINE) {
    //M4 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel(Z, E, PK);
    PL2 = PL2_rad_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2);
    PM1 = PM1_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
    PM3 = PM3_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    return (FluorYield(Z, M4_SHELL)*RadRate(Z,line))*
		PM4_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
  }
  else if (line>=M5P5_LINE && line<=M5N1_LINE) {
    //M5 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_rad_cascade_kissel(Z, E, PK);
    PL2 = PL2_rad_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_rad_cascade_kissel(Z, E, PK, PL1, PL2);
    PM1 = PM1_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
    PM3 = PM3_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    PM4 = PM4_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
    return (FluorYield(Z, M5_SHELL)*RadRate(Z,line))*
		PM5_rad_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4);
  }
  else {
    ErrorExit("Line not allowed in function CS_FluorLine_Kissel_Radiative_Cascade");
    return 0.0;
  }  
}

float CS_FluorLine_Kissel_Nonradiative_Cascade(int Z, int line, float E) {
  float PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, PM5;

  PK = PL1 = PL2 = PL3 = PM1 = PM2 = PM3 = PM4 = PM5 = 0.0;


  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function CS_FluorLine_Kissel_Nonradiative_Cascade");
    return 0.0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function CS_FluorLine_Kissel_Nonradiative_Cascade");
    return 0.0;
  }

  if (line>=KN5_LINE && line<=KB_LINE) {
    //K lines -> never cascade effect!
    return CS_Photo_Partial(Z, K_SHELL, E)*FluorYield(Z, K_SHELL)*RadRate(Z,line);
  }
  else if (line>=L1P5_LINE && line<=L1M1_LINE) {
    //L1 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    return PL1_auger_cascade_kissel(Z, E, PK)*FluorYield(Z, L1_SHELL)*RadRate(Z,line);
  }
  else if (line>=L2Q1_LINE && line<=L2M1_LINE) {
    //L2 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel(Z,E, PK);
    return (FluorYield(Z, L2_SHELL)*RadRate(Z,line))*
		PL2_auger_cascade_kissel(Z, E, PK, PL1);
  }
  else if (line>=L3Q1_LINE && line<=L3M1_LINE) {
    //L3 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel(Z, E, PK);
    PL2 = PL2_auger_cascade_kissel(Z, E, PK, PL1);
    return (FluorYield(Z, L3_SHELL)*RadRate(Z,line))*PL3_auger_cascade_kissel(Z, E, PK, PL1, PL2);
  }
  else if (line == LA_LINE) {
    return (CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L3M4_LINE,E)+CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L3M5_LINE,E)); 
  }
  else if (line == LB_LINE) {
    return (CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L2M4_LINE,E)+
    	CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L2M3_LINE,E)+
        CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L3N5_LINE,E)+
        CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L3O4_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L3O5_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L3O45_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L3N1_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L3O1_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L3N6_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L3N7_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L3N4_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L1M3_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L1M2_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L1M5_LINE,E)+
	CS_FluorLine_Kissel_Nonradiative_Cascade(Z,L1M4_LINE,E)
    );
  }
  else if (line>=M1P5_LINE && line<=M1N1_LINE) {
    //M1 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel(Z, E, PK);
    PL2 = PL2_auger_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_auger_cascade_kissel(Z, E, PK, PL1, PL2);
    return PM1_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3)*FluorYield(Z, M1_SHELL)*RadRate(Z,line);
  }
  else if (line>=M2P5_LINE && line<=M2N1_LINE) {
    //M2 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel(Z, E, PK);
    PL2 = PL2_auger_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_auger_cascade_kissel(Z, E, PK, PL1, PL2);
    PM1 = PM1_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    return (FluorYield(Z, M2_SHELL)*RadRate(Z,line))*
		PM2_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
  }
  else if (line>=M3Q1_LINE && line<=M3N1_LINE) {
    //M3 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel(Z, E, PK);
    PL2 = PL2_auger_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_auger_cascade_kissel(Z, E, PK, PL1, PL2);
    PM1 = PM1_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
    return (FluorYield(Z, M3_SHELL)*RadRate(Z,line))*
		PM3_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
  }
  else if (line>=M4P5_LINE && line<=M4N1_LINE) {
    //M4 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel(Z, E, PK);
    PL2 = PL2_auger_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_auger_cascade_kissel(Z, E, PK, PL1, PL2);
    PM1 = PM1_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
    PM3 = PM3_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    return (FluorYield(Z, M4_SHELL)*RadRate(Z,line))*
		PM4_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
  }
  else if (line>=M5P5_LINE && line<=M5N1_LINE) {
    //M5 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_auger_cascade_kissel(Z, E, PK);
    PL2 = PL2_auger_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_auger_cascade_kissel(Z, E, PK, PL1, PL2);
    PM1 = PM1_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
    PM3 = PM3_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    PM4 = PM4_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
    return (FluorYield(Z, M5_SHELL)*RadRate(Z,line))*
		PM5_auger_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4);
  }
  else {
    ErrorExit("Line not allowed in function CS_FluorLine_Kissel_Nonradiative_Cascade");
    return 0.0;
  }  
}

float CS_FluorLine_Kissel_Cascade(int Z, int line, float E) {
  float PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4, PM5;

  PK = PL1 = PL2 = PL3 = PM1 = PM2 = PM3 = PM4 = PM5 = 0.0;


  if (Z<1 || Z>ZMAX) {
    ErrorExit("Z out of range in function CS_FluorLine_Kissel_Cascade");
    return 0.0;
  }

  if (E <= 0.) {
    ErrorExit("Energy <=0 in function CS_FluorLine_Kissel_Cascade");
    return 0.0;
  }

  if (line>=KN5_LINE && line<=KB_LINE) {
    //K lines -> never cascade effect!
    return CS_Photo_Partial(Z, K_SHELL, E)*FluorYield(Z, K_SHELL)*RadRate(Z,line);
  }
  else if (line>=L1P5_LINE && line<=L1M1_LINE) {
    //L1 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    return PL1_full_cascade_kissel(Z, E, PK)*FluorYield(Z, L1_SHELL)*RadRate(Z,line);
  }
  else if (line>=L2Q1_LINE && line<=L2M1_LINE) {
    //L2 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel(Z,E, PK);
    return (FluorYield(Z, L2_SHELL)*RadRate(Z,line))*
		PL2_full_cascade_kissel(Z, E, PK, PL1);
  }
  else if (line>=L3Q1_LINE && line<=L3M1_LINE) {
    //L3 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel(Z, E, PK);
    PL2 = PL2_full_cascade_kissel(Z, E, PK, PL1);
    return (FluorYield(Z, L3_SHELL)*RadRate(Z,line))*PL3_full_cascade_kissel(Z, E, PK, PL1, PL2);
  }
  else if (line == LA_LINE) {
    return (CS_FluorLine_Kissel_Cascade(Z,L3M4_LINE,E)+CS_FluorLine_Kissel_Cascade(Z,L3M5_LINE,E)); 
  }
  else if (line == LB_LINE) {
    return (CS_FluorLine_Kissel_Cascade(Z,L2M4_LINE,E)+
    	CS_FluorLine_Kissel_Cascade(Z,L2M3_LINE,E)+
        CS_FluorLine_Kissel_Cascade(Z,L3N5_LINE,E)+
        CS_FluorLine_Kissel_Cascade(Z,L3O4_LINE,E)+
	CS_FluorLine_Kissel_Cascade(Z,L3O5_LINE,E)+
	CS_FluorLine_Kissel_Cascade(Z,L3O45_LINE,E)+
	CS_FluorLine_Kissel_Cascade(Z,L3N1_LINE,E)+
	CS_FluorLine_Kissel_Cascade(Z,L3O1_LINE,E)+
	CS_FluorLine_Kissel_Cascade(Z,L3N6_LINE,E)+
	CS_FluorLine_Kissel_Cascade(Z,L3N7_LINE,E)+
	CS_FluorLine_Kissel_Cascade(Z,L3N4_LINE,E)+
	CS_FluorLine_Kissel_Cascade(Z,L1M3_LINE,E)+
	CS_FluorLine_Kissel_Cascade(Z,L1M2_LINE,E)+
	CS_FluorLine_Kissel_Cascade(Z,L1M5_LINE,E)+
	CS_FluorLine_Kissel_Cascade(Z,L1M4_LINE,E)
    );
  }
  else if (line>=M1P5_LINE && line<=M1N1_LINE) {
    //M1 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel(Z, E, PK);
    PL2 = PL2_full_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_full_cascade_kissel(Z, E, PK, PL1, PL2);
    return PM1_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3)*FluorYield(Z, M1_SHELL)*RadRate(Z,line);
  }
  else if (line>=M2P5_LINE && line<=M2N1_LINE) {
    //M2 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel(Z, E, PK);
    PL2 = PL2_full_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_full_cascade_kissel(Z, E, PK, PL1, PL2);
    PM1 = PM1_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    return (FluorYield(Z, M2_SHELL)*RadRate(Z,line))*
		PM2_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
  }
  else if (line>=M3Q1_LINE && line<=M3N1_LINE) {
    //M3 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel(Z, E, PK);
    PL2 = PL2_full_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_full_cascade_kissel(Z, E, PK, PL1, PL2);
    PM1 = PM1_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
    return (FluorYield(Z, M3_SHELL)*RadRate(Z,line))*
		PM3_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
  }
  else if (line>=M4P5_LINE && line<=M4N1_LINE) {
    //M4 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel(Z, E, PK);
    PL2 = PL2_full_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_full_cascade_kissel(Z, E, PK, PL1, PL2);
    PM1 = PM1_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
    PM3 = PM3_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    return (FluorYield(Z, M4_SHELL)*RadRate(Z,line))*
		PM4_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
  }
  else if (line>=M5P5_LINE && line<=M5N1_LINE) {
    //M5 lines
    PK = CS_Photo_Partial(Z, K_SHELL, E);
    PL1 = PL1_full_cascade_kissel(Z, E, PK);
    PL2 = PL2_full_cascade_kissel(Z, E, PK, PL1);
    PL3 = PL3_full_cascade_kissel(Z, E, PK, PL1, PL2);
    PM1 = PM1_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3);
    PM2 = PM2_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1);
    PM3 = PM3_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2);
    PM4 = PM4_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3);
    return (FluorYield(Z, M5_SHELL)*RadRate(Z,line))*
		PM5_full_cascade_kissel(Z, E, PK, PL1, PL2, PL3, PM1, PM2, PM3, PM4);
  }
  else {
    ErrorExit("Line not allowed in function CS_FluorLine_Kissel_Cascade");
    return 0.0;
  }  
}

float CS_FluorLine_Kissel(int Z, int line, float E) {
	return CS_FluorLine_Kissel_Cascade(Z, line, E);
}

float CSb_FluorLine_Kissel(int Z, int line, float E) {
  return CS_FluorLine_Kissel_Cascade(Z, line, E)*AtomicWeight_arr[Z]/AVOGNUM;
}

float CSb_FluorLine_Kissel_Cascade(int Z, int line, float E) {
  return CS_FluorLine_Kissel_Cascade(Z, line, E)*AtomicWeight_arr[Z]/AVOGNUM;
}

float CSb_FluorLine_Kissel_Nonradiative_Cascade(int Z, int line, float E) {
  return CS_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E)*AtomicWeight_arr[Z]/AVOGNUM;
}

float CSb_FluorLine_Kissel_Radiative_Cascade(int Z, int line, float E) {
  return CS_FluorLine_Kissel_Radiative_Cascade(Z, line, E)*AtomicWeight_arr[Z]/AVOGNUM;
}

float CSb_FluorLine_Kissel_no_Cascade(int Z, int line, float E) {
  return CS_FluorLine_Kissel_no_Cascade(Z, line, E)*AtomicWeight_arr[Z]/AVOGNUM;
}
