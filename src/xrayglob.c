/*
Copyright (c) 2009, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "xraylib-defs.h"
#include "xrayglob.h"

/*////////////////////////////////////////////////////////////////////
/////            Variables                                       /////
/////////////////////////////////////////////////////////////////// */

struct MendelElement MendelArray[MENDEL_MAX] = {
	{1,"H"},{2,"He"},{3,"Li"},{4,"Be"},{5,"B"},{6,"C"},{7,"N"},{8,"O"},{9,"F"},{10,"Ne"},
	{11,"Na"},{12,"Mg"},{13,"Al"},{14,"Si"},{15,"P"},{16,"S"},{17,"Cl"},{18,"Ar"},{19,"K"},{20,"Ca"},
	{21,"Sc"},{22,"Ti"},{23,"V"},{24,"Cr"},{25,"Mn"},{26,"Fe"},{27,"Co"},{28,"Ni"},{29,"Cu"},{30,"Zn"},
	{31,"Ga"},{32,"Ge"},{33,"As"},{34,"Se"},{35,"Br"},{36,"Kr"},{37,"Rb"},{38,"Sr"},{39,"Y"},{40,"Zr"},
	{41,"Nb"},{42,"Mo"},{43,"Tc"},{44,"Ru"},{45,"Rh"},{46,"Pd"},{47,"Ag"},{48,"Cd"},{49,"In"},{50,"Sn"},
	{51,"Sb"},{52,"Te"},{53,"I"},{54,"Xe"},{55,"Cs"},{56,"Ba"},{57,"La"},{58,"Ce"},{59,"Pr"},{60,"Nd"},
	{61,"Pm"},{62,"Sm"},{63,"Eu"},{64,"Gd"},{65,"Tb"},{66,"Dy"},{67,"Ho"},{68,"Er"},{69,"Tm"},{70,"Yb"},
	{71,"Lu"},{72,"Hf"},{73,"Ta"},{74,"W"},{75,"Re"},{76,"Os"},{77,"Ir"},{78,"Pt"},{79,"Au"},{80,"Hg"},
	{81,"Tl"},{82,"Pb"},{83,"Bi"},{84,"Po"},{85,"At"},{86,"Rn"},{87,"Fr"},{88,"Ra"},{89,"Ac"},{90,"Th"},
	{91,"Pa"},{92,"U"},{93,"Np"},{94,"Pu"},{95,"Am"},{96,"Cm"},{97,"Bk"},{98,"Cf"},{99,"Es"},{100,"Fm"},
	{101,"Md"},{102,"No"},{103,"Lr"},{104,"Rf"},{105,"Db"},{106,"Sg"},{107,"Bh"}
	};

struct MendelElement MendelArraySorted[MENDEL_MAX];

Crystal_Array Crystal_arr;

double AtomicWeight_arr[ZMAX+1];
double EdgeEnergy_arr[ZMAX+1][SHELLNUM];
double LineEnergy_arr[ZMAX+1][LINENUM];
double FluorYield_arr[ZMAX+1][SHELLNUM];
double JumpFactor_arr[ZMAX+1][SHELLNUM];
double CosKron_arr[ZMAX+1][TRANSNUM];
double RadRate_arr[ZMAX+1][LINENUM];
double AtomicLevelWidth_arr[ZMAX+1][SHELLNUM];

int NE_Photo[ZMAX+1];
double *E_Photo_arr[ZMAX+1];
double *CS_Photo_arr[ZMAX+1];
double *CS_Photo_arr2[ZMAX+1];

int NE_Rayl[ZMAX+1];
double *E_Rayl_arr[ZMAX+1];
double *CS_Rayl_arr[ZMAX+1];
double *CS_Rayl_arr2[ZMAX+1];

int NE_Compt[ZMAX+1];
double *E_Compt_arr[ZMAX+1];
double *CS_Compt_arr[ZMAX+1];
double *CS_Compt_arr2[ZMAX+1];

int Nq_Rayl[ZMAX+1];
double *q_Rayl_arr[ZMAX+1];
double *FF_Rayl_arr[ZMAX+1];
double *FF_Rayl_arr2[ZMAX+1];

int Nq_Compt[ZMAX+1];
double *q_Compt_arr[ZMAX+1];
double *SF_Compt_arr[ZMAX+1];
double *SF_Compt_arr2[ZMAX+1];

int NE_Energy[ZMAX+1];
double *E_Energy_arr[ZMAX+1];
double *CS_Energy_arr[ZMAX+1];
double *CS_Energy_arr2[ZMAX+1];


int NE_Fi[ZMAX+1];
double *E_Fi_arr[ZMAX+1];
double *Fi_arr[ZMAX+1];
double *Fi_arr2[ZMAX+1];

int NE_Fii[ZMAX+1];
double *E_Fii_arr[ZMAX+1];
double *Fii_arr[ZMAX+1];
double *Fii_arr2[ZMAX+1];

int NE_Photo_Total_Kissel[ZMAX+1];
double *E_Photo_Total_Kissel[ZMAX+1];
double *Photo_Total_Kissel[ZMAX+1];
double *Photo_Total_Kissel2[ZMAX+1];

double Electron_Config_Kissel[ZMAX+1][SHELLNUM_K];
double EdgeEnergy_Kissel[ZMAX+1][SHELLNUM_K];

int NE_Photo_Partial_Kissel[ZMAX+1][SHELLNUM_K];
double *E_Photo_Partial_Kissel[ZMAX+1][SHELLNUM_K];
double *Photo_Partial_Kissel[ZMAX+1][SHELLNUM_K];
double *Photo_Partial_Kissel2[ZMAX+1][SHELLNUM_K];

int NShells_ComptonProfiles[ZMAX+1];
int Npz_ComptonProfiles[ZMAX+1];
double *UOCCUP_ComptonProfiles[ZMAX+1];
double *pz_ComptonProfiles[ZMAX+1];
double *Total_ComptonProfiles[ZMAX+1];
double *Total_ComptonProfiles2[ZMAX+1];
double *Partial_ComptonProfiles[ZMAX+1][SHELLNUM_C];
double *Partial_ComptonProfiles2[ZMAX+1][SHELLNUM_C];

double Auger_Rates[ZMAX+1][AUGERNUM];
double Auger_Yields[ZMAX+1][SHELLNUM_A];

double ElementDensity_arr[ZMAX+1];


