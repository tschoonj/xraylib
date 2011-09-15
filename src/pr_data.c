/*
Copyright (c) 2009, 2010, 2011, Teemu Ikonen and Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Teemu Ikonen and Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Teemu Ikonen and Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <stdio.h>
#include "xraylib.h"
#include "xrayglob.h"

#define OUTFILE "xrayglob_inline.c"
#define FLOAT_PER_LINE 4
#define INT_PER_LINE 10
#define NAME_PER_LINE 4

void XRayInit(void);
FILE *f;

#define PR_MATF(ROWMAX, COLMAX, ARRNAME) \
for(j = 0; j < (ROWMAX); j++) { \
  print_floatvec((COLMAX), ARRNAME[j]); \
  fprintf(f, ",\n"); \
} \
fprintf(f, "};\n\n");

#define PR_MATD(ROWMAX, COLMAX, ARRNAME) \
for(j = 0; j < (ROWMAX); j++) { \
  print_doublevec((COLMAX), ARRNAME[j]); \
  fprintf(f, ",\n"); \
} \
fprintf(f, "};\n\n");

#define PR_MATI(ROWMAX, COLMAX, ARRNAME) \
for(j = 0; j < (ROWMAX); j++) { \
  print_intvec((COLMAX), ARRNAME[j]); \
  fprintf(f, ",\n"); \
} \
fprintf(f, "};\n\n");

#define PR_DYNMATF(NVAR, EVAR, ENAME) \
  for(j = 0; j < ZMAX+1; j++) { \
    fprintf(f, "static float __%s_%d[] =\n", ENAME, j);\
    print_floatvec(NVAR[j], EVAR[j]); \
    fprintf(f, ";\n\n");\
  } \
\
  fprintf(f, "float *%s[] =\n", ENAME);\
  fprintf(f, "{\n"); \
  for(i = 0; i < ZMAX+1; i++) { \
    fprintf(f, "__%s_%d, ", ENAME, i);\
    if(i%NAME_PER_LINE == (NAME_PER_LINE-1))\
      fprintf(f, "\n");\
  }\
  fprintf(f, "};\n\n");    

#define PR_DYNMATD(NVAR, EVAR, ENAME) \
  for(j = 0; j < ZMAX+1; j++) { \
    fprintf(f, "static double __%s_%d[] =\n", ENAME, j);\
    print_doublevec(NVAR[j], EVAR[j]); \
    fprintf(f, ";\n\n");\
  } \
\
  fprintf(f, "double *%s[] =\n", ENAME);\
  fprintf(f, "{\n"); \
  for(j = 0; j < ZMAX+1; j++) { \
    fprintf(f, "__%s_%d, ", ENAME, j);\
    if(j%NAME_PER_LINE == (NAME_PER_LINE-1))\
      fprintf(f, "\n");\
  }\
  fprintf(f, "};\n\n");    

#define PR_DYNMATI(NVAR, EVAR, ENAME) \
  for(j = 0; j < ZMAX+1; j++) { \
    fprintf(f, "static int __%s_%d[] =\n", ENAME, j);\
    print_intvec(NVAR[j], EVAR[j]); \
    fprintf(f, ";\n\n");\
  } \
\
  fprintf(f, "int *%s[] =\n", ENAME);\
  fprintf(f, "{\n"); \
  for(j = 0; j < ZMAX+1; j++) { \
    fprintf(f, "__%s_%d, ", ENAME, j);\
    if(j%NAME_PER_LINE == (NAME_PER_LINE-1))\
      fprintf(f, "\n");\
  }\
  fprintf(f, "};\n\n");    

#define PR_DYNMAT_3DD_K(NVAR2D, EVAR, ENAME) \
  for (i = 0; i < ZMAX+1; i++) { \
    for (j = 0; j < SHELLNUM_K; j++) {\
      fprintf(f, "static double __%s_%i_%i[] = \n", ENAME, i, j);\
      print_doublevec(NVAR2D[i][j], EVAR[i][j]);\
      fprintf(f, ";\n\n");\
    }\
  }\
\
  fprintf(f, "double *%s[ZMAX+1][SHELLNUM_K] = {\n", ENAME);\
  for (i = 0; i < ZMAX+1; i++) {\
    fprintf(f,"{\n");\
    for (j = 0; j < SHELLNUM_K; j++) {\
      fprintf(f, "__%s_%i_%i, ", ENAME,i,j);\
      if(j%NAME_PER_LINE == (NAME_PER_LINE-1))\
        fprintf(f, "\n");\
    }\
    fprintf(f,"},\n");\
  }\
  fprintf(f,"\n};\n");\

#define PR_DYNMAT_3DD_C(NVAR2D, NVAR2D2, NVAR2D3, EVAR, ENAME) \
  for (i = 0; i < 102; i++) { \
    for (j = 0; j < NShells_ComptonProfiles[i]; j++) {\
      if (UOCCUP_ComptonProfiles[i][j] > 0.0) {\
      	fprintf(f, "static double __%s_%i_%i[] = \n", ENAME, i, j);\
      	print_doublevec(NVAR2D[i], EVAR[i][j]);\
      	fprintf(f, ";\n\n");\
      }\
      else {\
      	fprintf(f, "static double __%s_%i_%i[] = {};\n", ENAME, i, j);\
      }\
    }\
  }\
  fprintf(f, "double *%s[ZMAX+1][SHELLNUM_C] = {\n", ENAME);\
  for (i = 0; i < 102; i++) {\
    fprintf(f,"{\n");\
    for (j = 0; j < NShells_ComptonProfiles[i]; j++) {\
      fprintf(f, "__%s_%i_%i, ", ENAME,i,j);\
      if(j%NAME_PER_LINE == (NAME_PER_LINE-1))\
        fprintf(f, "\n");\
    }\
    fprintf(f,"},\n");\
  }\
  fprintf(f,"\n};\n");\

#define PR_NUMVEC1D(NVAR, NNAME) \
  fprintf(f, "int %s[] =\n", NNAME); \
  print_intvec(ZMAX+1, NVAR); \
  fprintf(f, ";\n\n");

//-----------------------------------------------------

void print_mendelvec(int arrmax, struct MendelElement *arr)
{
  int i;
  int MENDEL_PER_LINE = 10;
  fprintf(f, "{\n"); 
  for(i = 0; i < arrmax; i++) {
    fprintf(f, "{%d,\"%s\"}, ", arr[i].Zatom, arr[i].name);
    if(i%MENDEL_PER_LINE == (MENDEL_PER_LINE-1))
      fprintf(f, "\n");
  }
  fprintf(f, "}");
  fprintf(f, ";\n\n");
}

void print_floatvec(int arrmax, float *arr)
{
  int i;
  fprintf(f, "{\n"); 
  for(i = 0; i < arrmax; i++) {
    fprintf(f, "%.10E, ", arr[i]);
    if(i%FLOAT_PER_LINE == (FLOAT_PER_LINE-1))
      fprintf(f, "\n");
  }
  fprintf(f, "}");
}

void print_doublevec(int arrmax, double *arr)
{
  int i;
  fprintf(f, "{\n"); 
  for(i = 0; i < arrmax; i++) {
    fprintf(f, "%.10E, ", arr[i]);
    if(i%FLOAT_PER_LINE == (FLOAT_PER_LINE-1))
      fprintf(f, "\n");
  }
  fprintf(f, "}");
}

void print_intvec(int arrmax, int *arr)
{
  int i;
  fprintf(f, "{\n"); 
  for(i = 0; i < arrmax; i++) {
    fprintf(f, "%d, ", arr[i]);
    if(i%INT_PER_LINE == (INT_PER_LINE-1))
      fprintf(f, "\n");
  }
  fprintf(f, "}");
}


int main(void) 
{

  int i,j;

  XRayInit();

  f = fopen(OUTFILE, "w");
  if(f == NULL) {
    perror("file open");
  }

  fprintf(f, "#include \"xray_defs.h\"\n\n");

  fprintf(f, "struct MendelElement MendelArray[MENDEL_MAX] = \n");
  print_mendelvec(MENDEL_MAX, MendelArray);

  fprintf(f, "struct MendelElement MendelArraySorted[MENDEL_MAX] = \n");
  print_mendelvec(MENDEL_MAX, MendelArraySorted);

  Crystal_Struct* crystal;
  struct CrystalAtom* atom;

  for (i = 0; i < Crystal_arr.n_crystal; i++) {
    crystal = &Crystal_arr.crystal[i];
    fprintf(f, "struct CrystalAtom __atoms_%s[%i] = {", crystal->name, crystal->n_atom);
    for (j = 0; j < crystal->n_atom; j++) {
      if (j % 2 == 0) fprintf(f, "\n  ");
      atom = &crystal->atom[j];
      fprintf(f, "{%i, %f, %f, %f, %f}, ", atom->Zatom, atom->fraction, atom->x, atom->y, atom->z);
    }
    fprintf (f, "\n};\n\n");
  }

  fprintf(f, "Crystal_Struct __Crystal_arr[CRYSTALARRAY_MAX] = {\n");
  for (i = 0; i < Crystal_arr.n_crystal; i++) {
    crystal = &Crystal_arr.crystal[i];
    fprintf(f, "  {\"%s\", %f, %f, %f, %f, %f, %f, %f, %i, __atoms_%s},\n", crystal->name, 
              crystal->a, crystal->b, crystal->c, crystal->alpha, crystal->beta, crystal->gamma, 
              crystal->volume, crystal->n_atom, crystal->name);
  }
  fprintf (f, "};\n\n");

  fprintf(f, "Crystal_arr = {%i, %i, __Crystal_arr};/n/n", Crystal_arr.n_crystal, Crystal_arr.n_alloc);

  fprintf(f, "float AtomicWeight_arr[ZMAX+1] =\n");
  print_floatvec(ZMAX+1, AtomicWeight_arr);
  fprintf(f, ";\n\n");

  fprintf(f, "float EdgeEnergy_arr[ZMAX+1][SHELLNUM] = {\n");
  PR_MATF(ZMAX+1, SHELLNUM, EdgeEnergy_arr);

  fprintf(f, "float AtomicLevelWidth_arr[ZMAX+1][SHELLNUM] = {\n");
  PR_MATF(ZMAX+1, SHELLNUM, AtomicLevelWidth_arr);

  fprintf(f, "float LineEnergy_arr[ZMAX+1][LINENUM] = {\n");
  PR_MATF(ZMAX+1, LINENUM, LineEnergy_arr);

  fprintf(f, "float FluorYield_arr[ZMAX+1][SHELLNUM] = {\n");
  PR_MATF(ZMAX+1, SHELLNUM, FluorYield_arr);

  fprintf(f, "float JumpFactor_arr[ZMAX+1][SHELLNUM] = {\n");
  PR_MATF(ZMAX+1, SHELLNUM, JumpFactor_arr);

  fprintf(f, "float CosKron_arr[ZMAX+1][TRANSNUM] = {\n");
  PR_MATF(ZMAX+1, TRANSNUM, CosKron_arr);

  fprintf(f, "float RadRate_arr[ZMAX+1][LINENUM] = {\n");
  PR_MATF(ZMAX+1, LINENUM, RadRate_arr);

  PR_NUMVEC1D(NE_Photo, "NE_Photo");
  PR_DYNMATF(NE_Photo, E_Photo_arr, "E_Photo_arr");
  PR_DYNMATF(NE_Photo, CS_Photo_arr, "CS_Photo_arr");
  PR_DYNMATF(NE_Photo, CS_Photo_arr2, "CS_Photo_arr2");

  PR_NUMVEC1D(NE_Rayl, "NE_Rayl");
  PR_DYNMATF(NE_Rayl, E_Rayl_arr, "E_Rayl_arr");
  PR_DYNMATF(NE_Rayl, CS_Rayl_arr, "CS_Rayl_arr");
  PR_DYNMATF(NE_Rayl, CS_Rayl_arr2, "CS_Rayl_arr2");

  PR_NUMVEC1D(NE_Compt, "NE_Compt");
  PR_DYNMATF(NE_Compt, E_Compt_arr, "E_Compt_arr");
  PR_DYNMATF(NE_Compt, CS_Compt_arr, "CS_Compt_arr");
  PR_DYNMATF(NE_Compt, CS_Compt_arr2, "CS_Compt_arr2");

  PR_NUMVEC1D(Nq_Rayl, "Nq_Rayl");
  PR_DYNMATF(Nq_Rayl, q_Rayl_arr, "q_Rayl_arr");
  PR_DYNMATF(Nq_Rayl, FF_Rayl_arr, "FF_Rayl_arr");
  PR_DYNMATF(Nq_Rayl, FF_Rayl_arr2, "FF_Rayl_arr2");

  PR_NUMVEC1D(Nq_Compt, "Nq_Compt");
  PR_DYNMATF(Nq_Compt, q_Compt_arr, "q_Compt_arr");
  PR_DYNMATF(Nq_Compt, SF_Compt_arr, "SF_Compt_arr");
  PR_DYNMATF(Nq_Compt, SF_Compt_arr2, "SF_Compt_arr2");

//added by Tom Schoonjans 26/03/2008
  PR_NUMVEC1D(NE_Fi, "NE_Fi");
  PR_DYNMATF(NE_Fi, E_Fi_arr, "E_Fi_arr");
  PR_DYNMATF(NE_Fi, Fi_arr, "Fi_arr");
  PR_DYNMATF(NE_Fi, Fi_arr2, "Fi_arr2");

  PR_NUMVEC1D(NE_Fii, "NE_Fii");
  PR_DYNMATF(NE_Fii, E_Fii_arr, "E_Fii_arr");
  PR_DYNMATF(NE_Fii, Fii_arr, "Fii_arr");
  PR_DYNMATF(NE_Fii, Fii_arr2, "Fii_arr2");

//added by Tom Schoonjans 07/04/2008
  fprintf(f, "float Electron_Config_Kissel[ZMAX+1][SHELLNUM_K] = {\n");
  PR_MATF(ZMAX+1, SHELLNUM_K, Electron_Config_Kissel);

  fprintf(f, "double EdgeEnergy_Kissel[ZMAX+1][SHELLNUM_K] = {\n");
  PR_MATD(ZMAX+1, SHELLNUM_K, EdgeEnergy_Kissel);

  PR_NUMVEC1D(NE_Photo_Total_Kissel, "NE_Photo_Total_Kissel");
  PR_DYNMATD(NE_Photo_Total_Kissel,E_Photo_Total_Kissel,"E_Photo_Total_Kissel");
  PR_DYNMATD(NE_Photo_Total_Kissel,Photo_Total_Kissel,"Photo_Total_Kissel");
  PR_DYNMATD(NE_Photo_Total_Kissel,Photo_Total_Kissel2,"Photo_Total_Kissel2");

  fprintf(f, "int NE_Photo_Partial_Kissel[ZMAX+1][SHELLNUM_K] = {\n");
  PR_MATI(ZMAX+1, SHELLNUM_K, NE_Photo_Partial_Kissel);

  PR_DYNMAT_3DD_K(NE_Photo_Partial_Kissel, E_Photo_Partial_Kissel, "E_Photo_Partial_Kissel"); 
  PR_DYNMAT_3DD_K(NE_Photo_Partial_Kissel, Photo_Partial_Kissel, "Photo_Partial_Kissel"); 
  PR_DYNMAT_3DD_K(NE_Photo_Partial_Kissel, Photo_Partial_Kissel2, "Photo_Partial_Kissel2"); 

//added by Tom Schoonjans 28/07/2010
  PR_NUMVEC1D(NShells_ComptonProfiles, "NShells_ComptonProfiles");
  PR_NUMVEC1D(Npz_ComptonProfiles, "Npz_ComptonProfiles");
  PR_DYNMATD(NShells_ComptonProfiles,UOCCUP_ComptonProfiles,"UOCCUP_ComptonProfiles");
  PR_DYNMATD(Npz_ComptonProfiles,pz_ComptonProfiles,"pz_ComptonProfiles");
  PR_DYNMATD(Npz_ComptonProfiles,Total_ComptonProfiles,"Total_ComptonProfiles");
  PR_DYNMATD(Npz_ComptonProfiles,Total_ComptonProfiles2,"Total_ComptonProfiles2");
  PR_DYNMAT_3DD_C(Npz_ComptonProfiles, NShells_ComptonProfiles, UOCCUP_ComptonProfiles, Partial_ComptonProfiles,"Partial_ComptonProfiles");
  PR_DYNMAT_3DD_C(Npz_ComptonProfiles, NShells_ComptonProfiles, UOCCUP_ComptonProfiles, Partial_ComptonProfiles2,"Partial_ComptonProfiles2");

  fprintf(f, "double Auger_Transition_Total[ZMAX+1][SHELLNUM_A] = {\n");
  PR_MATD(ZMAX+1, SHELLNUM_A, Auger_Transition_Total);
  fprintf(f, "double Auger_Transition_Individual[ZMAX+1][AUGERNUM] = {\n");
  PR_MATD(ZMAX+1, AUGERNUM, Auger_Transition_Individual);

  fclose(f);

  return 0;
}
