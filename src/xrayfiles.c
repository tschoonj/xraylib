/*
Copyright (c) 2009-2018 Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans, Teemu Ikonen, and David Sagan
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THERE BE ANY LIABILIBY FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "config.h"
#include "xraylib-aux.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include "xrayglob.h"
#include "xraylib.h"
#include "xraylib-crystal-diffraction.h"

#define OUTD -9999

double Auger_Transition_Total[ZMAX+1][SHELLNUM_A];
double Auger_Transition_Individual[ZMAX+1][AUGERNUM];

void ArrayInit(void);

void XRayInit(void) {

}

void XRayInitFromPath(char *path)
{

  char XRayLibDir[MAXFILENAMESIZE];
  FILE *fp;
  char file_name[MAXFILENAMESIZE];
  char shell_name[25], line_name[25], trans_name[5], auger_name[10];
  int Z, iE;
  int i, ex, stat;
  int shell, line, trans, auger;
  double E, prob;
  char **error_lines=NULL;
  int nerror_lines=0;
  int found_error_line;
  int read_error=0;
  int NZ;

  /* Setup the Mendel table and the sorted Mendel table. */


  for (i = 0 ; i < MENDEL_MAX ; i++) {
		MendelArraySorted[i].name = xrl_strdup(MendelArray[i].name); 
		MendelArraySorted[i].Zatom = MendelArray[i].Zatom; 
	}

	qsort(MendelArraySorted, MENDEL_MAX, sizeof(struct MendelElement), compareMendelElements);

  /* Define XRayLibDir */

  if (path == NULL) {
      fprintf(stderr, "path cannot be NULL!");
      exit(1);
  }
  else {
    strcpy(XRayLibDir, path);
#ifdef _WIN32
    strcat(XRayLibDir, "\\data\\");
#else
    strcat(XRayLibDir, "/data/");
 #endif
  }

  ArrayInit();

  /*-------------------------------------------------------------------------- */
  /*
   * Parse atomicweight.dat
   */

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "atomicweight.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File atomicweight.dat not found");
    exit(1);
  }

  while ( !feof(fp) ) {
    double weight = 0.0;
    ex=fscanf(fp,"%d %lf", &Z, &weight);
    if (ex != 2) break;
    AtomicWeight_arr[Z] = weight;
    /*printf("%d\t%lf\n", Z, AtomicWeight_arr[Z]);*/
  }
  fclose(fp);

  /*-------------------------------------------------------------------------- */
  /*
   * Parse densities.dat
   */

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "densities.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File densities.dat not found");
    exit(1);
  }

  while ( !feof(fp) ) {
    double density = 0.0;
    ex=fscanf(fp,"%d %lf", &Z, &density);
    if (ex != 2) break;
    ElementDensity_arr[Z] = density;
    /* printf("%d\t%lf\n", Z, ElementDensity_arr[Z]);*/
  }
  fclose(fp);

  /*-------------------------------------------------------------------------- */
  /*
   * Parse Crystals.dat
   */

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "Crystals.dat");

  Crystal_arr.crystal = malloc(sizeof(Crystal_Struct) * CRYSTALARRAY_MAX);
  stat = Crystal_ReadFile(file_name, NULL, NULL);
  if (stat == 0) {
    fprintf(stderr, "Could not read Crystals.dat");
    exit(1);
  } 


  strcpy(file_name, XRayLibDir);
  strcat(file_name, "CS_Photo.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File CS_Photo.dat not found");
    exit(1);
  }
  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &NE_Photo[Z]);
    /* printf("%d\n", NE_Photo[Z]); */
    if (ex != 1) break;
    E_Photo_arr[Z] = malloc(NE_Photo[Z]*sizeof(double));
    CS_Photo_arr[Z] = malloc(NE_Photo[Z]*sizeof(double));
    CS_Photo_arr2[Z] = malloc(NE_Photo[Z]*sizeof(double));
    for (iE=0; iE<NE_Photo[Z]; iE++) {
      assert(fscanf(fp, "%lf%lf%lf", &E_Photo_arr[Z][iE], &CS_Photo_arr[Z][iE],
	     &CS_Photo_arr2[Z][iE]) == 3);
      /* printf("%e\t%e\t%e\n", E_Photo_arr[Z][iE], CS_Photo_arr[Z][iE],
           CS_Photo_arr2[Z][iE]);*/
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "CS_Rayl.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File CS_Rayl.dat not found");
    exit(1);
  }
  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &NE_Rayl[Z]);
    /* printf("%d\n", NE_Rayl[Z]); */
    if (ex != 1) break;
    E_Rayl_arr[Z] = malloc(NE_Rayl[Z]*sizeof(double));
    CS_Rayl_arr[Z] = malloc(NE_Rayl[Z]*sizeof(double));
    CS_Rayl_arr2[Z] = malloc(NE_Rayl[Z]*sizeof(double));
    for (iE=0; iE<NE_Rayl[Z]; iE++) {
      assert(fscanf(fp, "%lf%lf%lf", &E_Rayl_arr[Z][iE], &CS_Rayl_arr[Z][iE],
	     &CS_Rayl_arr2[Z][iE]) == 3);
      /* printf("%e\t%e\t%e\n", E_Rayl_arr[Z][iE], CS_Rayl_arr[Z][iE],
           CS_Rayl_arr2[Z][iE]);*/
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "CS_Compt.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File CS_Compt.dat not found");
    exit(1);
  }

  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &NE_Compt[Z]);
    /* printf("%d\n", NE_Compt[Z]); */
    if (ex != 1) break;
    E_Compt_arr[Z] = malloc(NE_Compt[Z]*sizeof(double));
    CS_Compt_arr[Z] = malloc(NE_Compt[Z]*sizeof(double));
    CS_Compt_arr2[Z] = malloc(NE_Compt[Z]*sizeof(double));
    for (iE=0; iE<NE_Compt[Z]; iE++) {
      assert(fscanf(fp, "%lf%lf%lf", &E_Compt_arr[Z][iE], &CS_Compt_arr[Z][iE],
	     &CS_Compt_arr2[Z][iE]) == 3);
      /* printf("%e\t%e\t%e\n", E_Compt_arr[Z][iE], CS_Compt_arr[Z][iE],
           CS_Compt_arr2[Z][iE]);*/
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "FF.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File FF.dat not found");
    exit(1);
  }
  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &Nq_Rayl[Z]);
    /* printf("%d\n", Nq_Rayl[Z]);*/
    if (ex != 1) break;
    q_Rayl_arr[Z] = malloc(Nq_Rayl[Z]*sizeof(double));
    FF_Rayl_arr[Z] = malloc(Nq_Rayl[Z]*sizeof(double));
    FF_Rayl_arr2[Z] = malloc(Nq_Rayl[Z]*sizeof(double));
    for (iE=0; iE<Nq_Rayl[Z]; iE++) {
      assert(fscanf(fp, "%lf%lf%lf", &q_Rayl_arr[Z][iE], &FF_Rayl_arr[Z][iE],
	     &FF_Rayl_arr2[Z][iE]) == 3);
      /* printf("%e\t%e\t%e\n", q_Rayl_arr[Z][iE], FF_Rayl_arr[Z][iE],
           FF_Rayl_arr2[Z][iE]);*/
    }
  }
  fclose(fp);
  
  strcpy(file_name, XRayLibDir);
  strcat(file_name, "SF.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File SF.dat not found");
    exit(1);
  }
  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &Nq_Compt[Z]);
    /* printf("%d\n", Nq_Compt[Z]); */
    if (ex != 1) break;
    q_Compt_arr[Z] = malloc(Nq_Compt[Z]*sizeof(double));
    SF_Compt_arr[Z] = malloc(Nq_Compt[Z]*sizeof(double));
    SF_Compt_arr2[Z] = malloc(Nq_Compt[Z]*sizeof(double));
    for (iE=0; iE<Nq_Compt[Z]; iE++) {
      assert(fscanf(fp, "%lf%lf%lf", &q_Compt_arr[Z][iE], &SF_Compt_arr[Z][iE],
	     &SF_Compt_arr2[Z][iE]) == 3);
      /* printf("%e\t%e\t%e\n", q_Compt_arr[Z][iE], SF_Compt_arr[Z][iE],
           SF_Compt_arr2[Z][iE]);*/
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "edges.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File edges.dat not found");
    exit(1);
  }
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d %s %lf", &Z, shell_name, &E);
    if (ex != 3) break;
    E /= 1000.0;
    for (shell=0; shell<SHELLNUM; shell++) {
      if (strcmp(shell_name, ShellName[shell]) == 0) {
	EdgeEnergy_arr[Z][shell] = E;
	/* printf("%d\t%s\t%e\n", Z, ShellName[shell], E);*/
	break;
      } 
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "fluor_lines.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File fluor_lines.dat not found");
    exit(1);
  }

  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d %s %lf", &Z, line_name, &E);
    if (ex != 3) break;
    /*printf("Z: %d -> ", Z);
    printf("x%sx", line_name);
    printf(" -> %f\n", E);*/
    if (strlen(line_name) > 5) {
    	fprintf(stderr,"line_name too long in fluor_lines.dat: %s\n",line_name);
	exit(1);
    }
    E /= 1000.0;
    read_error=1;
    for (line=0; line<LINENUM; line++) {
      if (strcmp(line_name, LineName[line]) == 0) {
	LineEnergy_arr[Z][line] = E;
        read_error=0;
	break;
      } 
    }
    if (read_error) {
        if (nerror_lines == 0) {
	    	fprintf(stderr, "%s is not present in the linenames database: adjust xraylib-lines.h and xrayvars.c/h\n",line_name);
		error_lines = malloc(sizeof(char *) * ++nerror_lines);
		error_lines[0] = xrl_strdup(line_name);
	}
	else {
		found_error_line = 0;
		for (i = 0 ; i < nerror_lines ; i++) {
			if (strcmp(line_name,error_lines[i]) == 0) {
				found_error_line = 1;
				break;
			}
		}
		if (!found_error_line) {
	    		fprintf(stderr, "%s is not present in the linenames database: adjust xraylib-lines.h and xrayvars.c/h\n",line_name);
			error_lines= (char **) realloc((char **) error_lines,sizeof(char *)*++nerror_lines);
			error_lines[nerror_lines-1] = xrl_strdup(line_name);
		}
	}
    }
  }
  fclose(fp);
  if (nerror_lines > 0) {
    fprintf(stderr,"Exiting due to too many errors\n");
    exit(1);
  }

  /*atomic level widths*/
  strcpy(file_name, XRayLibDir);
  strcat(file_name, "atomiclevelswidth.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File atomiclevelswidth.dat not found");
    exit(1);
  }
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d %s %lf", &Z, shell_name, &E);
    if (ex != 3) break;
    E /= 1000.0;
    read_error=1;
    nerror_lines=0;
    error_lines = NULL;
    for (shell=0; shell<SHELLNUM; shell++) {
      if (strcmp(shell_name, ShellName[shell]) == 0) {
	AtomicLevelWidth_arr[Z][shell] = E;
        read_error=0;
	break;
      } 
    }
    if (read_error) {
        if (nerror_lines == 0) {
	    	fprintf(stderr,"%s is not present in the shellnames database: adjust xraylib-shells.h and xrayvars.c/h\n",shell_name);
		error_lines = malloc(sizeof(char *) * ++nerror_lines);
		error_lines[0] = xrl_strdup(shell_name);
	}
	else {
		found_error_line = 0;
		for (i = 0 ; i < nerror_lines ; i++) {
			if (strcmp(shell_name,error_lines[i]) == 0) {
				found_error_line = 1;
				break;
			}
		}
		if (!found_error_line) {
	    		fprintf(stderr,"%s is not present in the shellnames database: adjust xraylib-shells.h and xrayvars.c/h\n",shell_name);
			error_lines= (char **) realloc((char **) error_lines,sizeof(char *)*++nerror_lines);
			error_lines[nerror_lines-1] = xrl_strdup(shell_name);
		}
	}
    }
  }
  fclose(fp);
  if (nerror_lines > 0) {
    fprintf(stderr,"Exiting due to too many errors\n");
    exit(1);
  }


  strcpy(file_name, XRayLibDir);
  strcat(file_name, "fluor_yield.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File fluor_yield.dat not found");
    exit(1);
  }
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d %s %lf", &Z, shell_name, &prob);
    if (ex != 3) break;
    for (shell=0; shell<SHELLNUM; shell++) {
      if (strcmp(shell_name, ShellName[shell]) == 0) {
	FluorYield_arr[Z][shell] = prob;
	break;
      } 
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "jump.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File jump.dat not found");
    exit(1);
  }
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d %s %lf", &Z, shell_name, &prob);
    if (ex != 3) break;
    for (shell=0; shell<SHELLNUM; shell++) {
      if (strcmp(shell_name, ShellName[shell]) == 0) {
	JumpFactor_arr[Z][shell] = prob;
	/* printf("%d\t%s\t%e\n", Z, ShellName[shell], prob);*/
	break;
      } 
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "coskron.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File coskron.dat not found");
    exit(1);
  }
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d %s %lf", &Z, trans_name, &prob);
    if (ex != 3) break;
    if (strlen(trans_name) > 4) {
    	fprintf(stderr,"trans_name too long in coskron: %s\n", line_name);
	exit(1);
    }
    for (trans=0; trans<TRANSNUM; trans++) {
      if (strcmp(trans_name, TransName[trans]) == 0) {
	CosKron_arr[Z][trans] = prob;
	/* printf("%d\t%s\t%e\n", Z, TransName[trans], prob);*/
	break;
      } 
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "radrate.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File radrate.dat not found");
    exit(1);
  }
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d %s %lf", &Z, line_name, &prob);
    if (ex != 3) break;
    if (strlen(line_name) > 5) {
    	fprintf(stderr,"line_name too long in radrate.dat: %s\n", line_name);
	exit(1);
    }
    read_error=1;
    for (line=0; line<LINENUM; line++) {
      if (strcmp(line_name, LineName[line]) == 0) {
	RadRate_arr[Z][line] = prob;
	/* printf("%d\t%s\t%e\n", Z, LineName[line], prob);*/
	read_error=0;
	break;
      } 
    }
    if (read_error) {
        if (nerror_lines == 0) {
	    	fprintf(stderr, "%s is not present in the linenames database: adjust xraylib-lines.h and xrayvars.c/h\n",line_name);
		error_lines = malloc(sizeof(char *) * ++nerror_lines);
		error_lines[0] = xrl_strdup(line_name);
	}
	else {
		found_error_line = 0;
		for (i = 0 ; i < nerror_lines ; i++) {
			if (strcmp(line_name,error_lines[i]) == 0) {
				found_error_line = 1;
				break;
			}
		}
		if (!found_error_line) {
	    		fprintf(stderr, "%s is not present in the linenames database: adjust xraylib-lines.h and xrayvars.c/h\n",line_name);
			error_lines= (char **) realloc((char **) error_lines,sizeof(char *)*++nerror_lines);
			error_lines[nerror_lines-1] = xrl_strdup(line_name);
		}
	}
    }
  }
  fclose(fp);
  if (nerror_lines > 0) {
    fprintf(stderr,"Exiting due to too many errors\n");
    exit(1);
  }

  /*auger non-radiative transitions*/
  strcpy(file_name, XRayLibDir);
  strcat(file_name, "auger_rates.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File auger_rates.dat not found");
    exit(1);
  }
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d %s %lf", &Z, auger_name, &prob);
    if (ex != 3) break;
    /*printf("%d %s %f\n", Z, auger_name, prob);*/
    read_error=1;
    for (shell=0; shell < SHELLNUM_A; shell++) {
      if (strcmp(auger_name, AugerNameTotal[shell]) == 0) {
	Auger_Transition_Total[Z][shell] = prob;
	/* printf("%d\t%s\t%e\n", Z, LineName[line], prob);*/
	read_error=0;
	break;
      } 
    }
    for (auger=0; auger < AUGERNUM; auger++) {
      if (strcmp(auger_name, AugerName[auger]) == 0) {
	Auger_Transition_Individual[Z][auger] = prob;
	/* printf("%d\t%s\t%e\n", Z, LineName[line], prob);*/
	read_error=0;
	break;
      } 
    }
    if (read_error) {
        if (nerror_lines == 0) {
	    	fprintf(stderr, "%s is not present in the Auger transition names database: adjust xraylib-auger.h and xrayvars.c/h\n",auger_name);
		error_lines = malloc(sizeof(char *) * ++nerror_lines);
		error_lines[0] = xrl_strdup(auger_name);
	}
	else {
		found_error_line = 0;
		for (i = 0 ; i < nerror_lines ; i++) {
			if (strcmp(auger_name,error_lines[i]) == 0) {
				found_error_line = 1;
				break;
			}
		}
		if (!found_error_line) {
	    		fprintf(stderr, "%s is not present in the Auger transition names database: adjust xraylib-auger.h and xrayvars.c/h\n",auger_name);
			error_lines= (char **) realloc((char **) error_lines,sizeof(char *)*++nerror_lines);
			error_lines[nerror_lines-1] = xrl_strdup(auger_name);
		}
	}
    }
  }

  fclose(fp);
  if (nerror_lines > 0) {
    fprintf(stderr, "Exiting due to too many errors\n");
    exit(1);
  }

  /*anomalous scattering factors */

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "fi.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File fi.dat not found");
    exit(1);
  }

  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &NE_Fi[Z]);
   /* printf("%d\n", NE_Fi[Z]); */
    if (ex != 1) break;
    E_Fi_arr[Z] = malloc(NE_Fi[Z]*sizeof(double));
    Fi_arr[Z] = malloc(NE_Fi[Z]*sizeof(double));
    Fi_arr2[Z] = malloc(NE_Fi[Z]*sizeof(double));
    for (iE=0; iE<NE_Fi[Z]; iE++) {
      assert(fscanf(fp, "%lf%lf%lf", &E_Fi_arr[Z][iE], &Fi_arr[Z][iE],
	     &Fi_arr2[Z][iE]) == 3);
      /* printf("%e\t%e\t%e\n", E_Fi_arr[Z][iE], Fi_arr[Z][iE],
           Fi_arr2[Z][iE]);*/
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "fii.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File fii.dat not found");
    exit(1);
  }

  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &NE_Fii[Z]);
    /* printf("%d\n", NE_Fii[Z]); */
    if (ex != 1) break;
    E_Fii_arr[Z] = malloc(NE_Fii[Z]*sizeof(double));
    Fii_arr[Z] = malloc(NE_Fii[Z]*sizeof(double));
    Fii_arr2[Z] = malloc(NE_Fii[Z]*sizeof(double));
    for (iE=0; iE<NE_Fii[Z]; iE++) {
      assert(fscanf(fp, "%lf%lf%lf", &E_Fii_arr[Z][iE], &Fii_arr[Z][iE],
	     &Fii_arr2[Z][iE]) == 3);
      /* printf("%e\t%e\t%e\n", E_Fii_arr[Z][iE], Fii_arr[Z][iE],
           Fii_arr2[Z][iE]);*/
    }
  }
  fclose(fp);

  /*read kissel data*/

  strcpy(file_name, XRayLibDir);
  strcat(file_name,"kissel_pe.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File kissel_pe.dat not found");
    exit(1);
  }

  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%i" , &NE_Photo_Total_Kissel[Z]);
    if (ex != 1) break;
    /*read the total PE cross sections*/
    E_Photo_Total_Kissel[Z] = malloc(NE_Photo_Total_Kissel[Z]*sizeof(double));
    Photo_Total_Kissel[Z] = malloc(NE_Photo_Total_Kissel[Z]*sizeof(double));
    Photo_Total_Kissel2[Z] = malloc(NE_Photo_Total_Kissel[Z]*sizeof(double));
    for (iE=0; iE<NE_Photo_Total_Kissel[Z]; iE++) {
      assert(fscanf(fp,"%lf%lf%lf",&E_Photo_Total_Kissel[Z][iE],&Photo_Total_Kissel[Z][iE],&Photo_Total_Kissel2[Z][iE]) == 3);
    }
    /*read the electronic configuration*/
    for (shell = 0 ; shell < SHELLNUM_K ; shell++) {
      assert(fscanf(fp,"%lf",&Electron_Config_Kissel[Z][shell]) == 1);
    }
    /*read the partial PE cross sections*/
    for (shell = 0 ; shell < SHELLNUM_K ; shell++) {
      assert(fscanf(fp,"%d",&NE_Photo_Partial_Kissel[Z][shell]) == 1);
      if (NE_Photo_Partial_Kissel[Z][shell] == 0 ) continue;
      assert(fscanf(fp,"%lf",&EdgeEnergy_Kissel[Z][shell]) == 1);
      E_Photo_Partial_Kissel[Z][shell]=malloc(NE_Photo_Partial_Kissel[Z][shell]*sizeof(double));
      Photo_Partial_Kissel[Z][shell]=malloc(NE_Photo_Partial_Kissel[Z][shell]*sizeof(double));
      Photo_Partial_Kissel2[Z][shell]=malloc(NE_Photo_Partial_Kissel[Z][shell]*sizeof(double));
      for (iE=0; iE<NE_Photo_Partial_Kissel[Z][shell]; iE++) {
        assert(fscanf(fp,"%lf%lf%lf",&E_Photo_Partial_Kissel[Z][shell][iE],&Photo_Partial_Kissel[Z][shell][iE],&Photo_Partial_Kissel2[Z][shell][iE]) == 3);
      }
    }

  }
  fclose(fp);

  /*read Compton profiles*/
  strcpy(file_name, XRayLibDir);
  strcat(file_name,"comptonprofiles.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File comptonprofiles.dat not found");
    exit(1);
  }

  for (Z = 1 ; Z <= ZMAX ; Z++) {
 	ex = fscanf(fp, "%d %d",NShells_ComptonProfiles+Z,Npz_ComptonProfiles+Z);
	if (ex != 2) break;
	/*fprintf(stdout,"%d %d\n", NShells_ComptonProfiles[Z],Npz_ComptonProfiles[Z]);*/
	UOCCUP_ComptonProfiles[Z] = malloc(NShells_ComptonProfiles[Z]*sizeof(double));
  	pz_ComptonProfiles[Z] = malloc(Npz_ComptonProfiles[Z]*sizeof(double));
	Total_ComptonProfiles[Z] = malloc(Npz_ComptonProfiles[Z]*sizeof(double));
	Total_ComptonProfiles2[Z] = malloc(Npz_ComptonProfiles[Z]*sizeof(double));
 	for (iE=0; iE < NShells_ComptonProfiles[Z] ; iE++) {
		double UOCCUP = 0.0;
		assert(fscanf(fp,"%lg", &UOCCUP) == 1);
		/*fprintf(stdout,"UOCCUP: %lf\n", UOCCUP);*/
		UOCCUP_ComptonProfiles[Z][iE] = UOCCUP;
		/*fprintf(stdout,"%lf\n", UOCCUP_ComptonProfiles[Z][iE]);*/
	} 
 	for (iE=0; iE < Npz_ComptonProfiles[Z] ; iE++) {
		assert(fscanf(fp,"%lf", &pz_ComptonProfiles[Z][iE]) == 1);
		/*fprintf(stdout,"%lf\n", pz_ComptonProfiles[Z][iE]);*/
	} 
 	for (iE=0; iE < Npz_ComptonProfiles[Z] ; iE++) {
		assert(fscanf(fp,"%lf", &Total_ComptonProfiles[Z][iE]) == 1);
		/*fprintf(stdout,"%lf\n", Total_ComptonProfiles[Z][iE]);*/
	} 
 	for (iE=0; iE < Npz_ComptonProfiles[Z] ; iE++) {
		assert(fscanf(fp,"%lf", &Total_ComptonProfiles2[Z][iE]) == 1);
		/*fprintf(stdout,"%lf\n", Total_ComptonProfiles2[Z][iE]);*/
	} 
	for (shell = 0 ; shell < NShells_ComptonProfiles[Z] ; shell++) {
		if (UOCCUP_ComptonProfiles[Z][shell] > 0.0) {
			Partial_ComptonProfiles[Z][shell] = malloc(Npz_ComptonProfiles[Z]*sizeof(double));
			for (iE = 0 ; iE < Npz_ComptonProfiles[Z] ; iE++) {
				assert(fscanf(fp, "%lf", &Partial_ComptonProfiles[Z][shell][iE]) == 1);
				/*fprintf(stdout, "%lf\n", Partial_ComptonProfiles[Z][shell][iE]);*/
			}
		}
		else
			Partial_ComptonProfiles[Z][shell] = NULL; 
	}
	for (shell = 0 ; shell < NShells_ComptonProfiles[Z] ; shell++) {
		if (UOCCUP_ComptonProfiles[Z][shell] > 0.0) {
			Partial_ComptonProfiles2[Z][shell] = malloc(Npz_ComptonProfiles[Z]*sizeof(double));
			for (iE = 0 ; iE < Npz_ComptonProfiles[Z] ; iE++) {
				assert(fscanf(fp, "%lf", &Partial_ComptonProfiles2[Z][shell][iE]) == 1);
				/*fprintf(stdout, "%lf\n", Partial_ComptonProfiles2[Z][shell][iE]);*/
			}
		}
		else
			Partial_ComptonProfiles2[Z][shell] = NULL; 
	}
  }
  fclose(fp);

  /* read mass energy-absorption coefficients */
  strcpy(file_name, XRayLibDir);
  strcat(file_name,"CS_Energy.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    fprintf(stderr, "File CS_Energy.dat not found");
    exit(1);
  }
  
  ex = fscanf(fp, "%i", &NZ);
  for (Z = 1 ; Z <= NZ ; Z++) {
    ex = fscanf(fp, "%d", &NE_Energy[Z]);
    E_Energy_arr[Z] = malloc(NE_Energy[Z]*sizeof(double));
    CS_Energy_arr[Z] = malloc(NE_Energy[Z]*sizeof(double));
    CS_Energy_arr2[Z] = malloc(NE_Energy[Z]*sizeof(double));
    for (iE=0 ; iE < NE_Energy[Z] ; iE++) {
    	assert(fscanf(fp, "%lf %lf %lf", &E_Energy_arr[Z][iE], &CS_Energy_arr[Z][iE], &CS_Energy_arr2[Z][iE]) == 3);
    }	
  }
  fclose(fp);
}

void ArrayInit()
{
  int Z, shell, line, trans, auger;

  for (Z=0; Z<=ZMAX; Z++) {
    NE_Photo[Z] = OUTD;
    NE_Rayl[Z] = OUTD;
    NE_Compt[Z] = OUTD;
    NE_Fi[Z] = OUTD;
    NE_Fii[Z] = OUTD;
    Nq_Rayl[Z] = OUTD;
    Nq_Compt[Z] = OUTD;
    AtomicWeight_arr[Z] = OUTD;
    NE_Photo_Total_Kissel[Z] = OUTD;
    NShells_ComptonProfiles[Z] = OUTD;
    Npz_ComptonProfiles[Z] = OUTD;
    ElementDensity_arr[Z] = OUTD;
    NE_Energy[Z] = OUTD;
   
    for (shell=0; shell<SHELLNUM; shell++) {
      EdgeEnergy_arr[Z][shell] = OUTD;
      FluorYield_arr[Z][shell] = OUTD;
      JumpFactor_arr[Z][shell] = OUTD;
      AtomicLevelWidth_arr[Z][shell] = OUTD;
    }
    for (shell=0; shell<SHELLNUM_K; shell++) {
      Electron_Config_Kissel[Z][shell] = OUTD;
      NE_Photo_Partial_Kissel[Z][shell] = OUTD;
    }	
    for (line=0; line<LINENUM; line++) {
      LineEnergy_arr[Z][line] = 0.0;
      RadRate_arr[Z][line] = 0.0;
    }
    for (trans=0; trans<TRANSNUM; trans++) {
      CosKron_arr[Z][trans] = 0.0;
    }
    for (shell = 0 ; shell < SHELLNUM_A ; shell++)
    	Auger_Transition_Total[Z][shell] = 0.0;
    for (auger = 0 ; auger < AUGERNUM ; auger++)
    	Auger_Transition_Individual[Z][auger] = 0.0;
  }
  memset(xrf_cross_sections_constants_full, 0, sizeof(xrf_cross_sections_constants_full));
  memset(xrf_cross_sections_constants_auger_only, 0, sizeof(xrf_cross_sections_constants_auger_only));
}









