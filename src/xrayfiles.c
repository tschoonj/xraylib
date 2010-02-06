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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xrayglob.h"

#define OUTD -9999
/*
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
*/
void ArrayInit(void);

void XRayInit(void)
{
  int ex;
  FILE *fp;
  char file_name[MAXFILENAMESIZE];
  char shell_name[5], line_name[5], trans_name[5];
  char *path;
  int Z, iE;
  int shell, line, trans;
  float E, prob;
  char buffer[1024];

  HardExit = 1;
  ExitStatus = 0;

  if ((path = getenv("XRAYLIB_DIR")) == NULL) {
    if ((path = getenv("HOME")) == NULL) {
      ErrorExit("Environmetal variables XRAYLIB_DIR and HOME not defined");
      return;
    }
    strcpy(XRayLibDir, path);
    strcat(XRayLibDir, "/.xraylib/data/");
  }
  else {
    strcpy(XRayLibDir, path);
    strcat(XRayLibDir, "/data/");
  }

  ArrayInit();

//  fprintf(stdout,"Initializing XRL\n");

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "atomicweight.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File atomicweight.dat not found");
    return;
  }
  while ( !feof(fp) ) {
    ex=fscanf(fp,"%d", &Z);
    if (ex != 1) break;
    fscanf(fp, "%f", &AtomicWeight_arr[Z]);
    // printf("%d\t%f\n", Z, AtomicWeight_arr[Z]);
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "CS_Photo.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File CS_Photo.dat not found");
    return;
  }
  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &NE_Photo[Z]);
    // printf("%d\n", NE_Photo[Z]); 
    if (ex != 1) break;
    E_Photo_arr[Z] = (float*)malloc(NE_Photo[Z]*sizeof(float));
    CS_Photo_arr[Z] = (float*)malloc(NE_Photo[Z]*sizeof(float));
    CS_Photo_arr2[Z] = (float*)malloc(NE_Photo[Z]*sizeof(float));
    for (iE=0; iE<NE_Photo[Z]; iE++) {
      fscanf(fp, "%f%f%f", &E_Photo_arr[Z][iE], &CS_Photo_arr[Z][iE],
	     &CS_Photo_arr2[Z][iE]);
      // printf("%e\t%e\t%e\n", E_Photo_arr[Z][iE], CS_Photo_arr[Z][iE],
      //     CS_Photo_arr2[Z][iE]);
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "CS_Rayl.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File CS_Rayl.dat not found");
    return;
  }
  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &NE_Rayl[Z]);
    // printf("%d\n", NE_Rayl[Z]); 
    if (ex != 1) break;
    E_Rayl_arr[Z] = (float*)malloc(NE_Rayl[Z]*sizeof(float));
    CS_Rayl_arr[Z] = (float*)malloc(NE_Rayl[Z]*sizeof(float));
    CS_Rayl_arr2[Z] = (float*)malloc(NE_Rayl[Z]*sizeof(float));
    for (iE=0; iE<NE_Rayl[Z]; iE++) {
      fscanf(fp, "%f%f%f", &E_Rayl_arr[Z][iE], &CS_Rayl_arr[Z][iE],
	     &CS_Rayl_arr2[Z][iE]);
      // printf("%e\t%e\t%e\n", E_Rayl_arr[Z][iE], CS_Rayl_arr[Z][iE],
      //     CS_Rayl_arr2[Z][iE]);
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "CS_Compt.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File CS_Compt.dat not found");
    return;
  }

  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &NE_Compt[Z]);
    // printf("%d\n", NE_Compt[Z]); 
    if (ex != 1) break;
    E_Compt_arr[Z] = (float*)malloc(NE_Compt[Z]*sizeof(float));
    CS_Compt_arr[Z] = (float*)malloc(NE_Compt[Z]*sizeof(float));
    CS_Compt_arr2[Z] = (float*)malloc(NE_Compt[Z]*sizeof(float));
    for (iE=0; iE<NE_Compt[Z]; iE++) {
      fscanf(fp, "%f%f%f", &E_Compt_arr[Z][iE], &CS_Compt_arr[Z][iE],
	     &CS_Compt_arr2[Z][iE]);
      // printf("%e\t%e\t%e\n", E_Compt_arr[Z][iE], CS_Compt_arr[Z][iE],
      //     CS_Compt_arr2[Z][iE]);
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "FF.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File FF.dat not found");
    return;
  }
  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &Nq_Rayl[Z]);
    // printf("%d\n", Nq_Rayl[Z]); 
    if (ex != 1) break;
    q_Rayl_arr[Z] = (float*)malloc(Nq_Rayl[Z]*sizeof(float));
    FF_Rayl_arr[Z] = (float*)malloc(Nq_Rayl[Z]*sizeof(float));
    FF_Rayl_arr2[Z] = (float*)malloc(Nq_Rayl[Z]*sizeof(float));
    for (iE=0; iE<Nq_Rayl[Z]; iE++) {
      fscanf(fp, "%f%f%f", &q_Rayl_arr[Z][iE], &FF_Rayl_arr[Z][iE],
	     &FF_Rayl_arr2[Z][iE]);
      // printf("%e\t%e\t%e\n", q_Rayl_arr[Z][iE], FF_Rayl_arr[Z][iE],
      //     FF_Rayl_arr2[Z][iE]);
    }
  }
  fclose(fp);
  
  strcpy(file_name, XRayLibDir);
  strcat(file_name, "SF.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File SF.dat not found");
    return;
  }
  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &Nq_Compt[Z]);
    // printf("%d\n", Nq_Compt[Z]); 
    if (ex != 1) break;
    q_Compt_arr[Z] = (float*)malloc(Nq_Compt[Z]*sizeof(float));
    SF_Compt_arr[Z] = (float*)malloc(Nq_Compt[Z]*sizeof(float));
    SF_Compt_arr2[Z] = (float*)malloc(Nq_Compt[Z]*sizeof(float));
    for (iE=0; iE<Nq_Compt[Z]; iE++) {
      fscanf(fp, "%f%f%f", &q_Compt_arr[Z][iE], &SF_Compt_arr[Z][iE],
	     &SF_Compt_arr2[Z][iE]);
      // printf("%e\t%e\t%e\n", q_Compt_arr[Z][iE], SF_Compt_arr[Z][iE],
      //     SF_Compt_arr2[Z][iE]);
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "edges.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File edges.dat not found");
    return;
  }
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d", &Z);
    if (ex != 1) break;
    fscanf(fp,"%s", shell_name);
    fscanf(fp,"%f", &E);  
    E /= 1000.0;
    for (shell=0; shell<SHELLNUM; shell++) {
      if (strcmp(shell_name, ShellName[shell]) == 0) {
	EdgeEnergy_arr[Z][shell] = E;
	// printf("%d\t%s\t%e\n", Z, ShellName[shell], E);
	break;
      } 
    }
  }
  fclose(fp);

  int read_error=0;
  strcpy(file_name, XRayLibDir);
  strcat(file_name, "fluor_lines.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File fluor_lines.dat not found");
    return;
  }
  HardExit=0;
  char **error_lines=NULL;
  int nerror_lines=0;
  int i;
  int found_error_line;
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d", &Z);
    if (ex != 1) break;
    fscanf(fp,"%s", line_name);
    fscanf(fp,"%f", &E);  
    E /= 1000.0;
    read_error=1;
    for (line=0; line<LINENUM; line++) {
      if (strcmp(line_name, LineName[line]) == 0) {
	LineEnergy_arr[Z][line] = E;
	//printf("%d\t%s\t%e\n", Z, LineName[line], E);
        read_error=0;
	break;
      } 
    }
    found_error_line=0;
    if (read_error) {
        if (nerror_lines == 0) {
	    	sprintf(buffer,"%s is not present in the linenames database: adjust lines.h and xrayvars.c/h\n",line_name);
		ErrorExit(buffer);
		error_lines = (char **) malloc(sizeof(char *) * ++nerror_lines);
		error_lines[0] = strdup(line_name);
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
	    		sprintf(buffer,"%s is not present in the linenames database: adjust lines.h and xrayvars.c/h\n",line_name);
			ErrorExit(buffer);
			error_lines= (char **) realloc((char **) error_lines,sizeof(char *)*++nerror_lines);
			error_lines[nerror_lines-1] = strdup(line_name);
		}
	}
    }
  }
  fclose(fp);
  HardExit=1;
  if (nerror_lines > 0) {
    sprintf(buffer,"Exiting due to too many errors\n");
    ErrorExit(buffer);
  }
  strcpy(file_name, XRayLibDir);
  strcat(file_name, "fluor_yield.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File fluor_yield.dat not found");
    return;
  }
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d", &Z);
    if (ex != 1) break;
    fscanf(fp,"%s", shell_name);
    fscanf(fp,"%f", &prob);  
    for (shell=0; shell<SHELLNUM; shell++) {
      if (strcmp(shell_name, ShellName[shell]) == 0) {
	FluorYield_arr[Z][shell] = prob;
	// printf("%d\t%s\t%e\n", Z, ShellName[shell], prob);
	break;
      } 
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "jump.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File jump.dat not found");
    return;
  }
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d", &Z);
    if (ex != 1) break;
    fscanf(fp,"%s", shell_name);
    fscanf(fp,"%f", &prob);  
    for (shell=0; shell<SHELLNUM; shell++) {
      if (strcmp(shell_name, ShellName[shell]) == 0) {
	JumpFactor_arr[Z][shell] = prob;
	// printf("%d\t%s\t%e\n", Z, ShellName[shell], prob);
	break;
      } 
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "coskron.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File coskron.dat not found");
    return;
  }
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d", &Z);
    if (ex != 1) break;
    fscanf(fp,"%s", trans_name);
    fscanf(fp,"%f", &prob);  
    for (trans=0; trans<TRANSNUM; trans++) {
      if (strcmp(trans_name, TransName[trans]) == 0) {
	CosKron_arr[Z][trans] = prob;
	// printf("%d\t%s\t%e\n", Z, TransName[trans], prob);
	break;
      } 
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "radrate.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File radrate.dat not found");
    return;
  }
  HardExit=0;
  while ( !feof(fp) ) {
    ex = fscanf(fp,"%d", &Z);
    if (ex != 1) break;
    fscanf(fp,"%s", line_name);
    fscanf(fp,"%f", &prob);
    read_error=1;
    for (line=0; line<LINENUM; line++) {
      if (strcmp(line_name, LineName[line]) == 0) {
	RadRate_arr[Z][line] = prob;
	// printf("%d\t%s\t%e\n", Z, LineName[line], prob);
	read_error=0;
	break;
      } 
    }
    found_error_line=0;
    if (read_error) {
        if (nerror_lines == 0) {
	    	sprintf(buffer,"%s is not present in the linenames database: adjust lines.h and xrayvars.c/h\n",line_name);
		ErrorExit(buffer);
		error_lines = (char **) malloc(sizeof(char *) * ++nerror_lines);
		error_lines[0] = strdup(line_name);
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
	    		sprintf(buffer,"%s is not present in the linenames database: adjust lines.h and xrayvars.c/h\n",line_name);
			ErrorExit(buffer);
			error_lines= (char **) realloc((char **) error_lines,sizeof(char *)*++nerror_lines);
			error_lines[nerror_lines-1] = strdup(line_name);
		}
	}
    }
  }
  fclose(fp);
  HardExit=1;
  if (nerror_lines > 0) {
    sprintf(buffer,"Exiting due to too many errors\n");
    ErrorExit(buffer);
  }

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "fi.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File fi.dat not found");
    return;
  }

  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &NE_Fi[Z]);
   // printf("%d\n", NE_Fi[Z]); 
    if (ex != 1) break;
    E_Fi_arr[Z] = (float*)malloc(NE_Fi[Z]*sizeof(float));
    Fi_arr[Z] = (float*)malloc(NE_Fi[Z]*sizeof(float));
    Fi_arr2[Z] = (float*)malloc(NE_Fi[Z]*sizeof(float));
    for (iE=0; iE<NE_Fi[Z]; iE++) {
      fscanf(fp, "%f%f%f", &E_Fi_arr[Z][iE], &Fi_arr[Z][iE],
	     &Fi_arr2[Z][iE]);
      // printf("%e\t%e\t%e\n", E_Fi_arr[Z][iE], Fi_arr[Z][iE],
      //     Fi_arr2[Z][iE]);
    }
  }
  fclose(fp);

  strcpy(file_name, XRayLibDir);
  strcat(file_name, "fii.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File fii.dat not found");
    return;
  }

  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%d", &NE_Fii[Z]);
    // printf("%d\n", NE_Fii[Z]); 
    if (ex != 1) break;
    E_Fii_arr[Z] = (float*)malloc(NE_Fii[Z]*sizeof(float));
    Fii_arr[Z] = (float*)malloc(NE_Fii[Z]*sizeof(float));
    Fii_arr2[Z] = (float*)malloc(NE_Fii[Z]*sizeof(float));
    for (iE=0; iE<NE_Fii[Z]; iE++) {
      fscanf(fp, "%f%f%f", &E_Fii_arr[Z][iE], &Fii_arr[Z][iE],
	     &Fii_arr2[Z][iE]);
      // printf("%e\t%e\t%e\n", E_Fii_arr[Z][iE], Fii_arr[Z][iE],
      //     Fii_arr2[Z][iE]);
    }
  }
  fclose(fp);

  //read kissel data

  strcpy(file_name, XRayLibDir);
  strcat(file_name,"kissel_pe.dat");
  if ((fp = fopen(file_name,"r")) == NULL) {
    ErrorExit("File kissel_pe.dat not found");
    return;
  }

  for (Z=1; Z<=ZMAX; Z++) {
    ex = fscanf(fp, "%i" , &NE_Photo_Total_Kissel[Z]);
    if (ex != 1) break;
    //read the total PE cross sections
    E_Photo_Total_Kissel[Z] = (double*)malloc(NE_Photo_Total_Kissel[Z]*sizeof(double));
    Photo_Total_Kissel[Z] = (double*)malloc(NE_Photo_Total_Kissel[Z]*sizeof(double));
    Photo_Total_Kissel2[Z] = (double*)malloc(NE_Photo_Total_Kissel[Z]*sizeof(double));
    for (iE=0; iE<NE_Photo_Total_Kissel[Z]; iE++) {
      fscanf(fp,"%lf%lf%lf",&E_Photo_Total_Kissel[Z][iE],&Photo_Total_Kissel[Z][iE],&Photo_Total_Kissel2[Z][iE]);
    }
    //read the electronic configuration
    for (shell = 0 ; shell < SHELLNUM_K ; shell++) {
      fscanf(fp,"%f",&Electron_Config_Kissel[Z][shell]);
    }
    //read the partial PE cross sections
    for (shell = 0 ; shell < SHELLNUM_K ; shell++) {
      fscanf(fp,"%i",&NE_Photo_Partial_Kissel[Z][shell]);
      if (NE_Photo_Partial_Kissel[Z][shell] == 0 ) continue;
      E_Photo_Partial_Kissel[Z][shell]=(double*)malloc(NE_Photo_Partial_Kissel[Z][shell]*sizeof(double));
      Photo_Partial_Kissel[Z][shell]=(double*)malloc(NE_Photo_Partial_Kissel[Z][shell]*sizeof(double));
      Photo_Partial_Kissel2[Z][shell]=(double*)malloc(NE_Photo_Partial_Kissel[Z][shell]*sizeof(double));
      for (iE=0; iE<NE_Photo_Partial_Kissel[Z][shell]; iE++) {
        fscanf(fp,"%lf%lf%lf",&E_Photo_Partial_Kissel[Z][shell][iE],&Photo_Partial_Kissel[Z][shell][iE],&Photo_Partial_Kissel2[Z][shell][iE]);
      }
    }

  }
}

void ArrayInit()
{
  int Z, shell, line, trans;

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
    for (shell=0; shell<SHELLNUM; shell++) {
      EdgeEnergy_arr[Z][shell] = OUTD;
      FluorYield_arr[Z][shell] = OUTD;
      JumpFactor_arr[Z][shell] = OUTD;
    }
    for (shell=0; shell<SHELLNUM_K; shell++) {
      Electron_Config_Kissel[Z][shell] = OUTD;
      NE_Photo_Partial_Kissel[Z][shell] = OUTD;
    }	
    for (line=0; line<LINENUM; line++) {
      LineEnergy_arr[Z][line] = OUTD;
      RadRate_arr[Z][line] = OUTD;
    }
    for (trans=0; trans<TRANSNUM; trans++) {
      CosKron_arr[Z][trans] = OUTD;
    }
  }
}









