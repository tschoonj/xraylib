/*
Copyright (c) 2011  David Sagan
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY David Sagan ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "xraylib-crystal-diffraction.h"
#include "xrayglob.h"
#include "xraylib.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define sind(x)  sin(x * DEGRAD)
#define cosd(x)  cos(x * DEGRAD)
#define tand(x)  tan(x * DEGRAD)
#define pow2(x)  pow(x, 2)
#define FALSE 0
#define TRUE 1

/*-------------------------------------------------------------------------------------------------- */

double c_abs(xrlComplex x) {
  double ans = x.re * x.re - x.im * x.im;
  ans = sqrt(ans);
  return ans;
}

/*-------------------------------------------------------------------------------------------------- */

xrlComplex c_mul(xrlComplex x, xrlComplex y) {
  xrlComplex ans;
  ans.re = x.re * y.re - x.im * y.im;
  ans.im = x.re * y.im + x.im * y.re;
  return ans;
}

/*-------------------------------------------------------------------------------------------------- */
/* Private function to extend the crystal array size. */

static void Crystal_ExtendArray (Crystal_Array** c_array, int n_new) {
  int i;

  /* Special case */

  /* Transfer data to a temp. */

  Crystal_Array* temp_array = malloc(sizeof(Crystal_Array));
  temp_array->n_crystal = (*c_array)->n_crystal;
  temp_array->n_alloc = (*c_array)->n_alloc + n_new;
  temp_array->crystal = malloc(temp_array->n_alloc * sizeof(Crystal_Array));

  for (i = 0; i < (*c_array)->n_crystal; i++) {
    temp_array->crystal[i] = (*c_array)->crystal[i];
  }

  /* Free memory but we do reuse c_array->crystal[i].atom and c_array->crystal[i].name memory.
   * Note: If c_array->crystal is pointing to the original Crystal_arr defined in xrayglob_inline.c
   * then we cannot free memory.
   */


  if ((*c_array)->crystal != Crystal_arr.crystal) free((*c_array)->crystal);

  *c_array = temp_array;

  return;

}

/*-------------------------------------------------------------------------------------------------- */

void Crystal_ArrayInit (Crystal_Array* c_array, int n_crystal_alloc) {

  c_array->n_crystal = 0;
  c_array->n_alloc = n_crystal_alloc;

  if (n_crystal_alloc == 0) {
    c_array->crystal = NULL;
  } else {
    c_array->crystal = malloc(n_crystal_alloc * sizeof(Crystal_Struct));
  }

}

/*-------------------------------------------------------------------------------------------------- */

void Crystal_ArrayFree (Crystal_Array* c_array) {

  int i;
  for (i = 0; i < c_array->n_crystal; i++) {
    Crystal_Free (&c_array->crystal[i]);
  }
  free(c_array);

}

/*-------------------------------------------------------------------------------------------------- */

Crystal_Struct* Crystal_MakeCopy (Crystal_Struct* crystal) {
  int n;

  Crystal_Struct* crystal_out = malloc(sizeof(Crystal_Struct));

  *crystal_out = *crystal;
  crystal_out->name = strdup(crystal->name);
  n = crystal->n_atom * sizeof(Crystal_Atom);
  crystal_out->atom = malloc(n);
  memcpy (crystal->atom, crystal_out->atom, n);

  return crystal_out;

}

/*-------------------------------------------------------------------------------------------------- */

void Crystal_Free (Crystal_Struct* crystal) {
  free(crystal->name);
  free(crystal->atom);
  free(crystal);
}

/*-------------------------------------------------------------------------------------------------- */

char **Crystal_GetCrystalsList(Crystal_Array *c_array, int *nCrystals) {
  char **rv = NULL;
  int i;

  if (c_array == NULL) {
  	c_array = &Crystal_arr;
  }
  rv = malloc(sizeof(char *) * (c_array->n_crystal + 1));
  for (i = 0 ; i < c_array->n_crystal ; i++)
    rv[i] = strdup(c_array->crystal[i].name);

  rv[c_array->n_crystal] = NULL;

  if (nCrystals != NULL)
    *nCrystals = c_array->n_crystal;

  return rv;
}

/*-------------------------------------------------------------------------------------------------- */

Crystal_Struct* Crystal_GetCrystal (const char* material, Crystal_Array* c_array) {

  if (c_array == NULL) {
  	c_array = &Crystal_arr;
  }

  return bsearch(material, c_array->crystal, c_array->n_crystal, sizeof(Crystal_Struct), matchCrystalStruct);

}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Bragg angle in radians.
 *
 */

double Bragg_angle (Crystal_Struct* crystal, double energy, int i_miller, int j_miller, int k_miller) {
  double d_spacing, wavelength;

  if (i_miller == 0 && j_miller == 0 && k_miller == 0) return 0;

  d_spacing = Crystal_dSpacing (crystal, i_miller, j_miller, k_miller);
  wavelength = KEV2ANGST / energy;
  return asin(wavelength / (2 * d_spacing));

}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Q scattering factor = Sin(theta) / wavelength
 *
 */

double Q_scattering_amplitude(Crystal_Struct* crystal, double energy,
                                    int i_miller, int j_miller, int k_miller, double rel_angle) {
  double wavelength;

  if (i_miller == 0 && j_miller == 0 && k_miller == 0)
    return 0;
  else {
    wavelength = KEV2ANGST / energy;
    return sin(rel_angle * Bragg_angle(crystal, energy, i_miller, j_miller, k_miller)) / wavelength;
  }

}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Atomic Factors f0, f', f''
 *
 */

void Atomic_Factors (int Z, double energy, double q, double debye_factor,
                                  double* f0, double* f_prime, double* f_prime2) {

  *f0       = FF_Rayl(Z, q) * debye_factor;
  *f_prime  = Fi(Z, energy) * debye_factor;
  *f_prime2 = -Fii(Z, energy) * debye_factor;

}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Compute F_H
 *
 */

xrlComplex Crystal_F_H_StructureFactor (Crystal_Struct* crystal, double energy,
                      int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle) {
  return Crystal_F_H_StructureFactor_Partial (crystal, energy, i_miller, j_miller, k_miller,
                                                                          debye_factor, rel_angle, 2, 2, 2);
}

void Crystal_F_H_StructureFactor2 (Crystal_Struct* crystal, double energy,
                      int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle, xrlComplex* result) {

	xrlComplex z = Crystal_F_H_StructureFactor_Partial (crystal, energy, i_miller, j_miller, k_miller,
                                                                          debye_factor, rel_angle, 2, 2, 2);
	result->re = z.re;
	result->im = z.im;
}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Compute F_H
 *
 */

xrlComplex Crystal_F_H_StructureFactor_Partial (Crystal_Struct* crystal, double energy,
                      int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle,
                      int f0_flag, int f_prime_flag, int f_prime2_flag) {

  double f0, f_prime, f_prime2, q;
  double f_re[120], f_im[120], H_dot_r;
  int f_is_computed[120] = {0};
  xrlComplex F_H = {0, 0};
  char buffer[512];
  int i, Z;
  Crystal_Struct* cc = crystal;  /* Just for an abbreviation. */

  /* Loop over all atoms and compute the f values */

  q = Q_scattering_amplitude(cc, energy, i_miller, j_miller, k_miller, rel_angle);

  for (i = 0; i < cc->n_atom; i++) {

    Z = cc->atom[i].Zatom;
    if (f_is_computed[Z]) continue;

    Atomic_Factors (Z, energy, q, debye_factor, &f0, &f_prime, &f_prime2);

    switch (f0_flag) {
    case 0:
      f_re[Z] = 0;
      break;
    case 1:
      f_re[Z] = 1;
      break;
    case 2:
      f_re[Z] = f0;
      break;
    default:
      sprintf (buffer, "Bad f0_flag argument in Crystal_F_H_StructureFactor_Partial: %i", f0_flag);
      ErrorExit(buffer);
      return F_H;
    }

    switch (f_prime_flag) {
    case 0:
      break;
    case 2:
      f_re[Z] = f_re[Z] + f_prime;
      break;
    default:
      sprintf (buffer, "Bad f_prime_flag argument in Crystal_F_H_StructureFactor_Partial: %i", f_prime_flag);
      ErrorExit(buffer);
      return F_H;
    }

    switch (f_prime2_flag) {
    case 0:
      f_im[Z] = 0;
      break;
    case 2:
      f_im[Z] = f_prime2;
      break;
    default:
      sprintf (buffer, "Bad f_prime2_flag argument in Crystal_F_H_StructureFactor_Partial: %i", f_prime2_flag);
      ErrorExit(buffer);
      return F_H;;
    }

    f_is_computed[Z] = 1;

  }

  /* Now compute F_H */

  for (i = 0; i < cc->n_atom; i++) {
    Z = cc->atom[i].Zatom;
    H_dot_r = TWOPI * (i_miller * cc->atom[i].x + j_miller * cc->atom[i].y + k_miller * cc->atom[i].z);
    F_H.re = F_H.re + cc->atom[i].fraction * (f_re[Z] * cos(H_dot_r) - f_im[Z] * sin(H_dot_r));
    F_H.im = F_H.im + cc->atom[i].fraction * (f_re[Z] * sin(H_dot_r) + f_im[Z] * cos(H_dot_r));
  }

  return F_H;

}

void Crystal_F_H_StructureFactor_Partial2(Crystal_Struct* crystal, double energy,
	int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle,
	int f0_flag, int f_prime_flag, int f_prime2_flag, xrlComplex* result) {

	xrlComplex z = Crystal_F_H_StructureFactor_Partial(crystal, energy, i_miller, j_miller, k_miller,
		debye_factor, rel_angle, f0_flag, f_prime_flag, f_prime2_flag);
	result->re = z.re;
	result->im = z.im;
}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Compute unit cell volume
 *
 */

double Crystal_UnitCellVolume (Crystal_Struct* crystal) {

  Crystal_Struct* cc = crystal;  /* Just for an abbreviation. */

  return cc->a * cc->b * cc->c *
                  sqrt( (1 - pow2(cosd(cc->alpha)) - pow2(cosd(cc->beta)) - pow2(cosd(cc->gamma))) +
                         2 * cosd(cc->alpha) * cosd(cc->beta) * cosd(cc->gamma));
}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Compute d-spacing between planes
 *
 */

double Crystal_dSpacing (Crystal_Struct* crystal, int i_miller, int j_miller, int k_miller) {
  Crystal_Struct* cc;

  if (i_miller == 0 && j_miller == 0 && k_miller == 0) return 0;

  cc = crystal;  /* Just for an abbreviation. */

  return (cc->volume / (cc->a * cc->b * cc->c)) * sqrt(1 / (

   pow2(i_miller * sind(cc->alpha) / cc->a) + pow2(j_miller * sind(cc->beta) / cc->b) +
          pow2(k_miller * sind(cc->gamma) / cc->c) +
          2 * i_miller * j_miller * (cosd(cc->alpha) * cosd(cc->beta)  - cosd(cc->gamma)) / (cc->a * cc->b) +
          2 * i_miller * k_miller * (cosd(cc->alpha) * cosd(cc->gamma) - cosd(cc->beta))  / (cc->a * cc->c) +
          2 * j_miller * k_miller * (cosd(cc->beta) * cosd(cc->gamma)  - cosd(cc->alpha)) / (cc->b * cc->c)));
}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Add a new CrystalStruct to an array of crystals.
 *
 */

int Crystal_AddCrystal (Crystal_Struct* crystal, Crystal_Array* c_array) {
  Crystal_Struct* a_cryst;

  if (c_array == NULL) c_array = &Crystal_arr;

  /* See if the crystal material is already present.
   * If so replace it.
   * Otherwise must be a new material...
   */

  a_cryst = bsearch(crystal->name, c_array->crystal, c_array->n_crystal, sizeof(Crystal_Struct), matchCrystalStruct);

  if (a_cryst == NULL) {
    if (c_array->n_crystal == c_array->n_alloc) Crystal_ExtendArray(&c_array, N_NEW_CRYSTAL);
    c_array->crystal[c_array->n_crystal++] = *Crystal_MakeCopy(crystal);
    a_cryst = &c_array->crystal[c_array->n_crystal];
  } else {
    *a_cryst = *Crystal_MakeCopy(crystal);
  }

  /* sort and return */
  a_cryst->volume = Crystal_UnitCellVolume(a_cryst);
  qsort(c_array->crystal, c_array->n_crystal, sizeof(Crystal_Struct), compareCrystalStructs);

  return 1;

}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Read in a set of crystal structs.
 */

int Crystal_ReadFile (const char* file_name, Crystal_Array* c_array) {

  FILE* fp;
  Crystal_Struct* crystal;
  Crystal_Atom* atom;
  int i, n, ex, found_it;
  char tag[21], compound[21], buffer[512];
  long floc;

  if (c_array == NULL) c_array = &Crystal_arr;
#ifdef _WIN32
  /* necesarry to avoid line-ending issues in windows, as pointed out by Matthew Wormington */
  if ((fp = fopen(file_name, "rb")) == NULL) {
#else
  if ((fp = fopen(file_name, "r")) == NULL) {
#endif
    sprintf (buffer, "Crystal file: %s not found\n", file_name);
    ErrorExit(buffer);
    return 0;
  }

  /* Loop over all lines of the file. */

  while (!feof(fp)) {

    /* Start of compound def looks like: "#S <num> <Compound>" */

    fgets (buffer, 100, fp);
    if (buffer[0] != '#' || buffer[1] != 'S') continue;

    ex = sscanf(buffer, "%20s %d %20s", tag, &i, compound);
    if (ex != 3) {
      sprintf (buffer, "In crystal file: %s\n  Malformed '#S <num> <crystal_name>' construct.", file_name);
      ErrorExit(buffer);
      return 0;
    }

    if (c_array->n_crystal == c_array->n_alloc) Crystal_ExtendArray(&c_array, N_NEW_CRYSTAL);
    crystal = &(c_array->crystal[c_array->n_crystal++]);

    crystal->name = strdup(compound);

    /*
     * Parse lines of the crystal definition before list of atom positions.
     * The only info we need to pickup here is the #UCELL unit cell parameters.
     */

    found_it = FALSE;

    while (!feof(fp)) {

      fgets (buffer, 100, fp);

      if (buffer[0] == '#' && buffer[1] == 'L') break;

      if (buffer[0] == '#' && buffer[1] == 'U' && buffer[2] == 'C' &&
          buffer[3] == 'E' && buffer[4] == 'L' && buffer[5] == 'L') {
        ex = sscanf(buffer,"%20s %lf %lf %lf %lf %lf %lf", tag, &crystal->a, &crystal->b, &crystal->c,
                                                       &crystal->alpha, &crystal->beta, &crystal->gamma);
        if (found_it) {
          sprintf (buffer, "In crystal file: %s\n  For crystal definition of: %s.\n  Multiple #UCELL lines found.",
                                                                      file_name, crystal->name);
          ErrorExit(buffer);
          return 0;
        }
        if (ex != 7) {
          sprintf (buffer, "In crystal file: %s\n  For crystal definition of: %s.\n Malformed '#UCELL' construct",
                                                                      file_name, crystal->name);
          ErrorExit(buffer);
          return 0;
        }
        found_it = TRUE;
      }

    }

    /* Error check */

    if (!found_it) {
      sprintf (buffer, "In crystal file: %s\n  For crystal definition of: %s.\n  No #UCELL line found for crystal.",
                                                                      file_name, crystal->name);
      ErrorExit(buffer);
      return 0;
    }

    /* Now read in the atom positions.
     * First count how many atoms there are and then backup to read in the locations.
     */

    floc = ftell(fp);  /* Memorize current location in file */

    n = 0;
    while (!feof(fp)) {
      fgets (buffer, 100, fp);
      if (buffer[0] == '#') break;
      n++;
    }

    if (n == 0 && feof(fp)) {
      sprintf (buffer, "In crystal file: %s\n  For crystal definition of: %s.\n  End of file before definition complete.",
                                                                      file_name, crystal->name);
      ErrorExit(buffer);
      return 0;
    }

    crystal->n_atom = n;
    crystal->atom = malloc(n * sizeof(Crystal_Atom));

    /* Now rewind and fill in the array */

    fseek(fp, floc, SEEK_SET);

    for (i = 0; i < n; i++) {
      atom = &(crystal->atom[i]);
      ex = fscanf(fp, "%i %lf %lf %lf %lf", &atom->Zatom, &atom->fraction, &atom->x, &atom->y, &atom->z);
      if (ex != 5) {
        sprintf (buffer, "In crystal file: %s\n  For crystal definition of: %s.\n  Atom position line %d\n  Error parsing atom position.",
                                                                      file_name, crystal->name, i);
        ErrorExit(buffer);
        return 0;
      }
    }

  }

  fclose(fp);

  /* Now sort */

  qsort(c_array->crystal, c_array->n_crystal, sizeof(Crystal_Struct), compareCrystalStructs);

  /* Now calculate the unit cell volumes */

  for (i = 0; i < c_array->n_crystal; i++) {
    c_array->crystal[i].volume = Crystal_UnitCellVolume(&c_array->crystal[i]);
  }

  return 1;

}
