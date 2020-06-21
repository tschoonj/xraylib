/*
Copyright (c) 2011-2018  David Sagan, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY David Sagan AND Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "config.h"
#include "xraylib-aux.h"
#include "xraylib-crystal-diffraction.h"
#include "xrayglob.h"
#include "xraylib.h"
#include "xraylib-error-private.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <errno.h>

#define sind(x)  sin(x * DEGRAD)
#define cosd(x)  cos(x * DEGRAD)
#define tand(x)  tan(x * DEGRAD)
#define pow2(x)  pow(x, 2)
#define FALSE 0
#define TRUE 1

/*-------------------------------------------------------------------------------------------------- */

double c_abs(xrlComplex x) {
  double ans = x.re * x.re + x.im * x.im;
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

static int Crystal_ExtendArray(Crystal_Array** c_array, int n_new, xrl_error **error) {
  int i;

  /* Special case */

  /* Transfer data to a temp. */

  Crystal_Array *temp_array = malloc(sizeof(Crystal_Array));
  if (temp_array == NULL) {
    xrl_set_error(error, XRL_ERROR_MEMORY, MALLOC_ERROR, strerror(errno));
    return 0;
  }
  temp_array->n_crystal = (*c_array)->n_crystal;
  temp_array->n_alloc = (*c_array)->n_alloc + n_new;
  temp_array->crystal = malloc(temp_array->n_alloc * sizeof(Crystal_Struct));
  if (temp_array->crystal == NULL) {
    xrl_set_error(error, XRL_ERROR_MEMORY, MALLOC_ERROR, strerror(errno));
    free(temp_array);
    return 0;
  }

  for (i = 0; i < (*c_array)->n_crystal; i++) {
    temp_array->crystal[i] = (*c_array)->crystal[i];
  }

  /* Free memory but we do reuse c_array->crystal[i].atom and c_array->crystal[i].name memory.
   * Note: If c_array->crystal is pointing to the original Crystal_arr defined in xrayglob_inline.c
   * then we cannot free memory.
   */


  if ((*c_array)->crystal != Crystal_arr.crystal)
    free((*c_array)->crystal);

  *c_array = temp_array;

  return 1;
}

/*-------------------------------------------------------------------------------------------------- */

Crystal_Array* Crystal_ArrayInit(int n_crystal_alloc, xrl_error **error) {

  Crystal_Array *c_array = malloc(sizeof(Crystal_Array));
  if (c_array == NULL) {
    xrl_set_error(error, XRL_ERROR_MEMORY, MALLOC_ERROR, strerror(errno));
    return NULL;
  }

  c_array->n_crystal = 0;
  c_array->n_alloc = n_crystal_alloc;

  if (n_crystal_alloc == 0) {
    c_array->crystal = NULL;
  } else if (n_crystal_alloc < 0) {
    free(c_array);
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Negative n_crystal_alloc is not allowed");
    return NULL;
  } else {
    c_array->crystal = malloc(n_crystal_alloc * sizeof(Crystal_Struct));
    if (c_array->crystal == NULL) {
      xrl_set_error(error, XRL_ERROR_MEMORY, MALLOC_ERROR, strerror(errno));
      free(c_array);
      return NULL;
    }
  }
  return c_array;
}

/*-------------------------------------------------------------------------------------------------- */

void Crystal_ArrayFree(Crystal_Array *c_array) {

  int i;
  if (c_array == NULL)
    return;
  for (i = 0; i < c_array->n_crystal; i++) {
    if (c_array->crystal[i].name)
      free(c_array->crystal[i].name);
    if (c_array->crystal[i].atom)
      free(c_array->crystal[i].atom);
  }
  if (c_array->crystal)
    free(c_array->crystal);
  free(c_array);
}

/*-------------------------------------------------------------------------------------------------- */

Crystal_Struct* Crystal_MakeCopy (Crystal_Struct *crystal, xrl_error **error) {
  int n;
  Crystal_Struct *crystal_out = NULL;

  if (crystal == NULL) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Crystal cannot be NULL");
    return NULL;
  }

  crystal_out = malloc(sizeof(Crystal_Struct));
  if (crystal_out == NULL) {
    xrl_set_error(error, XRL_ERROR_MEMORY, MALLOC_ERROR, strerror(errno));
    return NULL;
  }

  *crystal_out = *crystal;
  crystal_out->name = xrl_strdup(crystal->name);
  n = crystal->n_atom * sizeof(Crystal_Atom);
  crystal_out->atom = malloc(n);
  if (crystal_out->atom == NULL) {
    xrl_set_error(error, XRL_ERROR_MEMORY, MALLOC_ERROR, strerror(errno));
    free(crystal_out->name);
    free(crystal_out);
    return NULL;
  }
  memcpy(crystal_out->atom, crystal->atom, n);

  return crystal_out;
}

/*-------------------------------------------------------------------------------------------------- */

void Crystal_Free(Crystal_Struct* crystal) {
  if (crystal == NULL)
    return;
  free(crystal->name);
  free(crystal->atom);
  free(crystal);
}

/*-------------------------------------------------------------------------------------------------- */

char** Crystal_GetCrystalsList(Crystal_Array *c_array, int *nCrystals, xrl_error **error) {
  char **rv = NULL;
  int i;

  if (c_array == NULL) {
  	c_array = &Crystal_arr;
  }
  rv = malloc(sizeof(char *) * (c_array->n_crystal + 1));
  if (rv == NULL) {
    xrl_set_error(error, XRL_ERROR_MEMORY, MALLOC_ERROR, strerror(errno));
    return NULL;
  }
  for (i = 0 ; i < c_array->n_crystal ; i++)
    rv[i] = xrl_strdup(c_array->crystal[i].name);

  rv[c_array->n_crystal] = NULL;

  if (nCrystals != NULL)
    *nCrystals = c_array->n_crystal;

  return rv;
}

/*-------------------------------------------------------------------------------------------------- */

Crystal_Struct* Crystal_GetCrystal (const char* material, Crystal_Array* c_array, xrl_error **error) {
  Crystal_Struct *rv, *rv_copy;
  if (material == NULL) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Crystal cannot be NULL");
    return NULL;
  }

  if (c_array == NULL) {
    c_array = &Crystal_arr;
  }

  rv = bsearch(material, c_array->crystal, c_array->n_crystal, sizeof(Crystal_Struct), matchCrystalStruct);
  if (rv == NULL) {
    xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "Crystal %s is not present in array", material);
    return NULL;
  }

  rv_copy = Crystal_MakeCopy(rv, error);

  return rv_copy;
}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Bragg angle in radians.
 *
 */

double Bragg_angle(Crystal_Struct* crystal, double energy, int i_miller, int j_miller, int k_miller, xrl_error **error) {
  double d_spacing, wavelength;

  if (energy <= 0.0) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, NEGATIVE_ENERGY);
    return 0.0;
  }

  d_spacing = Crystal_dSpacing(crystal, i_miller, j_miller, k_miller, error);
  if (d_spacing == 0.0)
    return 0.0;

  wavelength = KEV2ANGST / energy;
  return asin(wavelength / (2 * d_spacing));

}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Q scattering factor = Sin(theta) / wavelength
 *
 */

double Q_scattering_amplitude(Crystal_Struct* crystal, double energy, int i_miller, int j_miller, int k_miller, double rel_angle, xrl_error **error) {

  if (energy <= 0.0) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, NEGATIVE_ENERGY);
    return 0.0;
  }

  if (i_miller == 0 && j_miller == 0 && k_miller == 0)
	  return 0.0;

  return energy * sin(rel_angle * Bragg_angle(crystal, energy, i_miller, j_miller, k_miller, error)) / KEV2ANGST;
}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Atomic Factors f0, f', f''
 *
 */

int Atomic_Factors (int Z, double energy, double q, double debye_factor, double *f0, double *f_prime, double *f_prime2, xrl_error **error) {

  if (debye_factor <= 0.0) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, NEGATIVE_DEBYE_FACTOR);
    if (f0)
      *f0 = 0.0;
    if (f_prime)
      *f_prime = 0.0;
    if (f_prime2)
      *f_prime2 = 0.0;
    return 0;
  }

  if ((f0 && (*f0 = FF_Rayl(Z, q, error) * debye_factor) == 0.0) ||
      (f_prime && (*f_prime  = Fi(Z, energy, error) * debye_factor) == 0.0) ||
      (f_prime2 && (*f_prime2 = -Fii(Z, energy, error) * debye_factor) == 0.0)) {
    if (f0)
      *f0 = 0.0;
    if (f_prime)
      *f_prime = 0.0;
    if (f_prime2)
      *f_prime2 = 0.0;
    return 0;
  }
  return 1;
}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Compute F_H
 *
 */

xrlComplex Crystal_F_H_StructureFactor(Crystal_Struct* crystal, double energy, int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle, xrl_error **error) {
  return Crystal_F_H_StructureFactor_Partial(crystal, energy, i_miller, j_miller, k_miller, debye_factor, rel_angle, 2, 2, 2, error);
}

XRL_EXTERN
void Crystal_F_H_StructureFactor2(Crystal_Struct* crystal, double energy, int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle, xrlComplex* result, xrl_error **error);

void Crystal_F_H_StructureFactor2(Crystal_Struct* crystal, double energy, int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle, xrlComplex* result, xrl_error **error) {

	xrlComplex z = Crystal_F_H_StructureFactor_Partial(crystal, energy, i_miller, j_miller, k_miller, debye_factor, rel_angle, 2, 2, 2, error);
	result->re = z.re;
	result->im = z.im;
}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Compute F_H
 *
 */

xrlComplex Crystal_F_H_StructureFactor_Partial (Crystal_Struct* crystal, double energy, int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle, int f0_flag, int f_prime_flag, int f_prime2_flag, xrl_error **error) {

  double f0, f_prime, f_prime2, q;
  double f_re[120], f_im[120], H_dot_r;
  int f_is_computed[120] = {0};
  xrlComplex F_H = {0, 0};
  int i, Z;
  Crystal_Struct* cc = crystal;  /* Just for an abbreviation. */
  xrl_error *tmp_error = NULL;

  /* Loop over all atoms and compute the f values */

  /* having a zero amplitude is perfectly acceptable if all Miller indices are zero */
  q = Q_scattering_amplitude(cc, energy, i_miller, j_miller, k_miller, rel_angle, &tmp_error);
  if (tmp_error != NULL) {
    xrl_propagate_error(error, tmp_error);
    return F_H;
  }

  for (i = 0; i < cc->n_atom; i++) {

    Z = cc->atom[i].Zatom;
    if (f_is_computed[Z])
      continue;

    if (Atomic_Factors(Z, energy, q, debye_factor, &f0, &f_prime, &f_prime2, error) == 0)
      return F_H;

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
      xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid f0_flag argument: %d", f0_flag);
      return F_H;
    }

    switch (f_prime_flag) {
    case 0:
      break;
    case 2:
      f_re[Z] = f_re[Z] + f_prime;
      break;
    default:
      xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid f_prime_flag argument: %d", f_prime_flag);
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
      xrl_set_error(error, XRL_ERROR_INVALID_ARGUMENT, "Invalid f_prime2_flag argument: %d", f_prime2_flag);
      return F_H;
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

XRL_EXTERN
void Crystal_F_H_StructureFactor_Partial2(Crystal_Struct* crystal, double energy,
	int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle,
	int f0_flag, int f_prime_flag, int f_prime2_flag, xrlComplex* result, xrl_error **error);

void Crystal_F_H_StructureFactor_Partial2(Crystal_Struct* crystal, double energy,
	int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle,
	int f0_flag, int f_prime_flag, int f_prime2_flag, xrlComplex* result, xrl_error **error) {

	xrlComplex z = Crystal_F_H_StructureFactor_Partial(crystal, energy, i_miller, j_miller, k_miller,
		debye_factor, rel_angle, f0_flag, f_prime_flag, f_prime2_flag, error);
	result->re = z.re;
	result->im = z.im;
}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Compute unit cell volume
 *
 */

double Crystal_UnitCellVolume(Crystal_Struct* crystal, xrl_error **error) {

  Crystal_Struct* cc = crystal;  /* Just for an abbreviation. */

  if (cc == NULL) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, CRYSTAL_NULL);
    return 0.0;
  }

  return cc->a * cc->b * cc->c *
                  sqrt( (1 - pow2(cosd(cc->alpha)) - pow2(cosd(cc->beta)) - pow2(cosd(cc->gamma))) +
                         2 * cosd(cc->alpha) * cosd(cc->beta) * cosd(cc->gamma));
}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Compute d-spacing between planes
 *
 */

double Crystal_dSpacing(Crystal_Struct* crystal, int i_miller, int j_miller, int k_miller, xrl_error **error) {
  Crystal_Struct* cc;

  if (crystal == NULL) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, CRYSTAL_NULL);
    return 0.0;
  }

  if (i_miller == 0 && j_miller == 0 && k_miller == 0) {
	  xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, INVALID_MILLER);
	  return 0;
  }

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

int Crystal_AddCrystal(Crystal_Struct* crystal, Crystal_Array* c_array, xrl_error **error) {
  Crystal_Struct* a_cryst;

  if (c_array == NULL)
    c_array = &Crystal_arr;

  if (crystal == NULL) {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, CRYSTAL_NULL);
    return 0;
  }

  /* See if the crystal material is already present.
   * If so replace it.
   * Otherwise must be a new material...
   */

  a_cryst = bsearch(crystal->name, c_array->crystal, c_array->n_crystal, sizeof(Crystal_Struct), matchCrystalStruct);

  if (a_cryst == NULL) {
    Crystal_Struct *tmp = NULL;
    if (c_array->n_crystal == c_array->n_alloc) {
      if (c_array == &Crystal_arr) {
        xrl_set_error_literal(error, XRL_ERROR_RUNTIME, "Extending internal is crystal array is not allowed");
	return 0;
      }
      else if (Crystal_ExtendArray(&c_array, N_NEW_CRYSTAL, error) == 0) {
        return 0;
      }
    }
    tmp = Crystal_MakeCopy(crystal, error);
    if (tmp == NULL)
      return 0;
    c_array->crystal[c_array->n_crystal++] = *tmp;
    free(tmp);
    a_cryst = &c_array->crystal[c_array->n_crystal];
  } else {
    xrl_set_error_literal(error, XRL_ERROR_INVALID_ARGUMENT, "Crystal already present in array");
    return 0;
  }

  /* sort and return */
  a_cryst->volume = Crystal_UnitCellVolume(a_cryst, NULL);
  qsort(c_array->crystal, c_array->n_crystal, sizeof(Crystal_Struct), compareCrystalStructs);

  return 1;

}

/*-------------------------------------------------------------------------------------------------- */
/*
 * Read in a set of crystal structs.
 */

int Crystal_ReadFile(const char* file_name, Crystal_Array* c_array, xrl_error **error) {

  FILE* fp;
  Crystal_Struct* crystal;
  Crystal_Atom* atom;
  int i, n, ex, found_it;
  char tag[21], compound[21], buffer[512];
  long floc;

  if (file_name == NULL) {
    xrl_set_error_literal(error, XRL_ERROR_IO, "NULL filenames are not allowed");
    return 0;
  }

  if (c_array == NULL)
    c_array = &Crystal_arr;

#ifdef _WIN32
  /* necesarry to avoid line-ending issues in windows, as pointed out by Matthew Wormington */
  if ((fp = fopen(file_name, "rb")) == NULL)
#else
  if ((fp = fopen(file_name, "r")) == NULL)
#endif
  { 
    xrl_set_error(error, XRL_ERROR_IO, "Could not open %s for reading: %s", file_name, strerror(errno));
    return 0;
  }

  /* Loop over all lines of the file. */

  while (!feof(fp)) {

    /* Start of compound def looks like: "#S <num> <Compound>" */

    char *rv = fgets (buffer, 100, fp);

    if (buffer[0] != '#' || buffer[1] != 'S')
      continue;

    ex = sscanf(buffer, "%20s %d %20s", tag, &i, compound);
    if (ex != 3) {
      xrl_set_error_literal(error, XRL_ERROR_IO, "Malformed '#S <num> <crystal_name>' construct");
      return 0;
    }

    if (c_array->n_crystal == c_array->n_alloc) {
      if (Crystal_ExtendArray(&c_array, N_NEW_CRYSTAL, error) == 0)
        return 0;
    }
    crystal = &(c_array->crystal[c_array->n_crystal++]);

    crystal->name = xrl_strdup(compound);

    /*
     * Parse lines of the crystal definition before list of atom positions.
     * The only info we need to pickup here is the #UCELL unit cell parameters.
     */

    found_it = FALSE;

    while (!feof(fp)) {

      rv = fgets (buffer, 100, fp);

      if (buffer[0] == '#' && buffer[1] == 'L') break;

      if (buffer[0] == '#' && buffer[1] == 'U' && buffer[2] == 'C' &&
          buffer[3] == 'E' && buffer[4] == 'L' && buffer[5] == 'L') {
        ex = sscanf(buffer,"%20s %lf %lf %lf %lf %lf %lf", tag, &crystal->a, &crystal->b, &crystal->c,
                                                       &crystal->alpha, &crystal->beta, &crystal->gamma);
        if (found_it) {
          xrl_set_error(error, XRL_ERROR_IO, "Multiple #UCELL lines found for crystal %s", crystal->name);
          return 0;
        }
        if (ex != 7) {
          xrl_set_error(error, XRL_ERROR_IO, "Malformed #UCELL line found for crystal %s", crystal->name);
          return 0;
        }
        found_it = TRUE;
      }

    }

    /* Error check */

    if (!found_it) {
      xrl_set_error(error, XRL_ERROR_IO, "No #UCELL line found for crystal %s", crystal->name);
      return 0;
    }

    /* Now read in the atom positions.
     * First count how many atoms there are and then backup to read in the locations.
     */

    floc = ftell(fp);  /* Memorize current location in file */

    n = 0;
    while (!feof(fp)) {
      rv = fgets (buffer, 100, fp);

      if (buffer[0] == '#') break;
      n++;
    }

    if (n == 0 && feof(fp)) {
      xrl_set_error_literal(error, XRL_ERROR_IO, "End of file encountered before definition was complete");
      return 0;
    }

    crystal->n_atom = n;
    crystal->atom = malloc(n * sizeof(Crystal_Atom));
    if (crystal->atom == NULL) {
      xrl_set_error(error, XRL_ERROR_MEMORY, MALLOC_ERROR, strerror(errno));
      return 0;
    }

    /* Now rewind and fill in the array */

    fseek(fp, floc, SEEK_SET);

    for (i = 0; i < n; i++) {
      atom = &(crystal->atom[i]);
      ex = fscanf(fp, "%i %lf %lf %lf %lf", &atom->Zatom, &atom->fraction, &atom->x, &atom->y, &atom->z);
      if (ex != 5) {
	xrl_set_error(error, XRL_ERROR_IO, "Could not parse atom position on line %d for crystal %s", i, crystal->name);
        return 0;
      }
    }

  }

  fclose(fp);

  /* Now sort */

  qsort(c_array->crystal, c_array->n_crystal, sizeof(Crystal_Struct), compareCrystalStructs);

  /* Now calculate the unit cell volumes */

  for (i = 0; i < c_array->n_crystal; i++) {
    c_array->crystal[i].volume = Crystal_UnitCellVolume(&c_array->crystal[i], NULL);
  }

  return 1;

}
