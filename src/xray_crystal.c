/*
Copyright (c) 2011  David Sagan
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "xray_crystal.h"
#include "xrayglob.h"
#include <stdio.h>
#include <string.h>

//--------------------------------------------------------------------------------------------------
// Private function to extend the crystal array size.

void Crystal_ExtendArray (Crystal_Array* c_array, int n_new) {

  // Special case

  // Transfer data to a temp.

  Crystal_Array* temp_array = malloc(sizeof(Crystal_Array));
  temp_array->n_crystal = c_array->n_crystal;
  temp_array->n_alloc = c_array->n_alloc + n_new;
  temp_array->crystal = malloc(temp_array->n_alloc * sizeof(Crystal_Array));

  int i;
  for (i = 0; i < c_array->n_crystal; i++) {
    temp_array->crystal[i] = c_array->crystal[i];
  }

  // Free memory but we do reuse c_array->crystal[i].atom and c_array->crystal[i].name memory.
  // Note: If c_array->crystal is pointing to the original Crystal_arr defined in xrayglob_inline.c
  // then we cannot free memory.

  if (c_array->crystal != Crystal_arr.crystal) free(c_array->crystal);    

  return;

}

//--------------------------------------------------------------------------------------------------

void Crystal_ArrayInit (Crystal_Array* c_array, int n_crystal_alloc) {

  c_array->n_crystal = 0;
  c_array->n_alloc = n_crystal_alloc;

  if (n_crystal_alloc == 0) {
    c_array->crystal = NULL;
  } else {
    c_array->crystal = malloc(n_crystal_alloc * sizeof(Crystal_Struct));
  }

}

//--------------------------------------------------------------------------------------------------

void Crystal_ArrayFree (Crystal_Array* c_array) {

  int i;
  for (i = 0; i < c_array->n_crystal; i++) {
    Crystal_Free (&c_array->crystal[i]);
  }
  free(c_array);

}

//--------------------------------------------------------------------------------------------------

Crystal_Struct* Crystal_MakeCopy (Crystal_Struct* crystal) {

  Crystal_Struct* crystal_out = malloc(sizeof(Crystal_Struct));
  *crystal_out = *crystal;
  crystal_out->atom = malloc(crystal->n_atom * sizeof(struct CrystalAtom));
  *crystal_out->atom = *crystal->atom;
  return crystal_out;

}

//--------------------------------------------------------------------------------------------------

void Crystal_Free (Crystal_Struct* crystal) {
  free(crystal->name);
  free(crystal->atom);
  free(crystal);
}


//--------------------------------------------------------------------------------------------------

Crystal_Struct* Crystal_GetCrystal (char* material, Crystal_Array* c_array) {

  if (c_array == NULL) c_array = &Crystal_arr;

  return bsearch(material, c_array->crystal, c_array->n_crystal, sizeof(Crystal_Struct), matchCrystalStruct);

}

//--------------------------------------------------------------------------------------------------
// Compute F_H

struct Complex Crystal_F_H_StructureFactor (Crystal_Struct* crystal, double energy, 
                      int i_miller, int j_miller, int k_miller, float debye_factor, float angle_rel) {

}

//--------------------------------------------------------------------------------------------------
// Compute unit cell volume

float Crystal_UnitCellVolume (Crystal_Struct* crystal) {

  float volume;

  volume = crystal->a * crystal->b * crystal->c;   // Temp until real calc is implemented.

  return volume;

}

//--------------------------------------------------------------------------------------------------
// Compute d-spacing between planes

float Crystal_dSpacing (Crystal_Struct* crystal, int i_miller, int j_miller, int k_miller) {


}

//--------------------------------------------------------------------------------------------------
// Add a new CrystalStruct to an array of crystals.

int Crystal_AddCrystal (Crystal_Struct* crystal, Crystal_Array* c_array) {

  if (c_array == NULL) c_array = &Crystal_arr;

  // See if the crystal material is already present.
  // If so replace it.
  // Otherwise must be a new material...

  Crystal_Struct* a_cryst;
  a_cryst = bsearch(crystal->name, c_array->crystal, c_array->n_crystal, sizeof(Crystal_Struct), matchCrystalStruct);
  
  if (a_cryst == NULL) {
    if (c_array->n_crystal == c_array->n_alloc) Crystal_ExtendArray(c_array, N_NEW_CRYSTAL);
    c_array->crystal[c_array->n_crystal++] = *Crystal_MakeCopy(crystal);
    a_cryst = &c_array->crystal[c_array->n_crystal];
  } else {
    *a_cryst = *Crystal_MakeCopy(crystal);
  }

  // sort and return
  a_cryst->volume = Crystal_UnitCellVolume(a_cryst);
	qsort(c_array->crystal, c_array->n_crystal, sizeof(Crystal_Struct), compareCrystalStructs);

  return EXIT_SUCCESS;

}

//--------------------------------------------------------------------------------------------------
// Read in a set of crystal structs.

int Crystal_ReadFile (char* file_name, Crystal_Array* c_array) {

  FILE* fp;
  Crystal_Struct* crystal;
  struct CrystalAtom* atom;
  int i, n, ex, found_it;
  char tag[21], compound[21], buffer[512];
  long floc;

  if (c_array == NULL) c_array = &Crystal_arr;

  if ((fp = fopen(file_name, "r")) == NULL) {
    sprintf (buffer, "Crystal file: %s not found\n", file_name);
    ErrorExit(buffer);
    return EXIT_FAILURE;
  }

  while (!feof(fp)) {

    // Start of compound def looks like: "#S <num> <Compound>"

    fgets (buffer, 100, fp);
    if (buffer[0] != '#' || buffer[1] != 'S') continue;

    ex = sscanf(buffer, "%20s %d %20s", &tag, &i, &compound);
    if (ex != 3) {
      sprintf (buffer, "In crystal file: %s\n  Malformed '#S <num> <crystal_name>' construct.", file_name);
      ErrorExit(buffer);
      return EXIT_FAILURE;
    }

    if (c_array->n_crystal == c_array->n_alloc) Crystal_ExtendArray(c_array, N_NEW_CRYSTAL);
    crystal = &(c_array->crystal[c_array->n_crystal++]);

    crystal->name = strdup(compound);

    // Parse lines of the crystal definition before list of atom positions.
    // The only info we need to pickup here is the #UCELL unit cell parameters.

    found_it = FALSE;

    while (!feof(fp)) {

      fgets (buffer, 100, fp);

      if (buffer[0] == '#' && buffer[1] == 'L') break;

      if (buffer[0] == '#' && buffer[1] == 'U' && buffer[2] == 'C' && 
          buffer[3] == 'E' && buffer[4] == 'L' && buffer[5] == 'L') {
        ex = sscanf(buffer,"%20s %f %f %f %f %f %f", &tag, &crystal->a, &crystal->b, &crystal->c, 
                                                       &crystal->alpha, &crystal->beta, &crystal->gamma);
        if (found_it) {
          sprintf (buffer, "In crystal file: %s\n  For crystal definition of: %s.\n  Multiple #UCELL lines found.", 
                                                                      file_name, crystal->name);
          ErrorExit(buffer);
          return EXIT_FAILURE;
        }
        if (ex != 7) {
          sprintf (buffer, "In crystal file: %s\n  For crystal definition of: %s.\n Malformed '#UCELL' construct",
                                                                      file_name, crystal->name);
          ErrorExit(buffer);
          return EXIT_FAILURE;
        }
        found_it = TRUE;
      }

    }

    // Error check

    if (!found_it) {
      sprintf (buffer, "In crystal file: %s\n  For crystal definition of: %s.\n  No #UCELL line found for crystal.",
                                                                      file_name, crystal->name);
      ErrorExit(buffer);
      return EXIT_FAILURE;
    }

    // Now read in the atom positions.
    // First count how many atoms there are and then backup to read in the locations.

    floc = ftell(fp);  // Memorize current location in file

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
      return EXIT_FAILURE;
    }

    crystal->n_atom = n;
    crystal->atom = malloc(n * sizeof(struct CrystalAtom));

    // Now rewind and fill in the array

    fseek(fp, floc, SEEK_SET);
    
    for (i = 0; i < n; i++) {
      atom = &(crystal->atom[i]);
      ex = fscanf(fp, "%i %f %f %f %f", &atom->Zatom, &atom->fraction, &atom->x, &atom->y, &atom->z);
      if (ex != 5) {
        sprintf (buffer, "In crystal file: %s\n  For crystal definition of: %s.\n  Atom position line %d\n  Error parsing atom position.",
                                                                      file_name, crystal->name, i);
        ErrorExit(buffer);
        return EXIT_FAILURE;
      }
    }

  }

  fclose(fp);

  // Now sort

	qsort(c_array->crystal, c_array->n_crystal, sizeof(Crystal_Struct), compareCrystalStructs);

  // Now calculate the unit cell volumes

  for (i = 0; i < c_array->n_crystal; i++) {
    c_array->crystal[i].volume = Crystal_UnitCellVolume(&c_array->crystal[i]);
  }

  return EXIT_SUCCESS;

}
