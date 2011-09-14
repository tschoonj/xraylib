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

struct CrystalStruct* Crystal_MakeCrystalStructCopy (struct CrystalStruct* crystal) {
  struct CrystalStruct* crystal_out = malloc(sizeof(CrystalStruct));
  *crystal_out = *crystal;
  crystal_out->atom = malloc(crystal->n_atom * sizeof(CrystalAtom));
  *crystal_out->atom = *crystal->atom;
  return crystal_out;
}

//--------------------------------------------------------------------------------------------------

void Crystal_FreeCrystalStruct (struct CrystalStruct* crystal) {
  free(crystal->atom);
  free(crystal);
}


//--------------------------------------------------------------------------------------------------

struct CrystalStruct* Crystal_GetCrystalStruct(char* material) {

  return bsearch(material, CrystalArray, crystalarray_max, sizeof(struct CrystalStruct), matchCrystalStruct);

}

//--------------------------------------------------------------------------------------------------
// Compute F_H

complex Crystal_F_H_StructureFactor (struct CrystalStruct* crystal, double energy, 
                      int i_miller, int j_miller, int k_miller, float debye_factor, float angle_rel) {

}

//--------------------------------------------------------------------------------------------------
// Compute unit cell volume

float Crystal_UnitCellVolume (struct CrystalStruct* crystal) {

  float volume;

  volume = crystal.a * crystal.b * crystal.c;   // Temp until real calc is implemented.

  return volume;

}

//--------------------------------------------------------------------------------------------------
// Compute d-spacing between planes

float Crystal_dSpacing (struct CrystalStruct crystal, int i_miller, int j_miller, int k_miller) {


}

//--------------------------------------------------------------------------------------------------
// Alphabetical list of material names.

char** Crystal_GetMaterialNames() {


}

//--------------------------------------------------------------------------------------------------
// Add a new CrystalStruct to the official array of crystals.

int Crystal_AddCrystalStruct (struct CrystalStruct* crystal) {


  // See if the crystal material is already present.
  // If so replace it.
  // Otherwise must be a new material...

  struct CrystalStruct* cryst;
  cryst = bsearch(crystal->material, CrystalArray, crystalarray_max, sizeof(struct CrystalStruct), matchCrystalStruct);
  
  if (crystal != NULL) {
    *cryst_ptr = crystal;
    return EXIT_SUCCESS;
  } else {
    if (crystalarray_max == CRYSTALARRAY_MAX) {
      ErrorExit("Number of Crystals in Crystals.dat exceeds internal array size.");
      return EXIT_FAILURE;
    }
    CrystalArray[++crystalarray_max] = crystal;
  }

  // Find volume, sort and return

  crystal->volume = Crystal_UnitCellVolume(&CrystalArray[i]);

	qsort(CrystalArray, crystalarray_max, sizeof(struct CrystalStruct), compareCrystalStructs);
  
  return EXIT_SUCCESS;

}

//--------------------------------------------------------------------------------------------------
// Read in a set of crystal structs.

int Crystal_ReadFile (char* file_name) {

  FILE* fp;
  struct CrystalStruct* crystal;
  struct CrystalAtom* atom;
  int i, n, ex, found_it;
  char tag[21], compound[21], buffer[128];
  long floc;

  if ((fp = fopen(file_name, "r")) == NULL) {
    printf ("Full file name: %s\n", file_name);
    ErrorExit("File Crystals.dat not found");
    return EXIT_FAILURE;
  }

  while (!feof(fp)) {

    // Start of compound def looks like: "#S <num> <Compound>"

    fgets (buffer, 100, fp);
    if (buffer[0] != '#' || buffer[1] != 'S') continue;

    ex = sscanf(buffer, "%20s %d %20s", &tag, &i, &compound);
    if (ex != 3) {
      ErrorExit("Malformed '#S <num> <crystal_name>' construct.");
      return EXIT_FAILURE;
    }

    crystal = &(CrystalArray[crystalarray_max++]);
    if (crystalarray_max > CRYSTALARRAY_MAX) {
      ErrorExit("Number of Crystals in Crystals.dat exceeds internal array size.");
      return EXIT_FAILURE;
    }

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
          printf ("For crystal definition of: %s in Crystals.dat.\n", crystal->name);
          ErrorExit ("Multiple #UCELL lines found.");
          return EXIT_FAILURE;
        }
        if (ex != 7) {
          printf ("For crystal definition of: %s in Crystals.dat.\n", crystal->name);
          ErrorExit("Malformed '#UCELL' construct");
          return EXIT_FAILURE;
        }
        found_it = TRUE;
      }

    }

    // Error check

    if (!found_it) {
      printf ("For crystal definition of: %s in Crystals.dat.\n", crystal->name);
      ErrorExit ("No #UCELL line found for crystal."); 
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
      printf ("For crystal definition of: %s in Crystals.dat.\n", crystal->name);
      ErrorExit ("End of file before definition complete.");
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
        printf ("For crystal definition of: %s in Crystals.dat. Atom position line %d\n", crystal->name, i);
        ErrorExit ("Error parsing atom position.");
        return EXIT_FAILURE;
      }
    }

  }

  fclose(fp);

  // Now sort

	qsort(CrystalArray, crystalarray_max, sizeof(struct CrystalStruct), compareCrystalStructs);

  // Now calculate the unit cell volumes

  for (i = 0; i < crystalarray_max; i++) {
    CrystalArray[i].volume = Crystal_UnitCellVolume(CrystalArray[i]);
  }

  return EXIT_SUCCESS;

}
