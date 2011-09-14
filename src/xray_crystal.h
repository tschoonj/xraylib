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


#ifndef XRAY_CRYSTAL
#define XRAY_CRYSTAL

#include "xray_defs.h"

// Copy a CrystalStruct.
// Crystal_FreeCrystalStruct must be called to destroy the copy made.

struct CrystalStruct* Crystal_MakeCrystalStructCopy (struct CrystalStruct* crystal);

// Destroy a CrystalStruct.

void Crystal_FreeCrystalStruct (struct CrystalStruct* crystal);

// Get a pointer to a CrystalStruct of a given material for use in other routines.
// Will return NULL is material is not known.

struct CrystalStruct* Crystal_GetCrystalStruct(char* material);

// Compute F_H

complex Crystal_F_H_StructureFactor (struct CrystalStruct* crystal, double energy, 
                      int i_miller, int j_miller, int k_miller, float debye_factor, float angle_rel);

// Compute unit cell volume.
// Note: Structures obtained from the CrystalArray array of crystals will alrady 
//  have their volume in crystal.volume.

float Crystal_UnitCellVolume (struct CrystalStruct* crystal);

// Compute d-spacing between planes

float Crystal_dSpacing (struct CrystalStruct* crystal, int i_miller, int j_miller, int k_miller);

// Alphabetical list of material names.

char** Crystal_GetMaterialNames();  

// Add a new CrystalStruct to the CrystalArray array of crystals.
// If the material already exists in the array then it is overwitten. 
// Return EXIT_SUCCESS or EXIT_FAILURE.

int Crystal_AddCrystalStruct (struct CrystalStruct* crystal);

// Read in a set of crystal structs.
// The crystals will be added to the CrystalArray array of crystals.
// Return EXIT_SUCCESS or EXIT_FAILURE.

int Crystal_ReadCrystals (char* file_name);

#endif
