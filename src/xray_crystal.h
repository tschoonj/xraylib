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

// Get a CrystalStruct for use in other routines.
// crystal.name will be "" if the material is unknown.

struct CrystalStruct Crystal_GetCrystalStruct(char* material);

// Compute F_H

complex Crystal_F_H_StructureFactor (struct CrystalStruct crystal, double energy, int i, int j, int k, 
                                                                      float debye_factor, float angle_rel);

// Compute unit cell volume

float Crystal_UnitCellVolume (struct CrystalStruct crystal);

// Alphabetical list of material names.

char** Crystal_GetMaterialNames();  

// Add a new CrystalStruct to the official array of crystals.
// If the material already exists in the array then it is overwitten. 
// For multi-threaded programs, locking will be needed while this function is executing.
// Note: This routine is not needed for computing F_H, etc.

bool Crystal_AddCrystalStruct (struct CrystalStruct material);

// Read in a set of crystal structs.
// If crystal_array is NULL then the crystals are added to the official array of crystals.
// On input, n_crystal should be the number of existing cyrstals in crystal_array.
// On output, n_crystal will be the total number of crystals.
// If crystal_array is NULL, n_crystals is ignored.
// For multi-threaded programs, locking will be needed while this function is executing.
// Note: This routine is not needed for computing F_H, etc.

bool Crystal_AddCrystalStructs (char* file_name, struct CrystalStruct* crystal_array, int* n_crystals);

#endif
