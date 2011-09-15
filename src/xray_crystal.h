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

// Note for multithreded programs: 
// The routines Crystal_ReadCrystals and CrystalAddCrystalStruct are not thread safe if crystals are 
// added to the official array. In this case, locking will have to be used.


// Note on memory usage:
// A CrystalStruct has a pointer to an array of CrystalAtom as well as a name string.
// Therefore Crystal_FreeMemory needs to be used to free memeory.

//--------------------------------------------------------------------------------
// Copy a CrystalStruct.

struct CrystalStruct Crystal_MakeCopy (struct CrystalStruct crystal);

//--------------------------------------------------------------------------------
// Free malloc'd memory in CrystalStruct.

void Crystal_FreeMemory (struct CrystalStruct crystal);

//--------------------------------------------------------------------------------
// Get a to a CrystalStruct of a given material from the crystal_array.
// If crystal_array is NULL then the official array of crystals is searched and the n_crystals argument is ignored.
// n_crystal is the number of defined cyrstals in crystal_array.
// If not found, the returned crystal will have .n_atom set to -1.

struct CrystalStruct Crystal_GetCrystal(char* material, struct CrystalStruct* crystal_array, int n_crystals);

//--------------------------------------------------------------------------------
// Compute F_H

struct Complex Crystal_F_H_StructureFactor (struct CrystalStruct crystal, double energy, 
                      int i_miller, int j_miller, int k_miller, float debye_factor, float angle_rel);

//--------------------------------------------------------------------------------
// Compute unit cell volume.
// Note: Structures obtained from crystal array will have their volume in .volume.

float Crystal_UnitCellVolume (struct CrystalStruct crystal);

//--------------------------------------------------------------------------------
// Compute d-spacing between planes

float Crystal_dSpacing (struct CrystalStruct crystal, int i_miller, int j_miller, int k_miller);

//--------------------------------------------------------------------------------
// Add a new CrystalStruct to crystal_array.
// If the material already exists in the array then the existing material data is overwitten. 
// If crystal_array is NULL then the crystals are added to the official array of crystals and
//   the n_crystals and array_max arguments are ignored.
// On input, n_crystal should be the number of existing cyrstals in crystal_array.
// On output, n_crystal will be the total number of crystals.
// array_max is the size of the crystal_array.
// Return: EXIT_SUCCESS or EXIT_FAILURE.

int Crystal_AddCrystal (struct CrystalStruct crystal, struct CrystalStruct* crystal_array, int* n_crystals, int array_max);

//--------------------------------------------------------------------------------
// Read in a set of crystal structs to crystal_array.
// If a material already exists in the array then the existing material data is overwitten. 
// If crystal_array is NULL then the crystals are added to the official array of crystals and
//   the n_crystals and array_max arguments are ignored.
// On input, n_crystal should be the number of existing cyrstals in crystal_array.
// On output, n_crystal will be the total number of crystals.
// array_max is the size of the crystal_array.
// Return: EXIT_SUCCESS or EXIT_FAILURE.

int Crystal_ReadFile (char* file_name, struct CrystalStruct crystal_array[], int* n_crystals, int array_max);

#endif
