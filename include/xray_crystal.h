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
// Crystal_FreeMemory needs to be used to free memeory.

//--------------------------------------------------------------------------------
// Initialize a new crystal array.

void Crystal_ArrayInit (Crystal_Array* c_array, int n_crystal_alloc);

//--------------------------------------------------------------------------------
// free memory from a crystal array.

void Crystal_ArrayFree (Crystal_Array* c_array);

//--------------------------------------------------------------------------------
// Copy a CrystalStruct.

Crystal_Struct* Crystal_MakeCopy (Crystal_Struct* crystal);

//--------------------------------------------------------------------------------
// Free malloc'd memory in a CrystalStruct.

void Crystal_Free (Crystal_Struct* crystal);

//--------------------------------------------------------------------------------
// Get a to a CrystalStruct of a given material from the crystal_array.
// If c_array is NULL then the official array of crystals is searched.
// If not found, NULL is returned.

Crystal_Struct* Crystal_GetCrystal(const char* material, Crystal_Array* c_array);

//--------------------------------------------------------------------------------
// Compute F_H

ComplexStruct Crystal_F_H_StructureFactor (Crystal_Struct* crystal, double energy, 
                      int i_miller, int j_miller, int k_miller, float debye_factor, float rel_angle);

//--------------------------------------------------------------------------------------------------
// Compute F_H

ComplexStruct Crystal_F_H_StructureFactor_Partial (Crystal_Struct* crystal, double energy, 
                      int i_miller, int j_miller, int k_miller, float debye_factor, float rel_angle,
                      int f0_flag, int f_prime_flag, int f_prime2_flag);

//--------------------------------------------------------------------------------
// Compute unit cell volume.
// Note: Structures obtained from crystal array will have their volume in .volume.

float Crystal_UnitCellVolume (Crystal_Struct* crystal);

//--------------------------------------------------------------------------------
// Compute d-spacing between planes.
// This routine assumes that if crystal.volume is nonzero then it holds a valid value.
// If (i, j, k) = (0, 0, 0) then zero is returned.

float Crystal_dSpacing (Crystal_Struct* crystal, int i_miller, int j_miller, int k_miller);

//--------------------------------------------------------------------------------
// Add a new CrystalStruct to crystal_array.
// The data is copied to crystal_array.
// If the material already exists in the array then the existing material data is overwitten. 
// If crystal_array is NULL then the crystals are added to the official array of crystals.
// Return: EXIT_SUCCESS or EXIT_FAILURE.

int Crystal_AddCrystal (Crystal_Struct* crystal, Crystal_Array* c_array);

//--------------------------------------------------------------------------------
// Read in a set of crystal structs to crystal_array.
// If a material already exists in the array then the existing material data is overwitten. 
// If crystal_array is NULL then the crystals are added to the official array of crystals.
// Return: EXIT_SUCCESS or EXIT_FAILURE.

int Crystal_ReadFile (const char* file_name, Crystal_Array* c_array);

#endif
