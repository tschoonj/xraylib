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


#ifndef XRAYLIB_CRYSTAL_DIFFRACTION_H
#define XRAYLIB_CRYSTAL_DIFFRACTION_H

#include "xraylib-defs.h"
#include "xraylib-error.h"

/* Note for multithreaded programs:
 * The routines Crystal_ReadCrystals and CrystalAddCrystalStruct are not thread safe if crystals are
 * added to the official array. In this case, locking will have to be used.
 *
 * Parameters:
 * energy    -- KeV
 * rel_angle -- photon angle / bragg angle
 *
 *
 * --------------------------------------------------------------------------------
 *  Allocate and initialize a new crystal array.
 *
 */

XRL_EXTERN
Crystal_Array* Crystal_ArrayInit(int n_crystal_alloc, xrl_error **error);

/*--------------------------------------------------------------------------------
 * free memory from a crystal array.
 */

XRL_EXTERN
void Crystal_ArrayFree (Crystal_Array* c_array);

/*--------------------------------------------------------------------------------
 * Copy a CrystalStruct.
 */

XRL_EXTERN
Crystal_Struct* Crystal_MakeCopy (Crystal_Struct* crystal, xrl_error **error);

/*--------------------------------------------------------------------------------
 * Free malloc'd memory in a CrystalStruct.
 */

XRL_EXTERN
void Crystal_Free (Crystal_Struct* crystal);

/*--------------------------------------------------------------------------------
 * Get a pointer to a CrystalStruct of a given material from the crystal_array.
 *
 * If c_array is NULL then the official array of crystals is searched.
 * If not found, NULL is returned.
 * Free the returned struct with Crystal_Free.
 */

XRL_EXTERN
Crystal_Struct* Crystal_GetCrystal(const char* material, Crystal_Array* c_array, xrl_error **error);

/*--------------------------------------------------------------------------------------------------
 * Bragg angle in radians.
 */

XRL_EXTERN
double Bragg_angle (Crystal_Struct* crystal, double energy, int i_miller, int j_miller, int k_miller, xrl_error **error);

/*--------------------------------------------------------------------------------------------------
 * Q scattering factor = Sin(theta) / wavelength
 */

XRL_EXTERN
double Q_scattering_amplitude(Crystal_Struct* crystal, double energy,
                                    int i_miller, int j_miller, int k_miller, double rel_angle, xrl_error **error);

/*--------------------------------------------------------------------------------------------------
 * Atomic Factors f0, f', f''
 */

XRL_EXTERN
int Atomic_Factors (int Z, double energy, double q, double debye_factor, double *f0, double *f_primep, double *f_prime2, xrl_error **error);

/*--------------------------------------------------------------------------------
 * Compute F_H
 * See also Crystal_F_H_StructureFactor_Partial
 */

XRL_EXTERN
xrlComplex Crystal_F_H_StructureFactor (Crystal_Struct* crystal, double energy,
                      int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle, xrl_error **error);

/*--------------------------------------------------------------------------------------------------
 * Compute F_H
 * See also Crystal_F_H_StructureFactor
 * The Atomic structure factor has three terms: F = f0 + f' + f''
 * For each of these three terms, there is a corresponding *_flag argument
 * which controls the numerical value used in computing F_H:
 *        *_flag = 0 --> Set this term to 0.
 *        *_flag = 1 --> Set this term to 1. Only used for f0.
 *        *_flag = 2 --> Set this term to the value given
 */
XRL_EXTERN
xrlComplex Crystal_F_H_StructureFactor_Partial (Crystal_Struct* crystal, double energy,
                      int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle,
                      int f0_flag, int f_prime_flag, int f_prime2_flag, xrl_error **error);

/*--------------------------------------------------------------------------------
 * Compute unit cell volume.
 * Note: Structures obtained from crystal array will have their volume in .volume.
 */

XRL_EXTERN
double Crystal_UnitCellVolume(Crystal_Struct* crystal, xrl_error **error);

/*--------------------------------------------------------------------------------
 * Compute d-spacing between planes.
 * This routine assumes that if crystal.volume is nonzero then it holds a valid value.
 * If (i, j, k) = (0, 0, 0) then zero is returned.
 */

XRL_EXTERN
double Crystal_dSpacing (Crystal_Struct* crystal, int i_miller, int j_miller, int k_miller, xrl_error **error);

/*--------------------------------------------------------------------------------
 * Add a new CrystalStruct to crystal_array.
 * The data is copied to crystal_array.
 * If the material already exists in the array then the existing material data is overwitten.
 * If crystal_array is NULL then the crystals are added to the official array of crystals.
 * Return: 1 on success and 0 on error.
 */

XRL_EXTERN
int Crystal_AddCrystal (Crystal_Struct* crystal, Crystal_Array* c_array, xrl_error **error);

/*--------------------------------------------------------------------------------
 * Read in a set of crystal structs to crystal_array.
 * If a material already exists in the array then the existing material data is overwitten.
 * If crystal_array is NULL then the crystals are added to the official array of crystals.
 * Return: 1 on success and 0 on error.
 */

XRL_EXTERN
int Crystal_ReadFile (const char* file_name, Crystal_Array* c_array, xrl_error **error);

/*--------------------------------------------------------------------------------
 * Returns a NULL-terminated array of strings containing the names of the crystals
 * in c_array. If c_array is NULL, then the builtin array of crystals will be used instead
 * If nCrystals is not NULL, it shall receive the number of crystalnames in the array.
 * The returned array should be freed firstly by using xrlFree to deallocate
 * all individual strings, and subsequently by using xrlFree to deallocate the array
 */

XRL_EXTERN
char **Crystal_GetCrystalsList(Crystal_Array *c_array, int *nCrystals, xrl_error **error);


#endif
