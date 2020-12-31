/* Copyright (c) 2018-2019, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <config.h>
#include "xraylib.h"
#include "xraylib-aux.h"
#include "xraylib-error-private.h"
#ifdef NDEBUG
  #undef NDEBUG
#endif
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#ifdef HAVE_COMPLEX_H
#include <complex.h>
#endif

int main(int argc, char **argv) {
	xrl_error *error = NULL;
	xrlComplex xrlCplx_1, xrlCplx_2, xrlCplx_3;
#ifdef HAVE_COMPLEX_H
	double complex cplx;
#endif
	Crystal_Array *c_array;
	char **crystals_list;
	int nCrystals;
	int i;
	double tmp;
	int rv;
	double f0, f_prime, f_prime2;
	Crystal_Struct *cs, *cs_copy;
	int current_ncrystals;

	/* complex math tests */
#ifdef HAVE_COMPLEX_H
	xrlCplx_1.re = 5;
	xrlCplx_1.im = -3;
	assert(fabs(c_abs(xrlCplx_1) - cabs(5 - 3 * I)) < 1E-6);
	xrlCplx_2.re = -33;
	xrlCplx_2.im = 25;
	xrlCplx_3 = c_mul(xrlCplx_1, xrlCplx_2);
	cplx = (5 - 3 * I) * (-33 + 25 * I);
	assert(fabs(xrlCplx_3.re -  creal(cplx)) < 1E-6);
	assert(fabs(xrlCplx_3.im -  cimag(cplx)) < 1E-6);

#else
	fprintf(stderr, "Not checking the complex math...");
#endif	

	/* basic array stuff */
	/* let's do something naughty -> trigger malloc error */
	/* This actually works on my iMac and the Travis-CI macOS VMs! */
	/* c_array = Crystal_ArrayInit(INT32_MAX, &error);
	assert(c_array == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_MEMORY);
	printf("error->message: %s\n", error->message);
	xrl_clear_error(&error);
	*/
	
	c_array = Crystal_ArrayInit(-1, &error);
	assert(c_array == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	xrl_clear_error(&error);

	c_array = Crystal_ArrayInit(0, &error);
	assert(c_array != NULL);
	assert(error == NULL);
	assert(c_array->crystal == NULL);
	assert(c_array->n_crystal == 0);
	Crystal_ArrayFree(c_array);

	crystals_list = Crystal_GetCrystalsList(NULL, &nCrystals, &error);
	assert(crystals_list != NULL);
	assert(error == NULL);
	assert(nCrystals == 38);
	for (i = 0 ; crystals_list[i] != NULL ; i++) {
		cs = Crystal_GetCrystal(crystals_list[i], NULL, &error);
		assert(cs != NULL);
		assert(error == NULL);
		assert(strcmp(crystals_list[i], cs->name) == 0);
		free(crystals_list[i]);
	}
	free(crystals_list);
	assert(i == nCrystals);

	cs = Crystal_GetCrystal(NULL, NULL, &error);
	assert(cs == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	xrl_clear_error(&error);
	Crystal_Free(cs);

	cs = Crystal_GetCrystal("non-existent-crystal", NULL, &error);
	assert(cs == NULL);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	xrl_clear_error(&error);
	Crystal_Free(cs);

	/* copy struct */
	cs = Crystal_GetCrystal("Diamond", NULL, &error);
	assert(cs != NULL);
	assert(error == NULL);
	cs_copy = Crystal_MakeCopy(cs, &error);
	assert(error == NULL);
	assert(cs_copy != NULL);
	free(cs_copy->name);
	cs_copy->name = xrl_strdup("Diamond-copy");

	rv = Crystal_AddCrystal(cs, NULL, &error);
	assert(rv == 0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, "Crystal already present in array") == 0);
	xrl_clear_error(&error);
	Crystal_Free(cs);

	rv = Crystal_AddCrystal(cs_copy, NULL, &error);
	assert(rv == 1);
	assert(error == NULL);
	Crystal_Free(cs_copy);
	crystals_list = Crystal_GetCrystalsList(NULL, &nCrystals, &error);
	assert(crystals_list != NULL);
	assert(error == NULL);
	assert(nCrystals == 39);
	for (i = 0 ; crystals_list[i] != NULL ; i++) {
		cs = Crystal_GetCrystal(crystals_list[i], NULL, &error);
		assert(cs != NULL);
		assert(error == NULL);
		assert(strcmp(crystals_list[i], cs->name) == 0);
		free(crystals_list[i]);
		Crystal_Free(cs);
	}
	free(crystals_list);

	cs = Crystal_GetCrystal("Diamond", NULL, &error);
	assert(cs != NULL);
	assert(error == NULL);

	current_ncrystals = nCrystals;

	for (i = 0 ; i < CRYSTALARRAY_MAX ; i++) {
		char name[100];
		assert(snprintf(name, 100, "Diamond copy %d", i) > 0);
		cs_copy = Crystal_MakeCopy(cs, &error);
		free(cs_copy->name);
		cs_copy->name = xrl_strdup(name);
		rv = Crystal_AddCrystal(cs_copy, NULL, &error);
		if (current_ncrystals < CRYSTALARRAY_MAX) {
			assert(rv == 1);
			assert(error == NULL);
			Crystal_GetCrystalsList(NULL, &nCrystals, &error);
			assert(nCrystals == ++current_ncrystals);
		}
		else {
			assert(rv == 0);
			assert(error != NULL);
			assert(error->code == XRL_ERROR_RUNTIME);
			assert(strcmp(error->message, "Extending internal is crystal array is not allowed") == 0);
			xrl_clear_error(&error);
			Crystal_GetCrystalsList(NULL, &nCrystals, &error);
			assert(nCrystals == CRYSTALARRAY_MAX);
			assert(error == NULL);
		}
		Crystal_Free(cs_copy);
	}
	Crystal_Free(cs);

	/* bragg angle */
	cs = Crystal_GetCrystal("Diamond", NULL, &error);
	assert(cs != NULL);
	assert(error == NULL);
	tmp = Bragg_angle(cs, 10.0, 1, 1, 1, &error);
	assert(fabs(tmp - 0.3057795845795849) < 1E-6);
	assert(error == NULL);

	tmp = Bragg_angle(NULL, 10.0, 1, 1, 1, &error);
	assert(tmp == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, CRYSTAL_NULL) == 0);
	xrl_clear_error(&error);

	tmp = Bragg_angle(cs, -10.0, 1, 1, 1, &error);
	assert(tmp == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	tmp = Bragg_angle(cs, 10.0, 0, 0, 0, &error);
	assert(tmp == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_MILLER) == 0);
	xrl_clear_error(&error);

	/* Q_scattering_amplitude */
	tmp = Q_scattering_amplitude(cs, 10.0, 1, 1, 1, PI/4.0, &error);
	assert(error == NULL);
	assert(fabs(tmp - 0.19184445408324474) < 1E-6);

	tmp = Q_scattering_amplitude(NULL, 10.0, 1, 1, 1, PI/4, &error);
	assert(tmp == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, CRYSTAL_NULL) == 0);
	xrl_clear_error(&error);

	tmp = Q_scattering_amplitude(cs, -10.0, 1, 1, 1, PI/4, &error);
	assert(tmp == 0.0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	xrl_clear_error(&error);

	tmp = Q_scattering_amplitude(cs, 10.0, 0, 0, 0, PI/4, &error);
	assert(tmp == 0.0);
	assert(error == NULL);

	/* Atomic_Factors */
	rv = Atomic_Factors(26, 10.0, 1.0, 10.0, &f0, &f_prime, &f_prime2, &error);
	assert(rv == 1);
	assert(error == NULL);
	assert(fabs(f0 - 65.15) < 1E-6);
	assert(fabs(f_prime + 0.22193271025027966) < 1E-6);
	assert(fabs(f_prime2 - 22.420270655080493) < 1E-6);

	rv = Atomic_Factors(-1, 10.0, 1.0, 10.0, &f0, &f_prime, &f_prime2, &error);
	assert(rv == 0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, Z_OUT_OF_RANGE) == 0);
	assert(f0 == 0.0);
	assert(f_prime == 0.0);
	assert(f_prime2 == 0.0);
	xrl_clear_error(&error);
		
	rv = Atomic_Factors(26, -10, 1.0, 10.0, &f0, &f_prime, &f_prime2, &error);
	assert(rv == 0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_ENERGY) == 0);
	assert(f0 == 0.0);
	assert(f_prime == 0.0);
	assert(f_prime2 == 0.0);
	xrl_clear_error(&error);
		
	rv = Atomic_Factors(26, 10, -1.0, 10.0, &f0, &f_prime, &f_prime2, &error);
	assert(rv == 0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_Q) == 0);
	assert(f0 == 0.0);
	assert(f_prime == 0.0);
	assert(f_prime2 == 0.0);
	xrl_clear_error(&error);
		
	rv = Atomic_Factors(26, 10, 1.0, -10.0, &f0, &f_prime, &f_prime2, &error);
	assert(rv == 0);
	assert(error != NULL);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, NEGATIVE_DEBYE_FACTOR) == 0);
	assert(f0 == 0.0);
	assert(f_prime == 0.0);
	assert(f_prime2 == 0.0);
	xrl_clear_error(&error);
		
	rv = Atomic_Factors(26, 10, 1.0, 10.0, NULL, &f_prime, &f_prime2, &error);
	assert(rv == 1);
	assert(error == NULL);
	assert(f0 == 0.0);
	assert(f_prime != 0.0);
	assert(f_prime2 != 0.0);
		
	rv = Atomic_Factors(26, 10, 1.0, 10.0, &f0, NULL, &f_prime2, &error);
	assert(rv == 1);
	assert(error == NULL);
	assert(f0 != 0.0);
	assert(f_prime2 != 0.0);
		
	rv = Atomic_Factors(26, 10, 1.0, 10.0, &f0, &f_prime, NULL, &error);
	assert(rv == 1);
	assert(error == NULL);
	assert(f0 != 0.0);
	assert(f_prime != 0.0);
		
	/* unit cell volume */
	tmp = Crystal_UnitCellVolume(cs, &error);
	assert(error == NULL);
	assert(fabs(tmp - 45.376673902751) < 1E-6);

	tmp = Crystal_UnitCellVolume(NULL, &error);
	assert(error != NULL);
	assert(tmp == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, CRYSTAL_NULL) == 0);
	xrl_clear_error(&error);

	/* crystal dspacing */
	tmp = Crystal_dSpacing(cs, 1, 1, 1, &error);
	assert(error == NULL);
	assert(fabs(tmp - 2.0592870875248344) < 1E-6);

	tmp = Crystal_dSpacing(NULL, 1, 1, 1, &error);
	assert(error != NULL);
	assert(tmp == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, CRYSTAL_NULL) == 0);
	xrl_clear_error(&error);

	tmp = Crystal_dSpacing(cs, 0, 0, 0, &error);
	assert(error != NULL);
	assert(tmp == 0.0);
	assert(error->code == XRL_ERROR_INVALID_ARGUMENT);
	assert(strcmp(error->message, INVALID_MILLER) == 0);
	xrl_clear_error(&error);

	Crystal_Free(cs);

	/* TODO: Test Crystal_F_H_StructureFactor and Crystal_F_H_StructureFactor_Partial */

	return 0;
}
