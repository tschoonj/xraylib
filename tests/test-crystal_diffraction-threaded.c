/* Copyright (c) 2026, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/* Concurrency test for the process-global crystal array (the c_array=NULL path):
   many threads add uniquely-named crystals while others read the array. A
   missing/broken lock shows up here as a crash, a corrupt read, or a wrong
   final count. */

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

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#define N_THREADS 8
#define ADDS_PER_THREAD 40 /* 8 * 40 + 38 built-ins = 358 < CRYSTALARRAY_MAX (512) */

static Crystal_Struct *seed = NULL;

static void free_list(char **list, int n) {
	int i;
	for (i = 0 ; i < n ; i++)
		free(list[i]);
	free(list);
}

static void worker(int tid) {
	int i;
	for (i = 0 ; i < ADDS_PER_THREAD ; i++) {
		xrl_error *error = NULL;
		char name[100];
		char **list;
		int n;
		Crystal_Struct *cs, *cs_copy;

		list = Crystal_GetCrystalsList(NULL, &n, &error);
		assert(list != NULL);
		assert(error == NULL);
		assert(n >= 38);
		free_list(list, n);

		cs = Crystal_GetCrystal("Diamond", NULL, &error);
		assert(cs != NULL);
		assert(error == NULL);
		assert(strcmp(cs->name, "Diamond") == 0);
		Crystal_Free(cs);

		assert(snprintf(name, sizeof(name), "CThread-%d-%d", tid, i) > 0);
		cs_copy = Crystal_MakeCopy(seed, &error);
		assert(cs_copy != NULL);
		assert(error == NULL);
		free(cs_copy->name);
		cs_copy->name = xrl_strdup(name);
		assert(Crystal_AddCrystal(cs_copy, NULL, &error) == 1);
		assert(error == NULL);
		Crystal_Free(cs_copy);
	}
}

#ifdef _WIN32
static DWORD WINAPI thread_start(LPVOID arg) {
	worker((int)(intptr_t) arg);
	return 0;
}
#else
static void *thread_start(void *arg) {
	worker((int)(intptr_t) arg);
	return NULL;
}
#endif

int main(int argc, char **argv) {
	xrl_error *error = NULL;
	char **list;
	int nCrystals, start, i;
#ifdef _WIN32
	HANDLE threads[N_THREADS];
#else
	pthread_t threads[N_THREADS];
#endif

	seed = Crystal_GetCrystal("Diamond", NULL, &error);
	assert(seed != NULL);
	assert(error == NULL);

	list = Crystal_GetCrystalsList(NULL, &start, &error);
	assert(list != NULL);
	assert(error == NULL);
	free_list(list, start);
	assert(start + N_THREADS * ADDS_PER_THREAD < CRYSTALARRAY_MAX);

	for (i = 0 ; i < N_THREADS ; i++) {
#ifdef _WIN32
		threads[i] = CreateThread(NULL, 0, thread_start, (LPVOID)(intptr_t) i, 0, NULL);
		assert(threads[i] != NULL);
#else
		assert(pthread_create(&threads[i], NULL, thread_start, (void *)(intptr_t) i) == 0);
#endif
	}

	for (i = 0 ; i < N_THREADS ; i++) {
#ifdef _WIN32
		assert(WaitForSingleObject(threads[i], INFINITE) == WAIT_OBJECT_0);
		CloseHandle(threads[i]);
#else
		assert(pthread_join(threads[i], NULL) == 0);
#endif
	}

	list = Crystal_GetCrystalsList(NULL, &nCrystals, &error);
	assert(list != NULL);
	assert(error == NULL);
	assert(nCrystals == start + N_THREADS * ADDS_PER_THREAD);
	for (i = 0 ; i < nCrystals ; i++) {
		Crystal_Struct *cs = Crystal_GetCrystal(list[i], NULL, &error);
		assert(cs != NULL);
		assert(error == NULL);
		assert(strcmp(cs->name, list[i]) == 0);
		Crystal_Free(cs);
	}
	free_list(list, nCrystals);

	Crystal_Free(seed);

	return 0;
}
