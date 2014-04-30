/*
Copyright (c) 2014, Tom Schoonjans and Antonio Brunetti
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans and Antonio Brunetti ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans and Antonio Brunetti BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "xraylib-cuda.h"
#include "xraylib-defs.h"

__device__ double AtomicLevelWidth_arr_d[(ZMAX+1)*SHELLNUM];

__device__ double AtomicLevelWidth_cu(int Z, int shell) {
  double atomic_level_width;

  if (Z<1 || Z>ZMAX) {
    return 0;
  }
  if (shell<0 || shell>=SHELLNUM) {
    return 0;
  }
  atomic_level_width = AtomicLevelWidth_arr_d[Z*SHELLNUM+shell];

  if (atomic_level_width < 0.) {
    return 0;
  }

  return atomic_level_width;
}

