/*
Copyright _cu(c) 2014, Tom Schoonjans and Antonio Brunetti
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans and Antonio Brunetti ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans and Antonio Brunetti BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES _cu(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT _cu(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "xraylib-cuda.h"
#include "xraylib-cuda-private.h"
#include "xraylib.h"

__device__ double CSb_Total_cu(int Z, double E)
{
  return CS_Total_cu(Z, E)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double CSb_Photo_cu(int Z, double E)
{
  return CS_Photo_cu(Z, E)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double CSb_Rayl_cu(int Z, double E)
{
  return CS_Rayl_cu(Z, E)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double CSb_Compt_cu(int Z, double E)
{
  return CS_Compt_cu(Z, E)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double CSb_FluorLine_cu(int Z, int line, double E)
{
  return CS_FluorLine_cu(Z, line, E)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double DCSb_Rayl_cu(int Z, double E, double theta)
{
  return DCS_Rayl_cu(Z, E, theta)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double DCSb_Compt_cu(int Z, double E, double theta)
{
  return DCS_Compt_cu(Z, E, theta)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double DCSPb_Rayl_cu(int Z, double E, double theta, double phi)
{
  return DCSP_Rayl_cu(Z, E, theta, phi)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

__device__ double DCSPb_Compt_cu(int Z, double E, double theta, double phi)
{
  return DCSP_Compt_cu(Z, E, theta, phi)*AtomicWeight_arr_d[Z]/AVOGNUM;
}

