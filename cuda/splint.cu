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
#include "xraylib-cuda-private.h"

__device__ double splint_cu(double *xa, double *ya, double *y2a, int n, double x)
{
	int klo, khi, k;
	double h, b, a, y;

	if (x >= xa[n]) {
	  y = ya[n];
	  return y;
	}

	if (x <= xa[1]) {
	  y = ya[1];
	  return y;
	}

	klo = 1;
	khi = n;
	while (khi-klo > 1) {
		k = (khi + klo) >> 1;
		if (xa[k] > x) khi = k;
		else klo = k;
	}

	h = xa[khi] - xa[klo];
	if (h == 0.0) {
	  y = (ya[klo] + ya[khi])/2.0;
	  return y;
	}
	a = (xa[khi] - x) / h;
	b = (x - xa[klo]) / h;
	y = a*ya[klo] + b*ya[khi] + ((a*a*a-a)*y2a[klo]
	     + (b*b*b-b)*y2a[khi])*(h*h)/6.0;
	return y;
}
