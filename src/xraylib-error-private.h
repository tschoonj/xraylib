/* Copyright (C) 2018 Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans 'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef XRAYLIB_ERROR_PRIVATE_H
#define XRAYLIB_ERROR_PRIVATE_H

#include "xraylib.h"
#include "xraylib-error.h"
#include <stdarg.h>

/*
 *  This file is mostly copy-pasted from GLib's error methods...
 */ 

#if __GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ > 4)
#define GNUC_PRINTF( format_idx, arg_idx )    \
  __attribute__((__format__ (__printf__, format_idx, arg_idx)))
#else /* !__GNUC__ */
#define GNUC_PRINTF( format_idx, arg_idx )
#endif /* !__GNUC__ */

XRL_EXTERN
xrl_error* xrl_error_new(xrl_error_code code, const char *format, ...) GNUC_PRINTF (2, 3);

XRL_EXTERN
xrl_error* xrl_error_new_literal(xrl_error_code code, const char *message);

XRL_EXTERN
xrl_error* xrl_error_new_valist(xrl_error_code code, const char *format, va_list args) GNUC_PRINTF(2, 0);

XRL_EXTERN
void xrl_set_error(xrl_error **err, xrl_error_code code , const char *format, ...) GNUC_PRINTF (3, 4);

XRL_EXTERN
void xrl_set_error_literal(xrl_error **err, xrl_error_code code, const char *message);

/* predefined error messages  */
#define Z_OUT_OF_RANGE "Z out of range"
#define NEGATIVE_ENERGY "Energy must be strictly positive"
#define NEGATIVE_DENSITY "Density must be strictly positive"
#define NEGATIVE_Q "q must be positive"
#define NEGATIVE_PZ "pz must be positive"
#define INVALID_SHELL "Invalid shell for this atomic number"
#define INVALID_LINE "Invalid line for this atomic number"
#define INVALID_CK "Invalid Coster-Kronig transition for this atomic number"
#define INVALID_AUGER "Invalid Auger transition macro for this atomic number"
#define UNKNOWN_SHELL "Unknown shell macro provided"
#define UNKNOWN_LINE "Unknown line macro provided"
#define UNKNOWN_CK "Unknown Coster-Kronig transition macro provided"
#define UNKNOWN_AUGER "Unknown Auger transition macro provided"
#define UNAVAILABLE_JUMP_FACTOR "Jump factor unavailable for element and shell"
#define UNAVAILABLE_FLUOR_YIELD "Fluorescence yield unavailable for atomic number and shell"
#define TOO_LOW_EXCITATION_ENERGY "The excitation energy too low to excite the shell"
#define UNAVAILABLE_PHOTO_CS "Photoionization cross section unavailable for atomic number and energy"
#define UNAVAILABLE_RAD_RATE "Radiative rate unavailable for this atomic number and line macro"
#define UNAVAILABLE_CK "Coster-Kronig transition probability unavailable for this atomic number and transition macro"
#define UNKNOWN_COMPOUND "Compound is not a valid chemical formula and is not present in the NIST compound database"
#define MALLOC_ERROR "Could not allocate memory: %s"
#define INVALID_MILLER "Miller indices cannot all be zero"
#define NEGATIVE_DEBYE_FACTOR "Debye-Waller factor must be strictly positive"
#define CRYSTAL_NULL "Crystal cannot be NULL"
#define SPLINT_X_TOO_LOW "Spline extrapolation is not allowed"
#define SPLINT_X_TOO_HIGH "Spline extrapolation is not allowed"
#define LININTERP_X_TOO_LOW "Linear extrapolation is not allowed"
#define LININTERP_X_TOO_HIGH "Linear extrapolation is not allowed"

#endif

