/* Copyright (C) 2018 Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans 'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef XRAYLIB_ERROR_H
#define XRAYLIB_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SWIG

typedef enum {
	XRL_ERROR_MEMORY, /* set in case of a memory allocation problem */
	XRL_ERROR_INVALID_ARGUMENT, /* set in case an invalid argument gets passed to a routine */
	XRL_ERROR_IO, /* set in case an error involving input/output occurred */
	XRL_ERROR_TYPE, /* set in case an error involving type conversion occurred */
	XRL_ERROR_UNSUPPORTED, /* set in case an unsupported feature has been requested */
	XRL_ERROR_RUNTIME /* set in case an unexpected runtime error occurred */
} xrl_error_code;


/**
 * xrl_error:
 * @code: error code, e.g. %XRL_ERROR_MEMORY
 * @message: human-readable informative error message
 *
 * The `xrl_error` structure contains information about
 * an error that has occurred.
 */
typedef struct _xrl_error xrl_error;

struct _xrl_error
{
  xrl_error_code code;
  char *message;
};

XRL_EXTERN
void xrl_error_free(xrl_error *error);

XRL_EXTERN
xrl_error* xrl_error_copy(const xrl_error *error);

XRL_EXTERN
int xrl_error_matches(const xrl_error *error, xrl_error_code code);

XRL_EXTERN
void xrl_propagate_error(xrl_error **dest, xrl_error *src);

XRL_EXTERN
void xrl_clear_error(xrl_error **err);

#endif

#ifdef __cplusplus
}
#endif

#endif
