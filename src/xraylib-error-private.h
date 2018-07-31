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

xrl_error* xrl_error_new(enum xrl_error_code code, const char *format, ...) GNUC_PRINTF (2, 3);

xrl_error* xrl_error_new_literal(enum xrl_error_code code, const char *message);

xrl_error* xrl_error_new_valist(enum xrl_error_code code, const char *format, va_list args) GNUC_PRINTF(2, 0);

void xrl_set_error(xrl_error **err, enum xrl_error_code code , const char *format, ...) GNUC_PRINTF (3, 4);

void xrl_set_error_literal(xrl_error **err, enum xrl_error_code code, const char *message);

#endif

