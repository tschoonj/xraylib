/* Copyright (C) 2018-2020 Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans 'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "config.h"
#include "xraylib-aux.h"
#include "xraylib-error-private.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* xrl_strdup_vprintf(const char *format, va_list args) {
	char *rv = NULL;

#ifdef _WIN32
	int bytes_needed = _vscprintf(format, args);
	if (bytes_needed < 0)
		return NULL;
	rv = malloc((bytes_needed + 1) * sizeof(char));
	if (_vsnprintf(rv, bytes_needed + 1, format, args) < 0) {
		free(rv);
		return NULL;
	}
#else
	if (vasprintf(&rv, format, args) < 0) {
		return NULL;
	}
#endif
	return rv;
}

xrl_error* xrl_error_new_valist(xrl_error_code code, const char *format, va_list args) {
	xrl_error *error;

	if (format == NULL) {
		fprintf(stderr, "xrl_error_new_valist: format cannot be NULL!\n");
		return NULL;
	}

	error = malloc(sizeof(xrl_error));
	error->code = code;
	error->message = xrl_strdup_vprintf(format, args);

	return error;
}

xrl_error* xrl_error_new(xrl_error_code code, const char *format, ...) {
	xrl_error* error;
	va_list args;
	
	if (format == NULL) {
		fprintf(stderr, "xrl_error_new: format cannot be NULL!\n");
		return NULL;
	}

	va_start(args, format);
	error = xrl_error_new_valist(code, format, args);
	va_end(args);

	return error;
}

xrl_error* xrl_error_new_literal(xrl_error_code code, const char *message) {
	xrl_error *error = NULL;

	if (message == NULL) {
		fprintf(stderr, "xrl_error_new_literal: message cannot be NULL!\n");
		return NULL;
	}

	error = malloc(sizeof(xrl_error));
	error->code = code;
	error->message = xrl_strdup(message);

	return error;
}

void xrl_error_free(xrl_error *error) {
	if (error == NULL)
		return;

	if (error->message)
		free(error->message);

	free(error);
}

xrl_error* xrl_error_copy(const xrl_error *error) {
	xrl_error *copy = NULL;

	if (error == NULL)
		return NULL;

	copy = malloc(sizeof(xrl_error));

	*copy = *error;

	copy->message = NULL;
	if (error->message)
		copy->message = xrl_strdup(error->message);

	return copy;
}

int xrl_error_matches(const xrl_error *error, xrl_error_code code) {
	return error && error->code == code;
}

#define ERROR_OVERWRITTEN_WARNING "xrl_error set over the top of a previous xrl_error or uninitialized memory.\n" \
               "This indicates a bug in someone's code. You must ensure an error is NULL before it's set.\n" \
"The overwriting error message was: %s"

void xrl_set_error(xrl_error **err, xrl_error_code code , const char *format, ...) {
	xrl_error *new = NULL;

	va_list args;

	if (err == NULL)
		return;

	va_start(args, format);
	new = xrl_error_new_valist(code, format, args);
	va_end(args);

	if (*err == NULL)
		*err = new;
	else {
		fprintf(stderr, ERROR_OVERWRITTEN_WARNING, new->message);
		xrl_error_free(new);
	}
}

void xrl_set_error_literal(xrl_error **err, xrl_error_code code, const char *message) {
	if (err == NULL)
		return;

	if (*err == NULL)
		*err = xrl_error_new_literal(code, message);
	else
		fprintf(stderr, ERROR_OVERWRITTEN_WARNING, message);
}

void xrl_propagate_error(xrl_error **dest, xrl_error *src) {
	if (src == NULL) {
		fprintf(stderr, "xrl_propagate_error: src cannot be NULL");
		return;
	}

	if (dest == NULL) {
		if (src)
			xrl_error_free(src);
		return;
	}
	else {
		if (*dest != NULL) {
			fprintf(stderr, ERROR_OVERWRITTEN_WARNING, src->message);
			xrl_error_free(src);
		}
		else {
			*dest = src;
		}
	}
}

void xrl_clear_error(xrl_error **err) {
	if (err && *err) {
		xrl_error_free(*err);
		*err = NULL;
	}
}
