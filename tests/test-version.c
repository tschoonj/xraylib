/* Copyright (c) 2018, Tom Schoonjans
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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int main(int argc, char **argv) {
	char *header_version = NULL;


#ifdef _WIN32
	int bytes_needed = _scprintf("%d.%d.%d", XRAYLIB_MAJOR, XRAYLIB_MINOR, XRAYLIB_MICRO);
	if (bytes_needed < 0)
		return 1;
	header_version = malloc((bytes_needed + 1) * sizeof(char));
	if (_snprintf(header_version, bytes_needed + 1, "%d.%d.%d", XRAYLIB_MAJOR, XRAYLIB_MINOR, XRAYLIB_MICRO) < 0) {
		return 1;
	}
#else
	if (asprintf(&header_version, "%d.%d.%d", XRAYLIB_MAJOR, XRAYLIB_MINOR, XRAYLIB_MICRO) < 0) {
		fprintf(stderr, "vasprintf error\n");
		return 1;
	}
#endif

	if (strcmp(header_version, PACKAGE_VERSION) != 0)
		return 1;

	return 0;
}
