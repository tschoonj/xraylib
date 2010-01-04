#ifndef _XRAYLIB_AUX_H
#define _XRAYLIB_AUX_H

#include "config.h"

/*
 *
 *  The following two functions are ISO C extensions and are not available on all platforms.
 *  We will use our own functions if they are absent from libc.
 *
 */


#ifndef HAVE_STRDUP
extern char *strdup(const char *str);
#endif

#ifndef HAVE_STRNDUP
extern char *strndup(const char *str, size_t len);
#endif



#endif
