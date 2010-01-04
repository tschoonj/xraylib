#include "xraylib-aux.h"
#include <stdlib.h>
#include <string.h>

#ifndef HAVE_STRDUP
char *strdup(const char *str) {
	char *dup= (char *)malloc( strlen(str)+1 );
	if (dup) strcpy(dup,str);
	return dup;
}
#endif

#ifndef HAVE_STRNDUP
char *strndup(const char *str, size_t len) {
	char *dup= (char *)malloc( len+1 );
	if (dup) {
		strncpy(dup,str,len);
		dup[len]= '\0';
	}
	return dup;
}
#endif
