#Copyright (c) 2009-2020 Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



AC_INIT([xraylib],[4.2.0],[Tom.Schoonjans@me.com])
AC_PREREQ([2.60])
AC_CONFIG_SRCDIR([include/xraylib.h])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])
AM_SILENT_RULES([yes])
AC_CANONICAL_HOST


LIB_CURRENT=12
LIB_REVISION=1
LIB_AGE=1
AC_SUBST(LIB_CURRENT)
AC_SUBST(LIB_REVISION)
AC_SUBST(LIB_AGE)
LIB_CURRENT_MINUS_AGE=`expr $LIB_CURRENT - $LIB_AGE`
AC_SUBST(LIB_CURRENT_MINUS_AGE)
AC_CONFIG_MACRO_DIR([m4])


AC_USE_SYSTEM_EXTENSIONS

m4_ifdef([AM_PROG_AR],[AM_PROG_AR])
#at least version 2.0 of libtool is required for creating the fortran bindings
LT_PREREQ([2.0.0])
#dlopen is necessary for IDL bindings
LT_INIT([dlopen disable-fast-install win32-dll disable-static])

AC_PROG_SED

AC_PROG_CC
AM_PROG_CC_C_O
#this next line may never be reached...
if test -z $CC ; then
	AC_MSG_ERROR([no C compiler was found on the system.])
fi

AC_CANONICAL_HOST

WSTRICT_CFLAGS=
AX_CHECK_COMPILE_FLAG([-Werror=deprecated],[WSTRICT_CFLAGS="${WSTRICT_CFLAGS} -Werror=deprecated"],,)
AX_CHECK_COMPILE_FLAG([-Werror=deprecated-declarations],[WSTRICT_CFLAGS="${WSTRICT_CFLAGS} -Werror=deprecated-declarations"],,)
AX_CHECK_COMPILE_FLAG([-Werror=implicit],[WSTRICT_CFLAGS="${WSTRICT_CFLAGS} -Werror=implicit"],,)
AX_CHECK_COMPILE_FLAG([-Werror=unused-function],[WSTRICT_CFLAGS="${WSTRICT_CFLAGS} -Werror=unused-function"],,)
AX_CHECK_COMPILE_FLAG([-Werror=parentheses],[WSTRICT_CFLAGS="${WSTRICT_CFLAGS} -Werror=parentheses"],,)
AX_CHECK_COMPILE_FLAG([-Werror=unused-result],[WSTRICT_CFLAGS="${WSTRICT_CFLAGS} -Werror=unused-result"],,)
# I can't compile the SWIG generated bindings with these on...
# see also https://github.com/swig/swig/issues/1278
#AX_CHECK_COMPILE_FLAG([-Werror=missing-declarations],[WSTRICT_CFLAGS="${WSTRICT_CFLAGS} -Werror=missing-declarations"],,)
#AX_CHECK_COMPILE_FLAG([-Werror=shadow],[WSTRICT_CFLAGS="${WSTRICT_CFLAGS} -Werror=shadow"],,)
#AX_CHECK_COMPILE_FLAG([-Werror=missing-prototypes],[WSTRICT_CFLAGS="${WSTRICT_CFLAGS} -Werror=missing-prototypes"],,)

AC_SUBST(WSTRICT_CFLAGS)

#check if libxrl should be built as a convenience library
AC_ARG_ENABLE([libxrl],[AS_HELP_STRING([--disable-libxrl],[do not build libxrl separately, link it statically into bindings])],[enable_libxrl=$enableval],[enable_libxrl=check])

AM_CONDITIONAL([LIBXRL_CONVENIENCE_BUILD], [test x$enable_libxrl = xno])

#headers check
AC_CHECK_HEADERS_ONCE([math.h stdio.h stdlib.h string.h ctype.h stddef.h locale.h complex.h])

#
#Some code is necessary in case we are cross compiling with mingw on a platform different from cygwin -> Wine is necessary!
#
WINE=
DISABLE_BINDINGS=no
CROSS_COMPILING=
LDFLAGS_LIBXRL=


AC_ARG_ENABLE([output-def], [AS_HELP_STRING([--disable-output-def],[build without the library definitions file (Windows only)])],[enable_output_def=$enableval],[enable_output_def=check] )


#cross_compiling variable is not reliable on modern bash shells due to wine integration... look at the compiler instead
#check instead of $build and $host differ

OS_WINDOWS_32=0
OS_WINDOWS_64=0
OS_WINDOWS=0
if test "x$host" != "x$build" ; then
	#cross compilation detected
	#if $host is mingw, look for wine
	case "$host" in
		i686-*mingw*)
			#disable bindings
			DISABLE_BINDINGS=yes
			#look for wine
			AC_CHECK_PROGS([WINE],[wine],["nowine"])
			;;
		x86_64-*mingw*)
			#disable bindings
			DISABLE_BINDINGS=yes
			#look for wine
			AC_CHECK_PROGS([WINE],[wine64],["nowine"])
			;;
		*)
			AC_MSG_ERROR([A platform was detected that is not supported for cross-compilation])
			;;
	esac

	if test "x$WINE" = "xnowine" ; then
		AC_MSG_ERROR([wine is necessary when cross-compiling for windows.])
	fi
	CROSS_COMPILING=yes
	LDFLAGS_LIBXRL="-no-undefined"
	if test "x$enable_output_def" != xno ; then
		LDFLAGS_LIBXRL+=" -Wl,--output-def,libxrl-$LIB_CURRENT_MINUS_AGE.def "
	fi
	LDFLAGS_FORTRAN="-no-undefined"
	LDFLAGS_PYTHON="-no-undefined -Wl,-subsystem,windows"
else
	case "$host" in
		i686-*mingw*)
			#build dll
			LDFLAGS_LIBXRL="-no-undefined"
			if test "x$enable_output_def" != xno ; then
				LDFLAGS_LIBXRL+=" -Wl,--output-def,libxrl-$LIB_CURRENT_MINUS_AGE.def "
			fi
			OS_WINDOWS_32=1
			OS_WINDOWS=1
			LDFLAGS_FORTRAN="-no-undefined"
			LDFLAGS_IDL="-no-undefined"
			LDFLAGS_PYTHON="-no-undefined -Wl,-subsystem,windows"
			;;
		x86_64-*mingw*)
			#build dll
			LDFLAGS_LIBXRL="-no-undefined"
			if test "x$enable_output_def" != xno ; then
				LDFLAGS_LIBXRL+=" -Wl,--output-def,libxrl-$LIB_CURRENT_MINUS_AGE.def "
			fi
			OS_WINDOWS_64=1
			OS_WINDOWS=1
			LDFLAGS_FORTRAN="-no-undefined"
			LDFLAGS_IDL="-no-undefined"
			LDFLAGS_PYTHON="-no-undefined -Wl,-subsystem,windows"
			;;
	esac
fi
AM_CONDITIONAL([OS_WINDOWS_64],[test x$OS_WINDOWS_64 = x1])
AM_CONDITIONAL([OS_WINDOWS_32],[test x$OS_WINDOWS_32 = x1])
AM_CONDITIONAL([OS_WINDOWS],[test x$OS_WINDOWS = x1])
AC_SUBST(OS_WINDOWS)
AC_SUBST(OS_WINDOWS_32)
AC_SUBST(OS_WINDOWS_64)

if test $OS_WINDOWS = 1 ; then
	if test x$enable_static = xyes -a x$enable_shared = xyes; then
		AC_MSG_ERROR([Can not build both shared and static at the same time on Windows.])
	fi
fi

AC_CHECK_FUNCS([strndup strdup _strdup]) # if not found, we use our own implementation

if test $OS_WINDOWS = 1 ; then
AC_CHECK_FUNC([_vscprintf], [], [AC_MSG_ERROR([_vscprintf must be present on the system])])
AC_CHECK_FUNC([_scprintf], [], [AC_MSG_ERROR([_scprintf must be present on the system])])
AC_CHECK_FUNC([_vsnprintf], [], [AC_MSG_ERROR([_vsnprintf must be present on the system])])
AC_CHECK_FUNC([_snprintf], [], [AC_MSG_ERROR([_snprintf must be present on the system])])
else
AC_CHECK_FUNC([vasprintf], [], [AC_MSG_ERROR([vasprintf must be present on the system])])
AC_CHECK_FUNC([asprintf], [], [AC_MSG_ERROR([asprintf must be present on the system])])
fi

AC_SUBST(LDFLAGS_LIBXRL)
AC_SUBST(LDFLAGS_FORTRAN)
AC_SUBST(LDFLAGS_IDL)
AC_SUBST(LDFLAGS_PYTHON)

# Detect if we need -lm
LT_LIB_M

# Symbol visibility handling.
#
# Taken from gtksourceview and modified where necessary
HIDDEN_VISIBILITY_CFLAGS=""
case "$host" in
    *-*-mingw*)
        dnl on mingw32 we do -fvisibility=hidden and __declspec(dllexport)
        AC_DEFINE([XRL_EXTERN], [__attribute__((visibility("default"))) __declspec(dllexport) extern],
                  [defines how to decorate public symbols while building])
        HIDDEN_VISIBILITY_CFLAGS="-fvisibility=hidden"
        ;;
    *)
        dnl on other compilers, check if we can do -fvisibility=hidden
        SAVED_CFLAGS="${CFLAGS}"
        CFLAGS="-fvisibility=hidden"
        AC_MSG_CHECKING([for -fvisibility=hidden compiler flag])
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]], [[return 0]])],[AC_MSG_RESULT(yes)
                       enable_fvisibility_hidden=yes],[AC_MSG_RESULT(no)
                       enable_fvisibility_hidden=no])
        CFLAGS="${SAVED_CFLAGS}"

        AS_IF([test "${enable_fvisibility_hidden}" = "yes"], [
            AC_DEFINE([XRL_EXTERN], [__attribute__((visibility("default"))) extern],
                      [defines how to decorate public symbols while building])
            HIDDEN_VISIBILITY_CFLAGS="-fvisibility=hidden"
        ])
        ;;
esac
AC_SUBST(HIDDEN_VISIBILITY_CFLAGS)

AM_CONDITIONAL([ENABLE_CROSS],[test x$CROSS_COMPILING = xyes])

AC_ARG_ENABLE([all-bindings],[AS_HELP_STRING([--disable-all-bindings],[build without bindings])],[enable_bindings=$enableval],[enable_bindings=check])

if test "x$enable_bindings" = xno ; then
	DISABLE_BINDINGS=yes
fi

#
#fortran 2003 bindings
#

AC_ARG_ENABLE([fortran2003], [AS_HELP_STRING([--disable-fortran2003],[build without the Fortran 2003 bindings])],[enable_fortran2003=$enableval],[enable_fortran2003=check] )




VALID_FORTRAN=


if test "x$enable_fortran2003" != xno && test "x$DISABLE_BINDINGS" = xno ; then
#check for fortran 2003 compiler
	AC_PROG_FC()
	if test -z $FC && test "x$enable_fortran2003" = xyes; then
		AC_MSG_ERROR([no fortran compiler was found on the system.])
	elif test -z $FC && test "x$enable_fortran2003" = xcheck; then
		AC_MSG_WARN([no fortran compiler was found on the system. The fortran bindings will not be built.])
		VALID_FORTRAN=no
	else
	AC_FC_SRCEXT(f90,[
	#check if it supports the required fortran 2003 features -> more thorough testing is certainly possible but looks a bit like overkill to me (Tom Schoonjans)
		AC_MSG_CHECKING([whether the fortran compiler supports the 2003 features])
		AC_LANG_PUSH(Fortran)
		AC_COMPILE_IFELSE([[
MODULE f2003_test
USE, INTRINSIC ::ISO_C_BINDING

TYPE, BIND(C) :: test_C
	INTEGER (C_INT) :: arrayLen
	TYPE (C_PTR) :: cpointer
ENDTYPE

TYPE :: test_F
	INTEGER (C_INT), DIMENSION(:), POINTER :: fpointer
ENDTYPE

INTERFACE
	FUNCTION foo (bar,morebar) BIND(C,NAME='foo')
		USE, INTRINSIC ::ISO_C_BINDING
		IMPLICIT NONE
		REAL (KIND=C_DOUBLE) :: foo
		INTEGER (KIND=C_INT), INTENT(IN) :: bar
		REAL (KIND=C_DOUBLE), INTENT(IN) :: morebar
	ENDFUNCTION foo
	FUNCTION strlen(s) BIND(C,NAME='strlen')
		USE, INTRINSIC ::ISO_C_BINDING
		IMPLICIT NONE
		CHARACTER (KIND=C_CHAR),DIMENSION(*) :: s
		INTEGER (C_SIZE_T) :: strlen
	ENDFUNCTION
ENDINTERFACE
ENDMODULE f2003_test
PROGRAM f2003_main
USE f2003_test
IMPLICIT NONE

CHARACTER (LEN=10,KIND=C_CHAR) :: string = C_CHAR_'123456789' // C_NULL_CHAR
TYPE (test_C) :: tester_C
TYPE (test_F) :: tester_F



IF (strlen(string) /= 9) THEN
	CALL EXIT(1)
ELSE
	CALL EXIT(0)
ENDIF

!next line should produce a compile-time error when using g95
CALL C_F_POINTER(tester_C%cpointer,tester_F%fpointer,[tester_C%arrayLen])

!use c_loc
tester_C%cpointer = C_LOC(tester_F%fpointer(1))


ENDPROGRAM f2003_main
	]],[VALID_FORTRAN=yes],[VALID_FORTRAN=no] )
		AC_LANG_POP(Fortran)
		rm f2003_test.mod
		AC_MSG_RESULT([$VALID_FORTRAN])
		if test "x$enable_fortran2003" != xcheck && test "x$VALID_FORTRAN" = xno; then
			AC_MSG_ERROR([--enable-fortran2003 was given but no compiler that supports the required fortran 2003 features was found on the system.])
		elif test "x$enable_fortran2003" = xcheck && test "x$VALID_FORTRAN" = xno; then
			AC_MSG_WARN([no suitable fortran 2003 compiler has been detected. The fortran bindings will not be built.])
		else
			AC_MSG_NOTICE([Building with Fortran 2003 bindings])
		fi
	],[
#if no compiler was found that supports f90 files -> disable fortran bindings compilation IF no explicit request after them was detected
        if test "x$enable_fortran2003" = xcheck ; then
		AC_MSG_WARN([no compiler supporting f90 extensions was found on the system. The fortran bindings will not be built.])
		VALID_FORTRAN=no
	elif test "x$enable_fortran2003" = xyes ; then
		AC_MSG_ERROR([--enable-fortran2003 was given but no compiler that supports f90 extensions was found on the system.])
	fi

	])
	fi
else
VALID_FORTRAN=no
fi

AM_CONDITIONAL([ENABLE_FORTRAN],[test x$VALID_FORTRAN = xyes])

#
#idl bindings
#


AC_ARG_ENABLE([idl],[AS_HELP_STRING([--disable-idl],[build without the idl bindings])],[enable_idl=$enableval],[enable_idl=check])

VALID_IDL=
ARCHFLAGS=

if test "x$enable_idl" != xno && test "x$DISABLE_BINDINGS" = xno  ; then
#search for header and libraries
#inspired by the gpulib autoconf macros...

	AC_ARG_WITH([idl-bindir],[AS_HELP_STRING([--with-idl-bindir],[set location of the idl binary])],[RSIIDL_BINDIR=$withval])

	if test -z "$RSIIDL_BINDIR" ; then
		#no location was presented as an option -> search the usual suspects...
		IDL_PATH="/usr/local/itt/idl/bin:/usr/local/rsi/idl/bin:/usr/local/pkg/graphics/rsi/idl/bin:/Applications/exelis/idl/bin:/usr/local/exelis/idl/bin:/Applications/itt/idl/bin:/usr/local/idl/bin:/Applications/itt/idl/idl/bin:/usr/local/itt/idl/idl/bin:/Applications/rsi/idl/bin:/c/ITT/IDL71/bin/bin.x86"
		AC_PATH_PROGS([RSIIDL_BIN], [idl],[],[$IDL_PATH])
		if test -z "$RSIIDL_BIN" && test "x$enable_idl" = xyes ; then
			#binaries not found while they were required through the enable-idl option
			AC_MSG_ERROR([IDL binaries not found in $IDL_PATH. Use --with-idl-bindir to set the location of the idl binary])
		elif test -z "$RSIIDL_BIN" && test "x$enable_idl" = xcheck ; then
			#binaries were not found but they were not requested explicitally through an option ->
			AC_MSG_WARN([IDL binaries not found in $IDL_PATH. Use --with-idl-bindir to set the location of the idl binary])
			VALID_IDL=no
		else
			#binaries were found...
		#	AC_MSG_RESULT([yes, in $IDL_PATH])
			RSIIDL_BINDIR=`AS_DIRNAME(["$RSIIDL_BIN"])`
		fi
	else
		#test if the user-supplied value contains the idl binary
		AC_PATH_PROGS([RSIIDL_BIN], [idl],[],[$RSIIDL_BINDIR])
		if test -z "$RSIIDL_BIN" ; then
			AC_MSG_ERROR([IDL binaries not found user-supplied $RSIIDL_BINDIR. Wrong value for --with-idl-bindir])
			VALID_IDL=no
		fi
	fi
	#moving on...
	if test -n "$RSIIDL_BINDIR" && test "x$VALID_IDL" != xno ; then
		RSIIDL_DIR="`AS_DIRNAME(["$RSIIDL_BINDIR"])`"
		if test "`AS_BASENAME(["$RSIIDL_DIR"])`" = "bin" ; then
			RSIIDL_DIR=`AS_DIRNAME(["$RSIIDL_DIR"])`
		fi
		#if this test succeeds then using version 5.6 or greater
		RSIIDL_INCDIR="$RSIIDL_DIR/external/include"
		AC_CHECK_FILE(["$RSIIDL_INCDIR/idl_export.h"],[RSIIDL_HASINC=yes],[RSIIDL_HASINC=no])

		#for older versions this would yield the header file
		if test "x$RSIIDL_HASINC" = xno ; then
			RSIIDL_INCDIR="$RSIIDL_DIR/external"
			AC_CHECK_FILE(["$RSIIDL_INCDIR/export.h"],[RSIIDL_HASINC=yes],[RSIIDL_HASINC=no])
		fi
		if test "x$RSIIDL_HASINC" = xno ; then
			AC_MSG_WARN([IDL header file was not found in $RSIIDL_INCDIR. This most likely indicates a problem with the IDL installation.])
			VALID_IDL=no
		fi
		#check for path to libidl.so
		if test "x$VALID_IDL" != xno ; then
  		case "$host" in

		    x86_64-*-linux*)
		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.linux.x86_64
		      ;;

		    *-linux*)
		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.linux.x86
		      ;;

		    *-sgi*)
#		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.sgi
		      AC_MSG_WARN([IRIX is currently not supported for the IDL bindings])
			VALID_IDL=no
		      ;;
		    x86_64-*-darwin*)
			if test -d $RSIIDL_BINDIR/bin.darwin.x86_64 ; then
		      		IDL_LIBBIN=$RSIIDL_BINDIR/bin.darwin.x86_64
			else
				AC_MSG_WARN([You appear to be running an IDL version on Mac OS X that does not have the required 64-bit IDL libraries. Please upgrade your IDL installation to at least version 7.1 if you would like to create the IDL bindings])
			VALID_IDL=no
		        fi
		      ;;



		    i386-*-darwin*)
	    		IDL_LIBBIN=$RSIIDL_BINDIR
		      ;;

		    ppc*-darwin*)
		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.darwin.ppc
		      ;;
		   *-solaris-*)
		      AC_MSG_WARN([Solaris is currently not supported for the IDL bindings])
			VALID_IDL=no
		      ;;
		   *-mingw32)
			IDL_LIBBIN=$RSIIDL_BINDIR
			#IDL_LIBADD="$RSIIDL_BINDIR/idl.lib"
			IDL_LIBADD="-L$RSIIDL_BINDIR -lidl"
			AC_SUBST(IDL_LIBADD)
			;;
		   *)
		      AC_MSG_WARN([Could not detect platform for IDL.])
			VALID_IDL=no
		      ;;

		  esac

		if test "x$VALID_IDL" != xno ; then
		AC_ARG_WITH([idl-libdir],[AS_HELP_STRING([--with-idl-libdir],[set location of the idl libraries])],[RSIIDL_LIBBIN=$withval],[RSIIDL_LIBBIN="$IDL_LIBBIN"])
		AC_CHECK_FILE([$IDL_LIBBIN/libidl.so],[RSIIDL_HASBIN_SO=yes],[RSIIDL_HASBIN_SO=no])
		AC_CHECK_FILE([$IDL_LIBBIN/libidl.dylib],[RSIIDL_HASBIN_DY=yes],[RSIIDL_HASBIN_DY=no])
		AC_CHECK_FILE([$IDL_LIBBIN/idl.dll],[RSIIDL_HASBIN_DLL=yes],[RSIIDL_HASBIN_DLL=no])
		if test "x$RSIIDL_HASBIN_SO" = xno && test "x$RSIIDL_HASBIN_DY" = xno && test "x$RSIIDL_HASBIN_DLL" = xno ; then
			AC_MSG_WARN([libidl.so, libidl.dylib or idl.dll could not be found. This means that you have either a corrupt or custom installation of idl. In the first case, set --with-idl-libdir to match the location of the IDL libraries])
			VALID_IDL=no
		fi
		if test "x$VALID_IDL" != xno ; then
		VALID_IDL=yes
		AC_PROG_MKDIR_P
		AC_PROG_INSTALL
		AC_SUBST(RSIIDL_BINDIR)
		AC_SUBST(RSIIDL_DIR)
		AC_SUBST(RSIIDL_INCDIR)
		AC_SUBST(RSIIDL_LIBBIN)
		AC_SUBST(IDL_CFLAGS)
		AC_SUBST(IDL_LD_FLAGS)
		AC_MSG_NOTICE([Building with IDL bindings])
		fi
		fi
	fi
	fi
fi #test "x$enable_idl" != xno

AC_SUBST(ARCHFLAGS)

AM_CONDITIONAL([ENABLE_IDL],[test x$VALID_IDL = xyes])

#search for swig which is necessary for both the perl and python bindings
AC_CHECK_PROGS([SWIG],[swig],["noswig"])


#
#perl bindings
#

AC_ARG_ENABLE([perl],[AS_HELP_STRING([--disable-perl],[build without the perl bindings])],[enable_perl=$enableval],[enable_perl=check])

VALID_PERL=

if test "x$SWIG" = xnoswig && test "x$enable_perl" = xyes && test "x$DISABLE_BINDINGS" = xno  ; then
	AC_MSG_ERROR([--enable-perl was given as an option but swig was not found on the system])
fi

if test "x$enable_perl" != xno && test "x$SWIG" != xnoswig && test "x$DISABLE_BINDINGS" = xno  ;  then
	#search for perl executable (although I can't imagine a modern system without perl...)
	if test -z $PERL ; then
		AC_CHECK_PROGS(PERL,[perl perl5],["noperl"])
	fi

	if test "x$PERL" = xnoperl  ; then
		if test "x$enable_perl" = xyes ; then
			AC_MSG_ERROR([--enable-perl was given as an option but Perl was not found on the system.])
		else
			AC_MSG_WARN([perl was not found on the system. The perl bindings will not be built.])
			VALID_PERL=no
		fi
	else
			#everything ok -> let's build those bindings
			AX_PERL_EXT
	  		AC_MSG_CHECKING([consistency of all components of perl development environment])
			ac_save_CFLAGS="$CFLAGS"
			CFLAGS="-I$PERL_EXT_INC $PERL_EXT_CPPFLAGS"
	       		AC_LANG_PUSH([C])
			AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
				#include <EXTERN.h>
				#include <perl.h>
				static PerlInterpreter *my_perl;
			]], [[
        		my_perl = perl_alloc();
        		perl_construct(my_perl);
			PL_exit_flags |= PERL_EXIT_DESTRUCT_END;
        		perl_run(my_perl);
        		perl_destruct(my_perl);
        		perl_free(my_perl);
			]])],[perlexists=yes],[perlexists=no])
			AC_MSG_RESULT([$perlexists])
			AC_LANG_POP
			CFLAGS="$ac_save_CFLAGS"
			VALID_PERL=$perlexists
	fi
else
VALID_PERL=no
fi #test enable perl

if test x$VALID_PERL = xyes ; then
	AC_MSG_NOTICE([Building with Perl bindings])
elif test x$VALID_PERL = xno && test "x$enable_perl" = xyes && test "x$DISABLE_BINDINGS" = xno ; then
	AC_MSG_ERROR([Cannot build perl bindings])
elif test x$VALID_PERL = xno && test "x$enable_perl" = xcheck && test "x$DISABLE_BINDINGS" = xno ; then
	AC_MSG_WARN([Cannot build perl bindings])
fi
AM_CONDITIONAL([ENABLE_PERL],[test x$VALID_PERL = xyes])

#
#Python bindings
#


AC_ARG_ENABLE([python],[AS_HELP_STRING([--disable-python],[build without the python bindings])],[enable_python=$enableval],[enable_python=check])

VALID_PYTHON=
PYTHON_FOUND=

if test "x$SWIG" = xnoswig && test "x$enable_python" = xyes && test "x$DISABLE_BINDINGS" = xno  ; then
	#don't even bother when swig is not found
	AC_MSG_ERROR([--enable-python was given as an option but swig was not found on the system])
elif test "x$SWIG" != xnoswig && test "x$enable_python" != xno && test "x$DISABLE_BINDINGS" = xno  ; then
	#verify the python installation
	AM_PATH_PYTHON([3.6], [PYTHON_FOUND=true],[PYTHON_FOUND=false])
	if test "x$PYTHON_FOUND" = xtrue ; then
		PYTHON_CPPFLAGS=
		PYTHON_LDFLAGS=
		AX_PYTHON_DEVEL
		if test "x$PYTHON" = x ; then
			if test "x$enable_python" = xyes ; then
				AC_MSG_ERROR([Incomplete python development package])
			else
				AC_MSG_WARN([Incomplete python development package])
			fi
			VALID_PYTHON=no
		else
			# check for numpy!
			AC_MSG_CHECKING([for numpy])
			res=`$PYTHON -c "import numpy" &>/dev/null`
			NUMPY=$?
			if test x$NUMPY = x0 ; then
				AC_MSG_RESULT([yes])
				NUMPY_HEADERS_PATH=`$PYTHON -c "from __future__ import print_function; import numpy; \
					print(numpy.get_include())"`
				if test $OS_WINDOWS = 1 ; then
					NUMPY_HEADERS_PATH=`cygpath -u $NUMPY_HEADERS_PATH`
				fi
				NUMPY_HEADERS=-I$NUMPY_HEADERS_PATH
				AC_SUBST(NUMPY_HEADERS)
				VALID_PYTHON=yes
				AC_MSG_NOTICE([Building with Python bindings])
			else
				VALID_PYTHON=no
				AC_MSG_RESULT([no])
				if test "x$enable_python" = xyes ; then
					AC_MSG_ERROR([Incomplete python development package: numpy not installed])
				else
					AC_MSG_WARN([Incomplete python development package: numpy not installed])
				fi
			fi
		fi

	fi
fi




AM_CONDITIONAL([ENABLE_PYTHON],[test x$VALID_PYTHON = xyes])


#
#Python numpy bindings using cython
#
AC_ARG_ENABLE([python-numpy],[AS_HELP_STRING([--disable-python-numpy],[build without the python-numpy bindings])],[enable_python_np=$enableval],[enable_python_np=check])
#for this we need to check for cython and for numpy, and python itself obviously
VALID_PYTHON_NUMPY=
AC_ARG_VAR(CYTHON, [the Cython command])
if test "x$enable_python_np" != xno && test  "x$DISABLE_BINDINGS" = xno  ; then
	##check for python if not done yet
	if test "x$VALID_PYTHON" != xyes ; then
		AM_PATH_PYTHON([3.6], [PYTHON_FOUND=true],[PYTHON_FOUND=false])
		if test "x$PYTHON_FOUND" = xtrue ; then
			PYTHON_CPPFLAGS=
			PYTHON_LDFLAGS=
			AX_PYTHON_DEVEL
			if test "x$PYTHON" = x ; then
				if test "x$enable_python_np" = xyes ; then
					AC_MSG_ERROR([Incomplete python development package])
				else
					AC_MSG_WARN([Incomplete python development package])
				fi
				VALID_PYTHON_EXE=no
			else
				# check for numpy!
				AC_MSG_CHECKING([for numpy])
				res=`$PYTHON -c "import numpy" &>/dev/null`
				NUMPY=$?
				if test x$NUMPY = x0 ; then
					AC_MSG_RESULT([yes])
					NUMPY_HEADERS_PATH=`$PYTHON -c "from __future__ import print_function; import numpy; \
						print(numpy.get_include())"`
					if test $OS_WINDOWS = 1 ; then
						NUMPY_HEADERS_PATH=`cygpath -u $NUMPY_HEADERS_PATH`
					fi
					NUMPY_HEADERS=-I$NUMPY_HEADERS_PATH
					AC_SUBST(NUMPY_HEADERS)
					VALID_PYTHON_EXE=yes
				else
					VALID_PYTHON_EXE=no
					AC_MSG_RESULT([no])
					if test "x$enable_python_np" = xyes ; then
						AC_MSG_ERROR([Incomplete python development package: numpy not installed])
					else
						AC_MSG_WARN([Incomplete python development package: numpy not installed])
					fi
				fi
			fi
		fi
	else
		VALID_PYTHON_EXE=yes
	fi
	if test "x$VALID_PYTHON_EXE" = xyes ; then
		#now check cython
		AC_CHECK_PROGS(CYTHON, [cython-[$PYTHON_VERSION] cython3 cython],["nocython"])
		if test x$CYTHON != "xnocython" ; then
			VALID_PYTHON_NUMPY=yes
			AC_MSG_NOTICE([Building with Python-NumPy bindings])
		elif test "x$enable_python_np" = "xyes" ; then
			AC_MSG_ERROR([Cannot build Python-NumPy bindings])
		else
			AC_MSG_WARN([Cannot build Python-NumPy bindings])
			VALID_PYTHON_NUMPY=no
		fi
		# check for OpenMP
		AC_OPENMP
	fi
fi
AM_CONDITIONAL([ENABLE_PYTHON_NUMPY],[test x$VALID_PYTHON_NUMPY = xyes])

if test "x$VALID_PYTHON" = xyes || test "x$VALID_PYTHON_NUMPY" = xyes ; then
#transform PYTHON_INCDIR and PYTHON_LIBS
PYTHON_INCLUDE_FIXED="'`echo $PYTHON_CPPFLAGS | $SED -e "s/ \{1,\}/','/g" -e  "s/-I//g" `'"
PYTHON_LIBS_FIXED="'`echo $PYTHON_LDFLAGS | $SED -e "s/ \{1,\}/','/g" -e "s/-l//g" `'"
#echo "PYTHON_INCLUDE_FIXED: $PYTHON_INCLUDE_FIXED"
#echo "PYTHON_LIBS_FIXED: $PYTHON_LIBS_FIXED"


AC_SUBST(PYTHON_INCLUDE_FIXED)
AC_SUBST(PYTHON_LIBS_FIXED)
AC_SUBST(PYTHON_XRL_VERSION,$VERSION)
AC_SUBST(PYTHONDIR,$pythondir)
AC_SUBST(PKGPYTHONDIR,$pkgpythondir)
AC_SUBST(PYEXECDIR,$pyexecdir)
AC_SUBST(PKGPYEXECDIR,$pkgpyexecdir)
AC_SUBST(XRLDATADIR,[`echo ${prefix}`/share/xraylib])
# Turn off certain errors for python bindings when -Wall -Werror is in effect -> Cython generates a lot of warnings!
CYTHON_ERROR_CFLAGS=
AX_CHECK_COMPILE_FLAG([-Wno-error=cpp],[CYTHON_ERROR_CFLAGS="${CYTHON_ERROR_CFLAGS} -Wno-error=cpp"],,)
AX_CHECK_COMPILE_FLAG([-Wno-error=attributes],[CYTHON_ERROR_CFLAGS="${CYTHON_ERROR_CFLAGS} -Wno-error=attributes"],,)
AX_CHECK_COMPILE_FLAG([-Wno-error=deprecated-declarations],[CYTHON_ERROR_CFLAGS="${CYTHON_ERROR_CFLAGS} -Wno-error=deprecated-declarations"],,)
AX_CHECK_COMPILE_FLAG([-Wno-error=unreachable-code],[CYTHON_ERROR_CFLAGS="${CYTHON_ERROR_CFLAGS} -Wno-error=unreachable-code"],,)
AX_CHECK_COMPILE_FLAG([-Wno-error=ignored-optimization-argument],[CYTHON_ERROR_CFLAGS="${CYTHON_ERROR_CFLAGS} -Wno-error=ignored-optimization-argument"],,)
AX_CHECK_COMPILE_FLAG([-Wno-error=unused-function],[CYTHON_ERROR_CFLAGS="${CYTHON_ERROR_CFLAGS} -Wno-error=unused-function"],,)
AC_SUBST(CYTHON_ERROR_CFLAGS)
fi
AC_ARG_ENABLE([python-integration],[AS_HELP_STRING([--enable-python-integration],[install the python bindings in the interpreters site-packages folder])],[enable_python_integration=$enableval],[enable_python_integration=check])
if test "x$enable_python_integration" = xyes ; then
	pythondir=$PYTHON_SITE_PKG
	pyexecdir=$PYTHON_SITE_PKG_EXEC
fi

#
#Java bindings
#

AC_ARG_ENABLE([java],[AS_HELP_STRING([--disable-java],[build without the java bindings])],[enable_java=$enableval],[enable_java=check])

VALID_JAVA=

if test "x$enable_java" = xyes && test "x$DISABLE_BINDINGS" = xno  ; then
	AC_CHECK_PROG(JAVA, [java], [java], [none])
	if test "x$JAVA" = "xnone" ; then
		if test "x$enable_java" = xyes ; then
			AC_MSG_ERROR([Cannot build java bindings])
		else
			AC_MSG_WARN([Cannot build java bindings])
			VALID_JAVA=no
		fi
	else
		VALID_JAVA=yes
	fi
fi

if test "x$VALID_JAVA" = xyes ; then
	AC_MSG_NOTICE([Building with java bindings])
fi
AM_CONDITIONAL([ENABLE_JAVA],[test x$VALID_JAVA = xyes])

#C++ example
AC_PROG_CXX
if test `AS_BASENAME([$CXX])` = "$CXX" ; then
  AC_CHECK_PROG(CXX_FULL, [$CXX], $CXX, [none])
fi
if test x$CXX_FULL != "xnone" ; then
	AC_PROG_CXXCPP
	AC_LANG_PUSH([C++])
	AC_CHECK_HEADERS([cstdio cstdlib], ,[CXX=""])
	AC_LANG_POP([C++])
	if test "x$CXX" != x ; then
		AC_MSG_NOTICE([C++ example and tests enabled])
		AC_SUBST(CXX)
	fi
fi
AM_CONDITIONAL([ENABLE_CXX],[test "x$CXX" != x])


#
#Lua bindings
#
AC_ARG_ENABLE([lua],[AS_HELP_STRING([--disable-lua],[build without the lua bindings])],[enable_lua=$enableval],[enable_lua=check])

VALID_LUA=


if test "x$SWIG" = xnoswig && test "x$enable_lua" = xyes && test "x$DISABLE_BINDINGS" = xno  ; then
	#don't even bother when swig is not found
	AC_MSG_ERROR([--enable-lua was given as an option but swig was not found on the system])
elif test "x$SWIG" != xnoswig && test "x$enable_lua" != xno && test "x$DISABLE_BINDINGS" = xno  ; then
	#search for lua interpreter
	AX_WITH_LUA
	if test "x$LUA" != x ; then
		AC_MSG_CHECKING([for lua version])
		LUA_VERSION=`$LUA -e 'print(_VERSION)' 2>&1 | cut -d' ' -f2`
		AC_SUBST(LUA_VERSION)
		AC_MSG_RESULT($LUA_VERSION)
		AX_LUA_HEADERS
		AX_LUA_LIBS
		if test "x$HAVE_LUA_H" != x && test "x$HAVE_LUALIB_H" != x && test "x$LUA_LIB" != x ; then
	  		AC_MSG_CHECKING([consistency of all components of lua development environment])
			ac_save_CFLAGS="$CFLAGS"
			ac_save_LIBS="$LIBS"
			CFLAGS="$LUA_INCLUDE"
			LIBS="$LUA_LIB"
	       		AC_LANG_PUSH([C])
			AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
			#include <lua.h>
			#include "lauxlib.h"
			]], [[
			lua_State *L = luaL_newstate();
			]])],[luaexists=yes],[luaexists=no])
			AC_MSG_RESULT([$luaexists])
			AC_LANG_POP
			CFLAGS="$ac_save_CFLAGS"
			LIBS="$ac_save_LIBS"
		else
			luaexists=no
		fi
	fi
	if test "x$LUA" = x || test "x$HAVE_LUA_H" = x || test "x$HAVE_LUALIB_H" = x || test "x$LUA_LIB" = x || test "x$luaexists" = xno ; then
		if test "x$enable_lua" = xyes ; then
			AC_MSG_ERROR([Cannot build lua bindings])
		else
			AC_MSG_WARN([Cannot build lua bindings])
			VALID_LUA=no
		fi
	else
		VALID_LUA=yes

		AC_MSG_NOTICE([Building with lua bindings])
	fi


fi

AM_CONDITIONAL([ENABLE_LUA],[test x$VALID_LUA = xyes])


#
#
# Ruby bindings
#
#

AC_ARG_ENABLE([ruby],[AS_HELP_STRING([--disable-ruby],[build without the ruby bindings])],[enable_ruby=$enableval],[enable_ruby=check])

AC_ARG_ENABLE([ruby-integration],[AS_HELP_STRING([--enable-ruby-integration],[install the ruby bindings in the interpreters sitearch folder])],[enable_ruby_integration=$enableval],[enable_ruby_integration=check])
VALID_RUBY=

if test "x$SWIG" = xnoswig && test "x$enable_ruby" = xyes && test "x$DISABLE_BINDINGS" = xno  ; then
	#don't even bother when swig is not found
	AC_MSG_ERROR([--enable-ruby was given as an option but swig was not found on the system])
elif test "x$SWIG" != xnoswig && test "x$enable_ruby" != xno && test "x$DISABLE_BINDINGS" = xno  ; then
	AX_RUBY_EXT
	if test "x$RUBY" = x ; then
		if test "x$enable_ruby" = xyes ; then
			AC_MSG_ERROR([Cannot build ruby bindings])
		else
			AC_MSG_WARN([Cannot build ruby bindings])
			VALID_RUBY=no
		fi
	else
		#ruby is present, let's check if it can be used to make bindings
		#borrowed from ax_ruby_devel.m4
		AC_MSG_CHECKING([for the mkmf Ruby package])
		ac_mkmf_result=`$RUBY -rmkmf -e ";" 2>&1`
		if test -z "$ac_mkmf_result"; then
			AC_MSG_RESULT([yes])
			#
    			# final check to see if everything compiles alright
        		#
			AC_CHECK_HEADERS([sys/time.h],[],[])
			AC_CHECK_HEADERS([time.h],[],[])
			AC_CHECK_TYPES([struct timespec],[],[],[
			#ifdef HAVE_SYS_TIME_H
			#include <sys/time.h>
			#endif])
			AC_CHECK_TYPES([struct timeval],[],[],[
			#ifdef HAVE_SYS_TIME_H
			#include <sys/time.h>
			#endif])
			AC_CHECK_DECLS([signbit],[AC_DEFINE(HAVE_SIGNBIT)],AC_CHECK_FUNCS([signbit]),[
			#ifdef HAVE_MATH_H
			#include <math.h>
			#endif])
		    	# save current global flags
		        ac_save_LIBS="$LIBS"
			LIBS="$RUBY_EXT_LDFLAGS"
			ac_save_CPPFLAGS="$CPPFLAGS"
			CPPFLAGS="$RUBY_EXT_CPPFLAGS $RUBY_EXT_INC"
	    		AC_MSG_CHECKING([consistency of all components of ruby development environment])
	        	AC_LANG_PUSH([C])
			AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
			#include <ruby.h>
			]], [[
			ruby_init();
			]])],[rubyexists=yes],[rubyexists=no])
			AC_MSG_RESULT([$rubyexists])
			AC_LANG_POP
			# turn back to default flags
			CPPFLAGS="$ac_save_CPPFLAGS"
			LIBS="$ac_save_LIBS"
			if test "x$rubyexists" = xno ; then
				if test "x$enable_ruby" = xyes ; then
					AC_MSG_ERROR([Cannot build ruby bindings])
				else
					AC_MSG_WARN([Cannot build ruby bindings])
					VALID_RUBY=no
				fi
			else
				VALID_RUBY=yes
				AC_MSG_NOTICE([Building with ruby bindings])
			fi
		else
			AC_MSG_RESULT([no])
			if test "x$enable_ruby" = xyes ; then
				AC_MSG_ERROR([Cannot build ruby bindings])
			else
				AC_MSG_WARN([Cannot build ruby bindings])
				VALID_RUBY=no
			fi
		fi
	fi
fi

AM_CONDITIONAL([ENABLE_RUBY],[test x$VALID_RUBY = xyes])

if test "x$VALID_RUBY" = xyes ; then

#get some more variables
RUBY_EXT_ARCH=`AS_BASENAME([$RUBY_EXT_LIB])`
RUBY_TEMP=`AS_DIRNAME([$RUBY_EXT_LIB])`
RUBY_EXT_VERSION=`AS_BASENAME([$RUBY_TEMP])`

if test "x$enable_ruby_integration" = xyes ; then
	RUBYDIR=$RUBY_EXT_LIB
else
	RUBYDIR=${libdir}/ruby/site_ruby/$RUBY_EXT_VERSION/$RUBY_EXT_ARCH
fi

AC_SUBST(RUBYDIR)

fi

##### PHP -> do not build by default
AC_ARG_ENABLE([php],[AS_HELP_STRING([--disable-php],[build without the php bindings])],[enable_php=$enableval],[enable_php=check])

VALID_PHP=


if test "x$SWIG" = xnoswig && test "x$enable_php" = xyes && test "x$DISABLE_BINDINGS" = xno  ; then
	AC_MSG_ERROR([--enable-php was given as an option but swig was not found on the system])
fi

AC_ARG_VAR(PHP,[the PHP interpreter])
AC_ARG_VAR(PHP_CONFIG,[the config utility of the PHP interpreter])


AC_ARG_ENABLE([php-integration],[AS_HELP_STRING([--enable-php-integration],[install the php bindings in the interpreters sitearch folder])],[enable_php_integration=$enableval],[enable_php_integration=check])

if test "x$enable_php" = xyes && test "x$SWIG" != xnoswig && test "x$DISABLE_BINDINGS" = xno  ;  then
	#search for php executable
	AC_CHECK_PROGS(PHP,[php],["nophp"])

	if test "x$PHP" = xnophp  ; then
		AC_MSG_ERROR([--enable-php was given as an option but php was not found on the system.])
		VALID_PHP=no
	else
		AC_CHECK_PROGS(PHP_CONFIG,[php-config],["nophpconfig"])
		if test "x$PHP_CONFIG" = xnophpconfig  ; then
			AC_MSG_ERROR([--enable-php was given as an option but php-config was not found on the system.])
			VALID_PHP=no
		else
			#for now no test compilation/link
			AC_MSG_CHECKING([for php LDFLAGS])
			PHP_LDFLAGS=`[$PHP_CONFIG --ldflags]`
			AC_MSG_RESULT([$PHP_LDFLAGS])
			#AC_MSG_CHECKING([for php LIBS])
			#PHP_LIBS=`[$PHP_CONFIG --libs]`
			#AC_MSG_RESULT([$PHP_LIBS])
			AC_MSG_CHECKING([for php CFLAGS])
			PHP_CFLAGS=`[$PHP_CONFIG --includes]`
			AC_MSG_RESULT([$PHP_CFLAGS])
			AC_MSG_CHECKING([for php extension directory])
			PHP_DIR=`[$PHP_CONFIG --extension-dir]`
			AC_MSG_RESULT([$PHP_DIR])
			AC_MSG_CHECKING([for php prefix])
			PHP_PREFIX=`[$PHP_CONFIG --prefix]`
			AC_MSG_RESULT([$PHP_PREFIX])
			PHP_PREFIX_DIR=${PHP_DIR#$PHP_PREFIX}
			PHP_PREFIX_DIR=$PHP_PREFIX_DIR
			AC_MSG_CHECKING([for php version])
			PHP_VERNUM=`[$PHP_CONFIG --vernum]`
			AC_MSG_RESULT([$PHP_VERNUM])
			if test "$PHP_VERNUM" -gt "60000" ; then
				PHP_SWIG="-php7"
			else
				PHP_SWIG="-php"
			fi

			if test "x$enable_php_integration" = xyes ; then
				PHP_DIR=$PHP_DIR
				PHP_SCRIPT_DIR="$PHP_PREFIX/share/php"
			else
				if test "x$prefix" = xNONE ; then
					PHP_DIR=["/usr/local/$PHP_PREFIX_DIR"]
					PHP_SCRIPT_DIR=["/usr/local/share/php"]
				else
					PHP_DIR=["${prefix}/$PHP_PREFIX_DIR"]
					PHP_SCRIPT_DIR=["${prefix}/share/php"]
				fi
			fi
			VALID_PHP=yes
			AC_MSG_NOTICE([Building with php bindings])
		fi
	fi
fi

AM_CONDITIONAL([ENABLE_PHP],[test x$VALID_PHP = xyes])

AC_SUBST(PHP_LDFLAGS)
#AC_SUBST(PHP_LIBS)
AC_SUBST(PHP_CFLAGS)
AC_SUBST(PHP_DIR)
AC_SUBST(PHP_SCRIPT_DIR)
AC_SUBST(PHP_SWIG)
AC_SUBST(PHP_PREFIX)

#
# Pascal bindings -> look for fpc
#
AC_ARG_ENABLE([pascal],[AS_HELP_STRING([--disable-pascal],[build without the pascal bindings])],[enable_pascal=$enableval],[enable_pascal=check])

VALID_PASCAL=

AC_ARG_VAR(FPC, [the Free Pascal compiler])
if test "x$enable_pascal" = xyes && test "x$DISABLE_BINDINGS" = xno ; then
	AC_CHECK_PROGS(FPC, [fpc], [nofpc])

	if test "x$FPC" = xnofpc ; then
		AC_MSG_ERROR([--enable-pascal was given as an option but Free Pascal was not found on the system.])
		VALID_PASCAL=no
	else
		#check for version
		AC_MSG_CHECKING([for Free Pascal compiler version])
		FPC_VERSION=`[$FPC -iW]`
		AC_MSG_RESULT([$FPC_VERSION])
		AC_MSG_CHECKING([for Free Pascal compiler host processor])
		FPC_HOST=`[$FPC -iSP]`
		AC_MSG_RESULT([$FPC_HOST])
		AC_MSG_CHECKING([for Free Pascal compiler operating system])
		FPC_OS=`[$FPC -iSO]`
		AC_MSG_RESULT([$FPC_OS])
		# some version checking may become necessary here... some old fpc compilers are really buggy like the one that ships with centos6
		FPC_DIR=${libdir}/fpc/$FPC_VERSION/units/$FPC_HOST-$FPC_OS/
		AC_SUBST(FPC_DIR)
		VALID_PASCAL=yes
	fi


fi

AM_CONDITIONAL([ENABLE_PASCAL],[test x$VALID_PASCAL = xyes])

AC_CONFIG_FILES([Makefile
				 src/Makefile
				 include/Makefile
				 example/Makefile
				 cplusplus/Makefile
				 cplusplus/tests/Makefile
				 idl/Makefile
				 python/Makefile
				 doc/Makefile
				 libxrl.pc
				 libxrlf03.pc
				 data/Makefile
				 java/Makefile
				 java/tests/Makefile
				 java/build.gradle
				 java/settings.gradle
				 perl/Makefile
				 lua/Makefile
				 fortran/Makefile
				 ruby/Makefile
				 php/Makefile
				 pascal/Makefile
				 tests/Makefile
				 fortran/tests/Makefile
				 python/tests/Makefile
				 lua/tests/Makefile
				 pascal/tests/Makefile
				 perl/tests/Makefile
				 php/tests/Makefile
				 ruby/tests/Makefile
])


AC_CONFIG_HEADERS([config.h])

abs_top_builddir=`pwd -P`
AC_SUBST(abs_top_builddir)
abs_top_srcdir=`AS_DIRNAME([$0])`
cd $abs_top_srcdir
abs_top_srcdir=`pwd -P`
cd $abs_top_builddir
AC_SUBST(abs_top_srcdir)

AM_CONDITIONAL([ABS_SRC_BUILD_EQUAL],[test x$abs_top_builddir = x$abs_top_srcdir])

AC_OUTPUT

AC_MSG_NOTICE([])
AC_MSG_NOTICE([])
if test "$VALID_FORTRAN" = "yes" || test "$VALID_IDL" = "yes" || test "$VALID_PERL" = "yes" || test "$VALID_PYTHON" = "yes" || test "$VALID_JAVA" = "yes" || test "$VALID_LUA" = "yes" || test "$VALID_RUBY" = yes || test "$VALID_PYTHON_NUMPY" = yes || test "$VALID_PHP" = "yes" || test "VALID_PASCAL" = "yes" ; then
	AC_MSG_NOTICE([Building xraylib with bindings for:])
	if test "$VALID_FORTRAN" = "yes" ; then
		AC_MSG_NOTICE([Fortran 2003])
	fi
	if test "$VALID_IDL" = yes ; then
		AC_MSG_NOTICE([IDL])
	fi
	if test "$VALID_PERL" = yes ; then
		AC_MSG_NOTICE([Perl])
	fi
	if test "$VALID_PYTHON" = yes ; then
		AC_MSG_NOTICE([Python])
	fi
	if test "$VALID_PYTHON_NUMPY" = yes ; then
		AC_MSG_NOTICE([Python-NumPy])
	fi
	if test "$VALID_JAVA" = yes ; then
		AC_MSG_NOTICE([Java])
	fi
	if test "$VALID_LUA" = yes ; then
		AC_MSG_NOTICE([Lua])
	fi
	if test "$VALID_RUBY" = yes ; then
		AC_MSG_NOTICE([Ruby])
	fi
	if test "$VALID_PHP" = yes ; then
		AC_MSG_NOTICE([PHP])
	fi
	if test "$VALID_PASCAL" = yes ; then
		AC_MSG_NOTICE([Pascal])
	fi
else
	AC_MSG_NOTICE([Building without bindings])
fi
