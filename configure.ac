#Copyright (c) 2009, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.




AC_PREREQ([2.60])
AC_INIT([xraylib],[2.13],[Tom.Schoonjans@UGent.be])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])

LIB_CURRENT=2
LIB_REVISION=0
LIB_AGE=0
AC_SUBST(LIB_CURRENT)
AC_SUBST(LIB_REVISION)
AC_SUBST(LIB_AGE)

AC_USE_SYSTEM_EXTENSIONS
AC_CHECK_FUNCS([strndup strdup])




#at least version 2.0 of libtool is required for creating the fortran bindings
LT_PREREQ([2.0.0])
#dlopen is necessary for IDL bindings
LT_INIT([dlopen disable-fast-install])
#AC_LIBTOOL_DLOPEN
AC_PROG_LIBTOOL

AC_PROG_CC
#this next line may never be reached...
if test -z $CC ; then
	AC_MSG_ERROR([no C compiler was found on the system.])
fi

AC_CANONICAL_HOST


#headers check
AC_CHECK_HEADERS([math.h stdio.h stdlib.h string.h ctype.h stddef.h])



#libraries check
AC_CHECK_LIB([m],[log],[],[AC_MSG_ERROR([the log function must be present in libm])])
AC_CHECK_LIB([m],[exp],[],[AC_MSG_ERROR([the exp function must be present in libm])])
AC_CHECK_LIB([m],[cos],[],[AC_MSG_ERROR([the cos function must be present in libm])])
AC_CHECK_LIB([m],[sin],[],[AC_MSG_ERROR([the sin function must be present in libm])])




#
#fortran 2003 bindings
#

AC_ARG_ENABLE([fortran2003], [AS_HELP_STRING([--disable-fortran2003],[build without the Fortran 2003 bindings])],[enable_fortran2003=$enableval],[enable_fortran2003=check] )



VALID_FORTRAN=
if test "x$enable_fortran2003" != xno ; then  
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

if test "x$enable_idl" != xno ; then  
#search for header and libraries
#inspired by the gpulib autoconf macros...

	AC_ARG_WITH([rsiidl-bindir],[AS_HELP_STRING([--with-rsiidl-bindir],[set location of the rsi-idl binary])],[RSIIDL_BINDIR=$withval])

	if test -z "$RSIIDL_BINDIR" ; then
		#no location was presented as an option -> search the usual suspects...
		IDL_PATH=/usr/local/itt/idl/bin:/usr/local/rsi/idl/bin:/usr/local/pkg/graphics/rsi/idl/bin:/Applications/rsi/idl/bin:/Applications/itt/idl/bin:/usr/local/idl/bin
		AC_PATH_PROGS([RSIIDL_BIN], [idl],[],[$IDL_PATH])
		if test -z "$RSIIDL_BIN" && test "x$enable_idl" = xyes ; then
			#binaries not found while they were required through the enable-idl option
			AC_MSG_ERROR([RSI-IDL binaries not found in $IDL_PATH. Use --with-rsiidl-bindir to set the location of the rsi-idl binary])
		elif test -z "$RSIIDL_BIN" && test "x$enable_idl" = xcheck ; then
			#binaries were not found but they were not requested explicitally through an option -> 
			AC_MSG_WARN([RSI-IDL binaries not found in $IDL_PATH. Use --with-rsiidl-bindir to set the location of the rsi-idl binary])
			VALID_IDL=no
		else
			#binaries were found...
		#	AC_MSG_RESULT([yes, in $IDL_PATH])
			RSIIDL_BINDIR=`AS_DIRNAME([$RSIIDL_BIN])`
		fi
	else
		#test if the user-supplied value contains the idl binary
		AC_PATH_PROGS([RSIIDL_BIN], [idl],[],[$RSIIDL_BINDIR])
		if test -z "$RSIIDL_BIN" ; then
			AC_MSG_ERROR([RSI-IDL binaries not found user-supplied $RSIIDL_BINDIR. Wrong value for --with-rsiidl-bindir])
		fi	
	fi
	#moving on...
	if test -n "$RSIIDL_BINDIR" ; then
		RSIIDL_DIR=`AS_DIRNAME([$RSIIDL_BINDIR])`
		#if this test succeeds then using version 5.6 or greater
		RSIIDL_INCDIR="$RSIIDL_DIR/external/include"
		AC_CHECK_FILE(["$RSIIDL_INCDIR/idl_export.h"],[RSIIDL_HASINC=yes],[RSIIDL_HASINC=no])

		#for older versions this would yield the header file
		if test "x$RSIIDL_HASINC" = xno ; then
			RSIIDL_INCDIR="$RSIIDL_DIR/external"
			AC_CHECK_FILE(["$RSIIDL_INCDIR/export.h"],[RSIIDL_HASINC=yes],[RSIIDL_HASINC=no])
		fi
		if test "x$RSIIDL_HASINC" = xno ; then
			AC_MSG_ERROR([RSI-IDL header file was not found in $RSIIDL_INCDIR. This most likely indicates a problem with the IDL installation.])
		fi
		#check for path to libidl.so
  		case "$host" in

		    x86_64-*-linux*)
		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.linux.x86_64
		      IDL_CFLAGS="-fPIC"
		      IDL_LD_FLAGS="-shared -Bsymbolic"
		      ;;

		    *-linux*)
		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.linux.x86
	              IDL_CFLAGS="-fPIC"
                      IDL_LD_FLAGS="-shared -Bsymbolic"
		      ;;

		    *-sgi*)
#		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.sgi
		      AC_MSG_ERROR([IRIX is currently not supported. Disable the bindings with --disable-idl])
		      ;;
		    x86_64-*-darwin*)
			if test -d $RSIIDL_BINDIR/bin.darwin.x86_64 ; then
		      		IDL_LIBBIN=$RSIIDL_BINDIR/bin.darwin.x86_64
		      		IDL_CFLAGS="-fPIC"
		      		IDL_LD_FLAGS="-flat_namespace -undefined suppress -bundle"
			else
				AC_MSG_ERROR([You appear to be running an IDL version on Mac OS X Snow Leopard that does not have the required 64-bit IDL libraries. Please upgrade your IDL installation to at least version 7.1])
		        fi
		      ;;



		    i386-*-darwin*)
	    		IDL_LIBBIN=$RSIIDL_BINDIR/bin.darwin.i386
			IDL_CFLAGS="-fPIC"
			IDL_LD_FLAGS="-flat_namespace -undefined suppress -bundle"
		      ;;

		    ppc*-darwin*)
		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.darwin.ppc
		      IDL_CFLAGS="-fPIC"
		      IDL_LD_FLAGS="-flat_namespace -undefined suppress -bundle"
		      ;;

		   *)
		      AC_MSG_ERROR([Could not detect platform for IDL.])
		      ;;

		  esac

		AC_ARG_WITH([rsiidl-libdir],[AS_HELP_STRING([--with-rsiidl-bindir],[set location of the rsi-idl libraries])],[RSIIDL_LIBBIN=$withval],[RSIIDL_LIBBIN="$IDL_LIBBIN"])
		AC_CHECK_FILE([$IDL_LIBBIN/libidl.so],[RSIIDL_HASBIN_SO=yes],[RSIIDL_HASBIN_SO=no])	
		AC_CHECK_FILE([$IDL_LIBBIN/libidl.dylib],[RSIIDL_HASBIN_DY=yes],[RSIIDL_HASBIN_DY=no])	
		if test "x$RSIIDL_HASBIN_SO" = xno && test "x$RSIIDL_HASBIN_DY" = xno  ; then
			AC_MSG_ERROR([libidl.so or libidl.dylib could not be found. This means that you have either a corrupt or custom installation of idl. In the first case, set --with-rsiidl-libdir to match the location of the IDL libraries])
		fi
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
fi #test "x$enable_idl" != xno

AC_SUBST(ARCHFLAGS)

AM_CONDITIONAL([ENABLE_IDL],[test x$VALID_IDL = xyes])

#search for swig which is necessary for both the perl and python bindings
AC_CHECK_PROGS([SWIG],[swig],["noswig"])
#find and xargs are necessary for the uninstallation of the perl bindings
AC_CHECK_PROGS([FIND],[find],["nofind"])
AC_CHECK_PROGS([XARGS],[xargs],["noxargs"])


#
#perl bindings
#

AC_ARG_ENABLE([perl],[AS_HELP_STRING([--disable-perl],[build without the perl bindings])],[enable_perl=$enableval],[enable_perl=check])

VALID_PERL=

if test "x$SWIG" = xnoswig && test "x$enable_perl" = xyes ; then
	AC_MSG_ERROR([--enable-perl was given as an option but swig was not found on the system])
fi
if test "x$FIND" = xnofind && test "x$enable_perl" = xyes ; then
	AC_MSG_ERROR([--enable-perl was given as an option but find was not found on the system])
fi
if test "x$SWIG" = xnoxargs && test "x$enable_perl" = xyes ; then
	AC_MSG_ERROR([--enable-perl was given as an option but xargs was not found on the system])
fi

MMFOUND=
if test "x$enable_perl" != xno && test "x$SWIG" != xnoswig && test "x$FIND" != xnofind && test "x$XARGS" != xnoargs ;  then
	#search for perl executable (although I can't imagine a modern system without perl...)
	if test -z $PERL ; then
		AC_CHECK_PROGS(PERL,[perl perl5],["noperl"])
	fi
	if test "x$PERL" != xnoperl ; then
		#see if the Perl distribution has the MakeMaker module (at least version 6.31)
		AS_IF([ $PERL -M'ExtUtils::MakeMaker 6.31' -e "exit 0;"],[MMFOUND=true],[MMFOUND=false])
	fi
	

	if test "x$PERL" = xnoperl || test "x$MMFOUND" = xfalse ; then
		if test "x$enable_perl" = xyes ; then
			if test "x$PERL" = xnoperl ; then
				AC_MSG_ERROR([--enable-perl was given as an option but Perl was not found on the system.])
			elif test "x$MMFOUND" = x$false ; then
				AC_MSG_ERROR([--enable-perl was given as an option but the ExtUtils::MakeMaker module was not found on the system or did not meet the minimum version requirement.])
			fi
		else
			AC_MSG_WARN([perl was not found on the system or the installation was incomplete (ExtUtils::MakeMaker requirement). The perl bindings will not be built.])
			VALID_PERL=no
		fi
	else
			#everything ok -> let's build those bindings
			VALID_PERL=yes
			AC_MSG_NOTICE([Building with Perl bindings])
	fi
else
VALID_PERL=no
fi #test enable perl

AM_CONDITIONAL([ENABLE_PERL],[test x$VALID_PERL = xyes])

#
#Python bindings
#


AC_ARG_ENABLE([python],[AS_HELP_STRING([--disable-python],[build without the python bindings])],[enable_python=$enableval],[enable_python=check])

VALID_PYTHON=

if test "x$SWIG" = xnoswig && test "x$enable_python" = xyes ; then
	#don't even bother when swig is not found
	AC_MSG_ERROR([--enable-python was given as an option but swig was not found on the system])
elif test "x$SWIG" = xswig && test "x$enable_python" != xno ; then
	#verify the python installation
	AM_PATH_PYTHON(,[PYTHON_FOUND=true],[PYTHON_FOUND=false])
	if test "x$PYTHON_FOUND" = xtrue ; then
		#let's try to use python-config
		AC_CHECK_PROGS([PYTHON_CONFIG],[python-config $PYTHON-config],"nopython-config")
		PYTHON_LIBS=
		PYTHON_INCDIR=
		if test "x$PYTHON_CONFIG" != xnopython-config ; then
			#we found it!
			#try it first though -> if the python installation is incomplete, python-config will exit with a bunch of errors
			AC_MSG_CHECKING([whether $PYTHON_CONFIG works properly])
			AS_IF([$PYTHON_CONFIG --includes],[
				PYTHON_INCDIR=`$PYTHON_CONFIG --includes`
				PYTHON_LIBS=`$PYTHON_CONFIG --libs`
				VALID_PYTHON=yes
			],[
			if test "x$enable_python" = xyes ; then
				AC_MSG_ERROR([python-config is present but exits with an error. This probably indicates a partial python installation. You may resolve this by installing the development packages.])
			else
				AC_MSG_WARN([python-config is present but exits with an error. This probably indicates a partial python installation. You may resolve this by installing the development packages.])
			fi
			VALID_PYTHON=no])
		else
			if test "x$enable_python" = xyes ; then
				AC_MSG_ERROR([python-config is not present. This probably indicates a partial python installation. You may resolve this by installing the development packages.])
			else
				AC_MSG_WARN([python-config is not present. This probably indicates a partial python installation. You may resolve this by installing the development packages.])
			fi
			VALID_PYTHON=no
		fi
	fi
fi




if test "x$VALID_PYTHON" = xyes ; then
AC_MSG_NOTICE([Building with Python bindings])
#transform PYTHON_INCDIR and PYTHON_LIBS
PYTHON_INCLUDE_FIXED="'`echo $PYTHON_INCDIR | sed -e "s/ \{1,\}/','/g" -e  "s/-I//g" `'"
PYTHON_LIBS_FIXED="'`echo $PYTHON_LIBS | sed -e "s/ \{1,\}/','/g" -e "s/-l//g" `'"
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
fi
AM_CONDITIONAL([ENABLE_PYTHON],[test x$VALID_PYTHON = xyes])

#C++ example
AC_PROG_CXX
if test $CXX ; then
	AC_MSG_NOTICE([C++ example enabled])
	AC_SUBST(CXX)
	AM_CONDITIONAL([ENABLE_CXX],[test $CXX])
fi








AC_CONFIG_FILES([Makefile src/Makefile include/Makefile example/Makefile idl/Makefile src/xraylib.i python/xrlsetup.py python/Makefile python/xrayhelp.py python/xraymessages.py doc/Makefile libxrl.pc libxrlf03.pc xraylib.spec data/Makefile])
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
