#Copyright (c) 2009, 2010, 2011, 2012, 2013 Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.




AC_INIT([xraylib],[2.16.1],[Tom.Schoonjans@me.com])
AC_PREREQ([2.60])
AC_CONFIG_SRCDIR([include/xraylib.h])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_CANONICAL_HOST


LIB_CURRENT=6
LIB_REVISION=1
LIB_AGE=3
AC_SUBST(LIB_CURRENT)
AC_SUBST(LIB_REVISION)
AC_SUBST(LIB_AGE)
LIB_CURRENT_MINUS_AGE=`expr $LIB_CURRENT - $LIB_AGE`
AC_SUBST(LIB_CURRENT_MINUS_AGE)
AC_CONFIG_MACRO_DIR([m4])


AC_USE_SYSTEM_EXTENSIONS
AC_CHECK_FUNCS([strndup strdup])




AM_PROG_AR
#at least version 2.0 of libtool is required for creating the fortran bindings
LT_PREREQ([2.0.0])
#dlopen is necessary for IDL bindings
LT_INIT([dlopen disable-fast-install win32-dll])
#AC_LIBTOOL_DLOPEN
AC_PROG_LIBTOOL

AC_PROG_CC
AM_PROG_CC_C_O
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
#Some code is necessary in case we are cross compiling with mingw on a platform different from cygwin -> Wine is necessary!
#
WINE=
DISABLE_BINDINGS=no
CROSS_COMPILING=
LDFLAGS_LIBXRL=

#cross_compiling variable is not reliable on modern bash shells due to wine integration... look at the compiler instead
#check instead of $build and $host differ


if test "x$host" != "x$build" ; then
	#cross compilation detected
	#if $host is mingw, look for wine
	case "$host" in 
		*mingw*)
			#disable bindings
			DISABLE_BINDINGS=yes
			#look for wine
			AC_CHECK_PROGS([WINE],[wine],["nowine"])
			;;
		*)
			AC_MSG_ERROR([A platform was detected that is not supported for cross-compilation])
			;;
	esac

	if test "x$WINE" = "xnowine" ; then
		AC_MSG_ERROR([wine is necessary when cross-compiling for windows.])
	fi
	CROSS_COMPILING=yes
	LDFLAGS+="-no-undefined"
	LDFLAGS_LIBXRL=-Wl,--output-def,libxrl-$LIB_CURRENT_MINUS_AGE.def
else
	case "$host" in
		*mingw*)
			#build dll
			LDFLAGS+="-no-undefined"
			LDFLAGS_LIBXRL=-Wl,--output-def,libxrl-$LIB_CURRENT_MINUS_AGE.def
			;;
	esac
fi


AC_SUBST(LDFLAGS_LIBXRL)



AM_CONDITIONAL([ENABLE_CROSS],[test x$CROSS_COMPILING = xyes])

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

	AC_ARG_WITH([rsiidl-bindir],[AS_HELP_STRING([--with-rsiidl-bindir],[set location of the rsi-idl binary])],[RSIIDL_BINDIR=$withval])

	if test -z "$RSIIDL_BINDIR" ; then
		#no location was presented as an option -> search the usual suspects...
		IDL_PATH=/usr/local/itt/idl/bin:/usr/local/rsi/idl/bin:/usr/local/pkg/graphics/rsi/idl/bin:/Applications/exelis/idl/bin:/usr/local/exelis/idl/bin:/Applications/itt/idl/bin:/usr/local/idl/bin:/Applications/itt/idl/idl/bin:/usr/local/itt/idl/idl/bin:/Applications/rsi/idl/bin
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
			VALID_IDL=no
		fi	
	fi
	#moving on...
	if test -n "$RSIIDL_BINDIR" && test "x$VALID_IDL" != xno ; then
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
			AC_MSG_WARN([RSI-IDL header file was not found in $RSIIDL_INCDIR. This most likely indicates a problem with the IDL installation.])
			VALID_IDL=no
		fi
		#check for path to libidl.so
		if test "x$VALID_IDL" != xno ; then
  		case "$host" in

		    x86_64-*-linux*)
		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.linux.x86_64
		      IDL_CFLAGS="-fPIC"
		      IDL_LD_FLAGS="-shared -Xlinker -Bsymbolic -Xlinker --warn-once -Xlinker -noinhibit-exec"
		      ;;

		    *-linux*)
		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.linux.x86
	              IDL_CFLAGS="-fPIC"
                      IDL_LD_FLAGS="-shared -Xlinker -Bsymbolic -Xlinker --warn-once -Xlinker -noinhibit-exec"
		      ;;

		    *-sgi*)
#		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.sgi
		      AC_MSG_WARN([IRIX is currently not supported. Disable the bindings with --disable-idl])
			VALID_IDL=no
		      ;;
		    x86_64-*-darwin*)
			if test -d $RSIIDL_BINDIR/bin.darwin.x86_64 ; then
		      		IDL_LIBBIN=$RSIIDL_BINDIR/bin.darwin.x86_64
		      		IDL_CFLAGS="-fPIC"
		      		IDL_LD_FLAGS="-Xlinker -flat_namespace -Xlinker -undefined -Xlinker suppress -Xlinker -bundle"
			else
				AC_MSG_WARN([You appear to be running an IDL version on Mac OS X Snow Leopard that does not have the required 64-bit IDL libraries. Please upgrade your IDL installation to at least version 7.1])
			VALID_IDL=no
		        fi
		      ;;



		    i386-*-darwin*)
	    		IDL_LIBBIN=$RSIIDL_BINDIR/bin.darwin.i386
			IDL_CFLAGS="-fPIC"
		      	IDL_LD_FLAGS="-Xlinker -flat_namespace -Xlinker -undefined -Xlinker suppress -Xlinker -bundle"
		      ;;

		    ppc*-darwin*)
		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.darwin.ppc
		      IDL_CFLAGS="-fPIC"
		      IDL_LD_FLAGS="-Xlinker -flat_namespace -Xlinker -undefined -Xlinker suppress -Xlinker -bundle"
		      ;;
		   *-solaris-*)
		      AC_MSG_WARN([Solaris is currently not supported. Disable the bindings with --disable-idl])
			VALID_IDL=no
		      ;;

		   *)
		      AC_MSG_WARN([Could not detect platform for IDL.])
			VALID_IDL=no
		      ;;

		  esac

		if test "x$VALID_IDL" != xno ; then
		AC_ARG_WITH([rsiidl-libdir],[AS_HELP_STRING([--with-rsiidl-bindir],[set location of the rsi-idl libraries])],[RSIIDL_LIBBIN=$withval],[RSIIDL_LIBBIN="$IDL_LIBBIN"])
		AC_CHECK_FILE([$IDL_LIBBIN/libidl.so],[RSIIDL_HASBIN_SO=yes],[RSIIDL_HASBIN_SO=no])	
		AC_CHECK_FILE([$IDL_LIBBIN/libidl.dylib],[RSIIDL_HASBIN_DY=yes],[RSIIDL_HASBIN_DY=no])	
		if test "x$RSIIDL_HASBIN_SO" = xno && test "x$RSIIDL_HASBIN_DY" = xno  ; then
			AC_MSG_WARN([libidl.so or libidl.dylib could not be found. This means that you have either a corrupt or custom installation of idl. In the first case, set --with-rsiidl-libdir to match the location of the IDL libraries])
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
#find and xargs are necessary for the uninstallation of the perl bindings
AC_CHECK_PROGS([FIND],[find],["nofind"])
AC_CHECK_PROGS([XARGS],[xargs],["noxargs"])


#
#perl bindings
#

AC_ARG_ENABLE([perl],[AS_HELP_STRING([--disable-perl],[build without the perl bindings])],[enable_perl=$enableval],[enable_perl=check])

VALID_PERL=

if test "x$SWIG" = xnoswig && test "x$enable_perl" = xyes && test "x$DISABLE_BINDINGS" = xno  ; then
	AC_MSG_ERROR([--enable-perl was given as an option but swig was not found on the system])
fi
if test "x$FIND" = xnofind && test "x$enable_perl" = xyes && test "x$DISABLE_BINDINGS" = xno  ; then
	AC_MSG_ERROR([--enable-perl was given as an option but find was not found on the system])
fi
if test "x$XARGS" = xnoxargs && test "x$enable_perl" = xyes && test "x$DISABLE_BINDINGS" = xno  ; then
	AC_MSG_ERROR([--enable-perl was given as an option but xargs was not found on the system])
fi

MMFOUND=
if test "x$enable_perl" != xno && test "x$SWIG" != xnoswig && test "x$FIND" != xnofind && test "x$XARGS" != xnoargs && test "x$DISABLE_BINDINGS" = xno  ;  then
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
AC_ARG_ENABLE([python-integration],[AS_HELP_STRING([--enable-python-integration],[install the python bindings in the interpreters site-packages folder])],[enable_python_integration=$enableval],[enable_python_integration=check])

VALID_PYTHON=

if test "x$SWIG" = xnoswig && test "x$enable_python" = xyes && test "x$DISABLE_BINDINGS" = xno  ; then
	#don't even bother when swig is not found
	AC_MSG_ERROR([--enable-python was given as an option but swig was not found on the system])
elif test "x$SWIG" = xswig && test "x$enable_python" != xno && test "x$DISABLE_BINDINGS" = xno  ; then
	#verify the python installation
	AM_PATH_PYTHON(,[PYTHON_FOUND=true],[PYTHON_FOUND=false])
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
			VALID_PYTHON=yes
		fi

	fi
fi




if test "x$VALID_PYTHON" = xyes ; then
AC_MSG_NOTICE([Building with Python bindings])
#transform PYTHON_INCDIR and PYTHON_LIBS
PYTHON_INCLUDE_FIXED="'`echo $PYTHON_CPPFLAGS | sed -e "s/ \{1,\}/','/g" -e  "s/-I//g" `'"
PYTHON_LIBS_FIXED="'`echo $PYTHON_LDFLAGS | sed -e "s/ \{1,\}/','/g" -e "s/-l//g" `'"
#echo "PYTHON_INCLUDE_FIXED: $PYTHON_INCLUDE_FIXED"
#echo "PYTHON_LIBS_FIXED: $PYTHON_LIBS_FIXED"

if test "x$enable_python_integration" = xyes ; then
	pythondir=$PYTHON_SITE_PKG
	pyexecdir=$PYTHON_SITE_PKG_EXEC
fi

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


#
#Java bindings
#

AC_ARG_ENABLE([java],[AS_HELP_STRING([--disable-java],[build without the java bindings])],[enable_java=$enableval],[enable_java=check])

VALID_JAVA=

if test "x$SWIG" = xnoswig && test "x$enable_java" = xyes && test "x$DISABLE_BINDINGS" = xno  ; then
	#don't even bother when swig is not found
	AC_MSG_ERROR([--enable-java was given as an option but swig was not found on the system])
elif test "x$SWIG" = xswig && test "x$enable_java" != xno && test "x$DISABLE_BINDINGS" = xno  ; then
	#search for java compiler
	AX_PROG_JAVAC
	#search for java virtual machine
	if test "x$JAVAC" != x ; then
		AX_PROG_JAVA
	fi
	#search for location of jni.h, using javah
	if test "x$JAVAC" != x && test "x$JAVA" != x ; then
		AX_PROG_JAVAH
	fi
	if test "x$JAVAC" = x || test "x$JAVA" = x || test "x$JAVACPPFLAGS" = x ; then
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
	case $host in
		*-*-darwin*)
			JAVADYNAMICLINKING="-dynamiclib -framework JavaVM"
			JAVACFLAGS=""
			JAVALIBRARYPREFIX="lib"
			JAVASO=".jnilib"
			LDSHARED="cc -bundle -undefined suppress -flat_namespace"
			CCSHARED=""
			JAVALDSHARED='$(CC)'
			JAVACXXSHARED='$(CXX)'

			;;
		*-*-solaris*)
			SO=".so"
			JAVADYNAMICLINKING=""
			JAVACFLAGS=""
			JAVALIBRARYPREFIX="lib"
			JAVASO=$SO
			LDSHARED="ld -G"
			CCSHARED=""
			JAVALDSHARED='$(LDSHARED)'
			JAVACXXSHARED='$(CXXSHARED)'
			;;
		*)
			SO=".so"
			JAVADYNAMICLINKING=""
			JAVACFLAGS=""
			JAVALIBRARYPREFIX="lib"
			JAVASO=$SO
			LDSHARED="gcc -shared"
			CCSHARED="-fpic"
			JAVALDSHARED='$(LDSHARED)'
			JAVACXXSHARED='$(CXXSHARED)'
			;;
	esac

	JAVAINC=$JAVACPPFLAGS
	JAVAROOT='$(builddir)'
	AC_SUBST(JAVAROOT)
	AC_SUBST(JAVA)
	AC_SUBST(JAVAC)
	AC_SUBST(JAVAINC)
	AC_SUBST(JAVADYNAMICLINKING)
	AC_SUBST(JAVALIBRARYPREFIX)
	AC_SUBST(JAVASO)
	AC_SUBST(LDSHARED)
	AC_SUBST(JAVALDSHARED)
	AC_SUBST(JAVACXXSHARED)
	AC_SUBST(JAVACFLAGS)
	AC_SUBST(CCSHARED)

fi
AM_CONDITIONAL([ENABLE_JAVA],[test x$VALID_JAVA = xyes])


#C++ example
AC_PROG_CXX
AC_PROG_CXXCPP
AC_LANG_PUSH([C++])
AC_CHECK_HEADERS([cstdio cstdlib], ,[CXX=""])
AC_LANG_POP([C++])
if test x$CXX != x ; then
	AC_MSG_NOTICE([C++ example enabled])
	AC_SUBST(CXX)
fi
AM_CONDITIONAL([ENABLE_CXX],[test $CXX])


#
#Lua bindings
#
AC_ARG_ENABLE([lua],[AS_HELP_STRING([--disable-lua],[build without the lua bindings])],[enable_lua=$enableval],[enable_lua=check])

VALID_LUA=


if test "x$SWIG" = xnoswig && test "x$enable_lua" = xyes && test "x$DISABLE_BINDINGS" = xno  ; then
	#don't even bother when swig is not found
	AC_MSG_ERROR([--enable-lua was given as an option but swig was not found on the system])
elif test "x$SWIG" = xswig && test "x$enable_lua" != xno && test "x$DISABLE_BINDINGS" = xno  ; then
	#search for lua interpreter
	AX_WITH_LUA
	if test "x$LUA" != x ; then
		AX_LUA_HEADERS
		AX_LUA_LIBS
	fi
	if test "x$LUA" = x || test "x$HAVE_LUA_H" = x || test "x$HAVE_LUALIB_H" = x || test "x$LUA_LIB" = x ; then
		if test "x$enable_lua" = xyes ; then
			AC_MSG_ERROR([Cannot build lua bindings])
		else
			AC_MSG_WARN([Cannot build lua bindings])
			VALID_LUA=no
		fi
	else
		VALID_LUA=yes
		AC_MSG_CHECKING([for lua version])
		LUA_VERSION=`$LUA -e 'print(_VERSION)' 2>&1 | cut -d' ' -f2`
		AC_SUBST(LUA_VERSION)
		AC_MSG_RESULT($LUA_VERSION)

		AC_MSG_NOTICE([Building with lua bindings])
		case "$host" in
			*-linux*)
				LUA_LD_FLAGS="-shared"
				LUA_CFLAGS="-fPIC"
				;;
			*-darwin*)
				LUA_LD_FLAGS="-bundle -undefined dynamic_lookup -all_load"
				LUA_CFLAGS=""
				;;
			*)
				LUA_LD_FLAGS="-shared"
				LUA_CFLAGS="-fPIC"
				;;
		esac
		AC_SUBST(LUA_LD_FLAGS)
		AC_SUBST(LUA_CFLAGS)
	fi
	

fi

AM_CONDITIONAL([ENABLE_LUA],[test x$VALID_LUA = xyes])

AC_CONFIG_FILES([Makefile src/Makefile include/Makefile example/Makefile idl/Makefile python/xrlsetup.py python/Makefile python/xrayhelp.py python/xraymessages.py doc/Makefile libxrl.pc libxrlf03.pc xraylib.spec data/Makefile java/Makefile nsis/xraylib.nsi nsis/python/Makefile nsis/Makefile nsis/python/python2.6/Makefile nsis/python/python2.7/Makefile nsis/python/python3.1/Makefile nsis/python/cli/Makefile nsis/python/cli_py2exe/Makefile nsis/python/cli_py2exe/xraylib-cli.nsi nsis/python/cli_py2exe/setup.py perl/Makefile lua/Makefile fortran/Makefile] nsis/python/python2.6/xrlsetup.py nsis/python/python2.7/xrlsetup.py nsis/python/python3.1/xrlsetup.py nsis/dotNet/Makefile nsis/python/python3.2/Makefile nsis/python/python3.2/xrlsetup.py nsis/python/python3.3/Makefile nsis/python/python3.3/xrlsetup.py)
AC_CONFIG_HEADERS([config.h])

abs_top_builddir=`pwd -P`
AC_SUBST(abs_top_builddir)
abs_top_srcdir=`AS_DIRNAME([$0])`
cd $abs_top_srcdir
abs_top_srcdir=`pwd -P`
cd $abs_top_builddir
AC_SUBST(abs_top_srcdir)

#windows slashes versions
abs_top_builddir_win=`echo $abs_top_builddir | sed -e 's/\//\\\/g'`
abs_top_srcdir_win=`echo $abs_top_srcdir | sed -e 's/\//\\\/g'`
AC_SUBST(abs_top_builddir_win)
AC_SUBST(abs_top_srcdir_win)

AM_CONDITIONAL([ABS_SRC_BUILD_EQUAL],[test x$abs_top_builddir = x$abs_top_srcdir])

AC_OUTPUT
