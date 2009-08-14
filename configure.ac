AC_INIT([xraylib],[2.12],[Tom.Schoonjans@UGent.be])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])

LIB_CURRENT=1
LIB_REVISION=0
LIB_AGE=0
AC_SUBST(LIB_CURRENT)
AC_SUBST(LIB_REVISION)
AC_SUBST(LIB_AGE)

#at least version 2.0 of libtool is necessary for creating the fortran bindings
#LT_PREREQ([2.0.0])
#dlopen is necessary for IDL bindings
#LT_INIT([dlopen])
AC_LIBTOOL_DLOPEN
AC_PROG_LIBTOOL

AC_PROG_CC


AC_CANONICAL_HOST


#headers check
AC_CHECK_HEADERS([math.h stdio.h stdlib.h string.h])



#libraries check
AC_CHECK_LIB([m],[log],[],[AC_MSG_ERROR([the log function must be present in libm])])
AC_CHECK_LIB([m],[exp],[],[AC_MSG_ERROR([the exp function must be present in libm])])
AC_CHECK_LIB([m],[cos],[],[AC_MSG_ERROR([the cos function must be present in libm])])
AC_CHECK_LIB([m],[sin],[],[AC_MSG_ERROR([the sin function must be present in libm])])





#fortran 2003 bindings

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
INTERFACE
	FUNCTION foo (bar,morebar) BIND(C,NAME='foo')
		USE, INTRINSIC ::ISO_C_BINDING 
		IMPLICIT NONE
		REAL (KIND=C_DOUBLE) :: foo
		INTEGER (KIND=C_INT), INTENT(IN) :: bar
		REAL (KIND=C_DOUBLE), INTENT(IN) :: morebar
	ENDFUNCTION foo
ENDINTERFACE
ENDMODULE f2003_test
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
	elif test "x$enable_fortran2003" = yes ; then
		AC_MSG_ERROR([--enable-fortran2003 was given but no compiler that supports f90 extensions was found on the system.])
	fi

	])
	fi
else	
VALID_FORTRAN=no
fi

AM_CONDITIONAL([ENABLE_FORTRAN],[test x$VALID_FORTRAN = xyes])


#idl bindings
AC_ARG_ENABLE([idl],[AS_HELP_STRING([--disable-idl],[build without the idl bindings])],[enable_idl=$enableval],[enable_idl=check])

VALID_IDL=

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
			AC_MSG_RESULT([yes, in $IDL_PATH])
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
#IDL 7.1 might cause problems on Darwin -> makes distinction between i386 and x86_64... to be verified
		    i686-*-darwin*)
		      IDL_LIBBIN=$RSIIDL_BINDIR/bin.darwin.i386
		      IDL_CFLAGS="-fPIC"
		      IDL_LD_FLAGS="-flat_namespace -undefined suppress -bundle"
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
		AC_CHECK_FILES([$IDL_LIBBIN/libidl.so $IDL_LIBBIN/libidl.dylib],[RSIIDL_HASBIN=yes],[RSIIDL_HASBIN=no])	
		if test "x$RSIIDL_HASBIN" = xno ; then
			AC_MSG_ERROR([libidl.so could not be found. This means that you have either a corrupt or custom installation of idl. In the first case, set --enable-rsiidl-libdir to match the location of the IDL libraries])
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

AM_CONDITIONAL([ENABLE_IDL],[test x$VALID_IDL = xyes])


AC_CONFIG_FILES([Makefile src/Makefile include/Makefile src/prdata/Makefile example/Makefile idl/Makefile])
AC_CONFIG_HEADERS([config.h])



AC_OUTPUT
