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

#headers check
AC_CHECK_HEADERS([math.h stdio.h stdlib.h string.h])



#libraries check
AC_CHECK_LIB([m],[log],[],[AC_MSG_ERROR([the log function must be present in libm])])
AC_CHECK_LIB([m],[exp],[],[AC_MSG_ERROR([the exp function must be present in libm])])
AC_CHECK_LIB([m],[cos],[],[AC_MSG_ERROR([the cos function must be present in libm])])
AC_CHECK_LIB([m],[sin],[],[AC_MSG_ERROR([the sin function must be present in libm])])



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


AC_CONFIG_FILES([Makefile src/Makefile include/Makefile src/prdata/Makefile example/Makefile])
AC_CONFIG_HEADERS([config.h])



AC_OUTPUT
