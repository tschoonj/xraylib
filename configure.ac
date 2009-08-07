AC_INIT([xraylib],[2.12],[Tom.Schoonjans@UGent.be])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])

LIB_CURRENT=1
LIB_REVISION=0
LIB_AGE=0
AC_SUBST(LIB_CURRENT)
AC_SUBST(LIB_REVISION)
AC_SUBST(LIB_AGE)

AC_PROG_LIBTOOL
AC_PROG_CC

#headers check
AC_CHECK_HEADERS([math.h stdio.h stdlib.h string.h stdbool.h])


#libraries check
AC_CHECK_LIB([m],[log])
AC_CHECK_LIB([m],[exp])
AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([m],[sin])


AC_CONFIG_FILES([Makefile src/Makefile include/Makefile src/prdata/Makefile example/Makefile])
AC_CONFIG_HEADERS([config.h])



AC_OUTPUT
