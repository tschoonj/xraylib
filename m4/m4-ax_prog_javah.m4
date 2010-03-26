# ===========================================================================
#       http://www.gnu.org/software/autoconf-archive/ax_prog_javah.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_PROG_JAVAH
#
# DESCRIPTION
#
#   AX_PROG_JAVAH tests the availability of the javah header generator and
#   looks for the jni.h header file. If available, JAVAH is set to the full
#   path of javah and CPPFLAGS is updated accordingly.
#
# LICENSE
#
#   Copyright (c) 2008 Luc Maisonobe <luc@spaceroots.org>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved. This file is offered as-is, without any
#   warranty.

#serial 5

AU_ALIAS([AC_PROG_JAVAH], [AX_PROG_JAVAH])
AC_DEFUN([AX_PROG_JAVAH],[
#AC_REQUIRE([AC_CANONICAL_SYSTEM])dnl
AC_REQUIRE([AC_PROG_CPP])dnl
AC_PATH_PROG(JAVAH,javah)
if test "x$ac_cv_path_JAVAH" != x ; then
    AC_MSG_CHECKING([for jni.h and jni_md.h headers])
    if test -L $ac_cv_path_JAVAH ; then
	while test -L $ac_cv_path_JAVAH 
	 do 
dnl	 ac_cv_path_JAVAH=`readlink $ac_cv_path_JAVAH`
dnl 	since readlink doesn't appear on all systems (I hate Solaris...), let's use an awk trick
 	 ac_cv_path_JAVAH=`ls -l $ac_cv_path_JAVAH | awk -F\> '{print $NF}'` 
	 done
    fi
    ac_save_CPPFLAGS="$CPPFLAGS"
changequote(, )dnl
    ac_dir=`echo $ac_cv_path_JAVAH | sed 's,\(.*\)/[^/]*/[^/]*$,\1/,'`
    ac_machdep=`echo $host_os | sed 's,[-0-9].*,,' | sed 's,cygwin,win32,'`
changequote([, ])dnl
    JAVACPPFLAGS="$ac_save_CPPFLAGS -I$ac_dir/include -I$ac_dir/Headers -I$ac_dir/include/$ac_machdep"
    CPPFLAGS=$JAVACPPFLAGS
    AC_PREPROC_IFELSE(
    	[AC_LANG_PROGRAM([[#include <jni.h>]],[[#include <jni_md.h>]])],
	AC_MSG_RESULT([ok])
	CPPFLAGS="$ac_save_CPPFLAGS",
	AC_MSG_RESULT([could not find jni.h]
	JAVACPPFLAGS=
	CPPFLAGS="$ac_save_CPPFLAGS",
	)
    )
else
    AC_MSG_WARN([Could not locate javah])
    JAVACPPFLAGS=
fi])
