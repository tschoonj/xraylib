#!/bin/bash

#functions

make_python(){

PYTHON_INCLUDE=/usr/include/python2.5

echo -n "Python Include Directory? (${PYTHON_INCLUDE}) "
read arg
if [ ! $arg = '' ]; then PYTHON_INCLUDE=$arg; fi

if [ ! -d $PYTHON_INCLUDE ]; then
    echo "Directory $PYTHON_INCLUDE does not exist"
    echo
    exit
elif [ ! -f ${PYTHON_INCLUDE}/Python.h ]; then
    echo "File Python.h does not exist in Directory $PYTHON_INCLUDE"
    echo
    exit
fi

export PYTHON_INCLUDE

make -f ${MAKEFILE} -e python

}

make_perl(){
make -f ${MAKEFILE} clean
make -f ${MAKEFILE} perl
}


make_shared(){
#future option: allow different compilers
 make -f ${MAKEFILE} all
 

}

make_sharedplusplus(){
make -f ${MAKEFILE} clean
CC=g++ PROGNAME=libxrl++ make -f ${MAKEFILE} -e all
}

make_f2003(){
 printf "Supported compilers:\n"
 printf "\tIntel Fortran (version 10.0 and higher) [1]\n"
 printf "\tGFortran (version 4.3.0 and higher)     [2]\n"
 printf "\tG95                                     [3]\n"
 printf "Enter the number of the desired compiler or 0 \nto build the regular libraries: "
 read comp
 case "$comp" in
   1 ) F2003=ifort FFLAGS="-module ../include" make -f ${MAKEFILE} fortran ;;
   2 )  rm -f xraylib.mod
	F2003=gfortran make -f ${MAKEFILE} fortran
       	cp xraylib.mod ../include/        
	 ;;
   3 ) F2003=g95 FFLAGS="-fmod=../include" make -f ${MAKEFILE} fortran ;;
   * ) make_shared
 esac
}

make_idl(){
IDL_INCLUDE=/usr/local/idl/external/
if [ $KERNEL = "Darwin" ] ; then
 IDL_INCLUDE=/Applications/itt/idl/external/
fi
echo -n "IDL Export Include Directory? (${IDL_INCLUDE}) "
read arg
if [ ! $arg = '' ]; then IDL_INCLUDE=$arg; fi
 
if [ ! -d $IDL_INCLUDE ]; then
    echo "Directory $IDL_INCLUDE does not exist"
    echo
    exit
elif [ ! -f ${IDL_INCLUDE}/export.h ]; then
    echo "File export.h does not exist in Directory $IDL_DIR"
    echo
    exit
fi    

export IDL_INCLUDE

make -f ${MAKEFILE} -e idl

}

VERSION=xraylib_v2.11
if [ ! $INSTALL_DIR ]; then
    INSTALL_DIR=$HOME
fi

if [ -d ${INSTALL_DIR}/.xraylib ]; then
    if [ -e ${INSTALL_DIR}/.xraylib_old ]; then
        rm -fr ${INSTALL_DIR}/.xraylib_old
    fi
    mv ${INSTALL_DIR}/.xraylib ${INSTALL_DIR}/.xraylib_old
fi

if [ ! -d ${INSTALL_DIR}/.${VERSION} ]; then
    mkdir ${INSTALL_DIR}/.${VERSION}
fi

cd src


DARWIN_FLAG=0
PYTH_FLAG=1
IDL_FLAG=1
PERL_FLAG=1
F2003_FLAG=1
CPLUSPLUS_FLAG=1
INLINE_FLAG=1

printf "\nThis script will install XrayLib\nPlease follow the instructions\n"
printf "\nTwo versions are available.\n"
printf "\t1) Physical data is hard coded.\n\tCompiling this on slow machines is discouraged (default)\n"
printf "\t2) Physical data will be initialized at runtime\n\tInitialization may take up to 10 seconds\n"
printf "Please select which one you would like to install:"
read yn
case "$yn" in
  2 ) INLINE_FLAG=0 && MAKEFILE=Makefile.external ;;
  * ) INLINE_FLAG=1 && MAKEFILE=Makefile.inline ;;
esac



#check for OS 
KERNEL=`uname -s`

if [ $KERNEL = "Darwin" ] ; then
	printf "Mac OS X has been detected\n"
	printf "Disabling Python extensions\n"
	DARWIN_FLAG=1
	PYTH_FLAG=0
fi




make -f ${MAKEFILE} clean

if [ $DARWIN_FLAG -eq 0 ] ; then

echo
echo 'Do you want to install the command line script (requires python)'
echo -n 'and the python module ([y]/n)? '
read yn
case "$yn" in
  N* | n* ) PYTH_FLAG=0;;
  *) make_python ;; 
esac
fi

echo -n 'Do you want to install the IDL module ([y]/n)? '
read yn
case "$yn" in
  N* | n* ) IDL_FLAG=0;;
  *) make_idl;;
esac

echo
echo 'Do you want to install the Perl module? (requires Perl) ([y]/n)? '
read yn
case "$yn" in
  N* | n* ) PERL_FLAG=0;;
  *) make_perl ;; 
esac

#shared and static C libraries will always be installed...

echo
echo 'Do you want to install the Fortran bindings?'
echo '(requires F2003 compiler) ([y]/n)? '
read yn
case "$yn" in
  N* | n* ) F2003_FLAG=0;;
  *) F2003_FLAG=1 ;; 
esac


if [ $F2003_FLAG = 0 ] ; then
	make_shared
else
	make_f2003
fi


#todo: IDL (DLM), octave, environment info stuff





echo
echo 'Do you want to install the C++ shared library? (requires C++ compiler) ([y]/n)? '
read yn
case "$yn" in
  N* | n* ) CPLUSPLUS_FLAG=0;;
  *) make_sharedplusplus ;; 
esac






#./make_shared.sh
#./make_shared++.sh

cd ..
echo "Installing files on directory ${INSTALL_DIR}/.${VERSION}/"
cp -r * ${INSTALL_DIR}/.${VERSION}/.
echo
echo "Linking directory ${INSTALL_DIR}/.${VERSION}/"
echo "to directory ${INSTALL_DIR}/.xraylib"
ln -s ${INSTALL_DIR}/.${VERSION} ${INSTALL_DIR}/.xraylib
echo

echo 'If you use the bash shell add the following lines to the file .bashrc'
echo '(or bashrc_private if you use it) in your home directory:'
echo
echo "export XRAYLIB_DIR=${INSTALL_DIR}/.xraylib"
echo 'export PATH=${PATH}:${XRAYLIB_DIR}/bin:'
if [ $PYTH_FLAG -eq 1 ]; then
    echo 'export PYTHONPATH=${PYTHONPATH}:${XRAYLIB_DIR}/bin:'
fi

if [ $PERL_FLAG -eq 1 ] ; then
  PERL_LIB=`find perl -name 'xraylib.pm'`
  PERL_LIB=`dirname $PERL_LIB`
  printf "export PERL5LIB=%s:%s/%s\n" '${PERL5LIB}' '${XRAYLIB_DIR}' $PERL_LIB
fi

if [ $IDL_FLAG -eq 1 ] ; then
	echo 'export IDL_DLM_PATH="<IDL_DEFAULT>:${XRAYLIB_DIR}/idl"'
fi


echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${XRAYLIB_DIR}/lib'
echo 'alias xrl=xraylib'
echo
echo 'If you use the tcsh shell add the following lines to the file .cshrc'
echo '(or cshrc_private if you use it) in your home directory:'
echo
echo "setenv XRAYLIB_DIR ${INSTALL_DIR}/.xraylib"
echo 'setenv PATH ${PATH}:${XRAYLIB_DIR}/bin:'
if [ $PYTH_FLAG -eq 1 ]; then
    echo 'if ( ! $?PYTHONPATH ) setenv PYTHONPATH'
    echo 'setenv PYTHONPATH ${PYTHONPATH}:${XRAYLIB_DIR}/bin:'
fi
if [ $PERL_FLAG -eq 1 ] ; then
  PERL_LIB=`find perl -name 'xraylib.pm'`
  PERL_LIB=`dirname $PERL_LIB`
  echo 'if ( ! $?PERL5LIB ) setenv PERL5LIB'
  printf "setenv PERL5LIB %s:%s/%s\n" '${PERL5LIB}' '${XRAYLIB_DIR}' $PERL_LIB
fi
if [ $IDL_FLAG -eq 1 ] ; then
	echo 'setenv IDL_DLM_PATH "<IDL_DEFAULT>:"${XRAYLIB_DIR}/idl"'
fi
echo 'if ( ! $?LD_LIBRARY_PATH) setenv LD_LIBRARY_PATH'
echo 'setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${XRAYLIB_DIR}/lib'
echo 'alias xrl xraylib'
echo






