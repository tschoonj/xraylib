#!/usr/bin/env bash

set -e
set -x

export CYTHON=cython2

cd $APPVEYOR_BUILD_FOLDER

autoreconf -fi

./configure --disable-all-bindings CFLAGS="-std=c89 -pedantic"
make
make check
make distclean

./configure --disable-all-bindings --disable-shared --enable-static
make
make check
make distclean

./configure --disable-python --disable-python-numpy --enable-fortran2003
make
make check
#make distcheck
make distclean

export PYTHON=python2
./configure --enable-python --enable-python-numpy
make
make check
#make distcheck
make distclean

export PYTHON=python3
./configure --enable-python --enable-python-numpy
make
make check
#make distcheck
make distclean
