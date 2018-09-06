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

exit 0

./configure --disable-python --disable-python-numpy
make
make check
make distcheck
make distclean

export PYTHON=python2
./configure --enable-python --enable-python-numpy
make
make check
make distcheck
make distclean

export PYTHON=python3
./configure --enable-python --enable-python-numpy
make
make check
make distcheck
make distclean
