#!/usr/bin/env bash

set -e
set -x

export CYTHON=cython
export PATH="/C/Program Files (x86)/Inno Setup 6:$HOME/install/bin:$PATH"

cd $APPVEYOR_BUILD_FOLDER

autoreconf -fi

./configure --disable-all-bindings CFLAGS="-std=c89 -pedantic"
make
make check
make windows
make distclean

./configure --disable-all-bindings --disable-shared --enable-static
make
make check
make distclean

./configure --disable-python --disable-python-numpy --enable-fortran2003 --enable-pascal FPC=/c/lazarus/fpc/3.0.4/bin/x86_64-win64/fpc
make
make check
make distclean

./configure --enable-python --enable-python-numpy PYTHON=python3
make
make check
make distclean

./configure
make distcheck FPC=/c/lazarus/fpc/3.0.4/bin/x86_64-win64/fpc
