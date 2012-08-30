#Copyright (c) 2010, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

SUBDIRS = python dotNet

nsis:
	$(SWIG) -I${top_srcdir}/include -includeall -o xraylib_wrap.c -python ${top_srcdir}/src/xraylib.i
	$(MAKE) -C python python
	cp ../src/libxrl-$(LIB_CURRENT_MINUS_AGE).def .
	lib /machine:i386 /def:libxrl-$(LIB_CURRENT_MINUS_AGE).def
	rm libxrl-$(LIB_CURRENT_MINUS_AGE).def
	cp ../src/.libs/libxrl-$(LIB_CURRENT_MINUS_AGE).dll .
	implib.exe -a -c libxrl-$(LIB_CURRENT_MINUS_AGE)-bcc.lib libxrl-$(LIB_CURRENT_MINUS_AGE).dll
	cp ../idl/xraylib_idl.c .
	echo "!MAKE_DLL.COMPILER_NAME='GCC'" > batch.pro
	echo "!MAKE_DLL.CC='gcc.exe %X -fPIC -I\"C:\Program Files (x86)\ITT\IDL\IDL80\external\include\" -c -D_REENTRANT %C -o %O'" >> batch.pro
	echo "CD,'C:\MinGW\msys\1.0\home\Administrator\xraylib-$(VERSION)\nsis'" >> batch.pro
	echo "make_dll,'xraylib_idl','libxrlidl','IDL_Load',INPUT_DIR='.',EXTRA_CFLAGS='-I\"C:\MinGW\msys\1.0\home\Administrator\xraylib-$(VERSION)\include\" -I\"C:\MinGW\msys\1.0\home\Administrator\xraylib-$(VERSION)\src\"',EXTRA_LFLAGS='\"libxrl-3.lib\" /LIBPATH:\"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.0A\Lib\"',OUTPUT='.', COMPILE_DIRECTORY='.'" >> batch.pro
	echo "exit" >> batch.pro
	IDL_PATH="C:\MinGW\msys\1.0\home\Administrator\xraylib-$(VERSION)\nsis" idl batch
	makensis xraylib.nsi
	$(MAKE) -C python/cli_py2exe nsis


EXTRA_DIST = xraylib.nsi.in License.rtf  README EnvVarUpdate.nsh
