#Copyright (c) 2010, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

SUBDIRS = python

if OS_WINDOWS_32
windows:
	$(MAKE) common
	lib /machine:i386 /def:libxrl-$(LIB_CURRENT_MINUS_AGE).def
	implib.exe -a -c libxrl-$(LIB_CURRENT_MINUS_AGE)-bcc.lib libxrl-$(LIB_CURRENT_MINUS_AGE).dll
	iscc.exe "-dLIB_CURRENT_MINUS_AGE=$(LIB_CURRENT_MINUS_AGE)" "-dMyAppVersion=$(VERSION)" "-dabs_top_srcdir_win=$(abs_top_srcdir_win)" "-dabs_top_builddir_win=$(abs_top_builddir_win)" "${srcdir}/xraylib.iss"
	rm libxrl-$(LIB_CURRENT_MINUS_AGE).def
	$(MAKE) -C python/cli_py2exe nsis
endif

if OS_WINDOWS_64
windows:
	$(MAKE) common
	$(MAKE) idl64
	lib /machine:X64 /def:libxrl-$(LIB_CURRENT_MINUS_AGE).def
	iscc.exe "-dXRL64" "-dLIB_CURRENT_MINUS_AGE=$(LIB_CURRENT_MINUS_AGE)" "-dMyAppVersion=$(VERSION)" "-dabs_top_srcdir_win=$(abs_top_srcdir_win)" "-dabs_top_builddir_win=$(abs_top_builddir_win)" "${srcdir}/xraylib.iss"
	rm libxrl-$(LIB_CURRENT_MINUS_AGE).def
endif

common:
	$(MAKE) -C python python
	cp ../src/libxrl-$(LIB_CURRENT_MINUS_AGE).def .
	cp ../src/.libs/libxrl-$(LIB_CURRENT_MINUS_AGE).dll .
	unix2dos -n ${top_srcdir}/README README.txt
	unix2dos -n ${top_srcdir}/AUTHORS AUTHORS.txt
	unix2dos -n ${top_srcdir}/BUGS BUGS.txt
	unix2dos -n ${top_srcdir}/TODO TODO.txt
	unix2dos -n ${top_srcdir}/Changelog Changelog.txt
	unix2dos -n ${top_srcdir}/doc/xraybanner.txt xraybanner.txt
	unix2dos -n ${top_srcdir}/doc/xraydoc.txt xraydoc.txt
	unix2dos -n ${top_srcdir}/doc/xrayfunc.txt xrayfunc.txt
	unix2dos -n ${top_srcdir}/doc/xrayhelp.txt xrayhelp.txt
	unix2dos -n ${srcdir}/README README2.txt
	unix2dos -n ${top_srcdir}/example/xrlexample8.cs xrlexample8.cs
	unix2dos -n ${top_srcdir}/example/xrlexample1.c xrlexample1.c
	unix2dos -n ${top_srcdir}/example/xrlexample6.cpp xrlexample6.cpp
	unix2dos -n ${top_srcdir}/include/xraylib.h xraylib.h
	unix2dos -n ${top_srcdir}/include/xraylib-parser.h xraylib-parser.h
	unix2dos -n ${top_srcdir}/include/xraylib-lines.h xraylib-lines.h
	unix2dos -n ${top_srcdir}/include/xraylib-shells.h xraylib-shells.h
	unix2dos -n ${top_srcdir}/include/xraylib-auger.h xraylib-auger.h
	unix2dos -n ${top_srcdir}/include/xraylib-crystal-diffraction.h xraylib-crystal-diffraction.h
	unix2dos -n ${top_srcdir}/include/xraylib-defs.h xraylib-defs.h
	unix2dos -n ${top_srcdir}/include/xraylib-nist-compounds.h xraylib-nist-compounds.h
	unix2dos -n ${top_srcdir}/include/xraylib-radionuclides.h xraylib-radionuclides.h
	unix2dos -n ${top_srcdir}/example/xrlexample5.py xrlexample5.py
	unix2dos -n ${top_srcdir}/example/xrlexample13.py xrlexample13.py
	unix2dos -n ${srcdir}/python/cli/xraylib-cli.py xraylib-cli.py
	unix2dos -n ${srcdir}/python/cli/xraymessages.py xraymessages.py
	unix2dos -n ${srcdir}/python/cli/xrayhelp.py xrayhelp.py
	unix2dos -n ${top_srcdir}/example/xrlexample4.pro xrlexample4.pro
	unix2dos -n ${top_srcdir}/idl/libxrlidl.dlm libxrlidl.dlm 
	unix2dos -n ${top_srcdir}/idl/xraylib.pro xraylib.pro 
	unix2dos -n ${top_srcdir}/idl/xraylib_help.pro xraylib_help.pro 
	unix2dos -n ${top_srcdir}/idl/xraylib_lines.pro xraylib_lines.pro 
	unix2dos -n ${top_srcdir}/idl/xraylib_shells.pro xraylib_shells.pro 
	unix2dos -n ${top_srcdir}/idl/xraylib_auger.pro xraylib_auger.pro 
	unix2dos -n ${top_srcdir}/idl/xraylib_nist_compounds.pro xraylib_nist_compounds.pro 
	unix2dos -n ${top_srcdir}/idl/xraylib_radionuclides.pro xraylib_radionuclides.pro 
	unix2dos -n ${top_srcdir}/pascal/xraylib.pas xraylib.pas
	unix2dos -n ${top_srcdir}/example/xrlexample14.pas xrlexample14.pas

idl32:
	cp ${top_srcdir}/idl/xraylib_idl.c .
	echo "!MAKE_DLL.COMPILER_NAME='GCC'" > batch.pro
	echo "!MAKE_DLL.CC='gcc.exe %X -fPIC -I\"C:\Program Files (x86)\ITT\IDL71\external\include\" -c -D_REENTRANT %C -o %O'" >> batch.pro
	echo "CD,'C:\MinGW\msys\1.0\home\Administrator\xraylib\build\windows'" >> batch.pro
	echo "make_dll,'xraylib_idl','libxrlidl','IDL_Load',INPUT_DIR='.',EXTRA_CFLAGS='-I\"C:\MinGW\msys\1.0\home\Administrator\xraylib\include\" -I\"C:\MinGW\msys\1.0\home\Administrator\xraylib\src\"',EXTRA_LFLAGS='\"libxrl-$(LIB_CURRENT_MINUS_AGE).lib\" /LIBPATH:\"C:\Program Files (x86)\Microsoft SDKs\Windows\v7.0A\Lib\"',OUTPUT='.', COMPILE_DIRECTORY='.', /SHOW_ALL, /VERBOSE" >> batch.pro
	echo "exit" >> batch.pro
	IDL_PATH="C:\MinGW\msys\1.0\home\Administrator\xraylib\build\windows" idl batch

idl64:
	cp ${top_srcdir}/idl/xraylib_idl.c .
	echo "!MAKE_DLL.COMPILER_NAME='GCC'" > batch.pro
	echo "!MAKE_DLL.CC='gcc.exe %X -fPIC -I\"C:\Exelis\IDL83\external\include\" -c -D_REENTRANT %C -o %O'" >> batch.pro
	echo "CD,'C:\msys\1.0\home\schoon\github\xraylib\build\windows'" >> batch.pro
	echo "make_dll,'xraylib_idl','libxrlidl','IDL_Load',INPUT_DIR='.',EXTRA_CFLAGS='-I\"C:\msys\1.0\home\schoon\github\xraylib\include\" -I\"C:\msys\1.0\home\schoon\github\xraylib\src\"',EXTRA_LFLAGS='\"libxrl-$(LIB_CURRENT_MINUS_AGE).lib\" /LIBPATH:\"C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\lib\amd64\" /LIBPATH:\"C:\Program Files\Microsoft SDKs\Windows\v7.1\Lib\x64\"',OUTPUT='.', COMPILE_DIRECTORY='.', /SHOW_ALL, /VERBOSE, DLL_PATH=mydll" >> batch.pro
	echo "print,mydll" >> batch.pro
	echo "exit" >> batch.pro
	IDL_PATH="C:\msys\1.0\home\schoon\github\xraylib\build\windows" /C/Exelis/IDL83/bin/bin.x86_64/idl batch



EXTRA_DIST = License.rtf  README xraylib.iss
