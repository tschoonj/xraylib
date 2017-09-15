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
	$(MAKE) -C python python
	lib /machine:i386 /def:libxrl-$(LIB_CURRENT_MINUS_AGE).def
	implib.exe -a -c libxrl-$(LIB_CURRENT_MINUS_AGE)-bcc.lib libxrl-$(LIB_CURRENT_MINUS_AGE).dll
	iscc.exe "-dLIB_CURRENT_MINUS_AGE=$(LIB_CURRENT_MINUS_AGE)" "-dMyAppVersion=$(VERSION)" "-dabs_top_srcdir_win=$(abs_top_srcdir_win)" "-dabs_top_builddir_win=$(abs_top_builddir_win)" "${srcdir}/xraylib.iss"
	rm libxrl-$(LIB_CURRENT_MINUS_AGE).def
	$(MAKE) -C python/cli_py2exe nsis
endif

if OS_WINDOWS_64
windows:
	$(MAKE) common
	lib /machine:X64 /def:libxrl-$(LIB_CURRENT_MINUS_AGE).def
	iscc.exe "-dXRL64" "-dLIB_CURRENT_MINUS_AGE=$(LIB_CURRENT_MINUS_AGE)" "-dMyAppVersion=$(VERSION)" "-dabs_top_srcdir_win=$(abs_top_srcdir_win)" "-dabs_top_builddir_win=$(abs_top_builddir_win)" "${srcdir}/xraylib.iss"
	rm libxrl-$(LIB_CURRENT_MINUS_AGE).def
endif

common:
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
	unix2dos -n ${top_srcdir}/pascal/xraylib.pas xraylib.pas
	unix2dos -n ${top_srcdir}/example/xrlexample14.pas xrlexample14.pas



EXTRA_DIST = License.rtf  README xraylib.iss
