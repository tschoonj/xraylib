#Copyright (c) 2009,2010, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#Makefile.am for the examples. These can be used to verify that the compilation went ok by typing in "make check" 


AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include


#since the C lib will always be built, always check xrlexample1
check_PROGRAMS = xrlexample1$(EXEEXT)
if ENABLE_FORTRAN
  check_PROGRAMS += xrlexample3
endif
check_SCRIPTS=
if ENABLE_PERL
  check_SCRIPTS += xrlexample2.sh
endif
if ENABLE_IDL
  check_SCRIPTS += xrlexample4.sh
endif
if ENABLE_PYTHON
  check_SCRIPTS += xrlexample5.sh
endif
if ENABLE_CXX
  check_PROGRAMS += xrlexample6$(EXEEXT)
endif
if ENABLE_JAVA
  check_SCRIPTS += xrlexample7.sh
endif

xrlexample1_SOURCES = xrlexample1.c
xrlexample1_LDADD = ../src/libxrl.la
xrlexample1_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include
xrlexample1_CFLAGS = $(ARCHFLAGS)

if ENABLE_FORTRAN
  xrlexample3_SOURCES = xrlexample3.f90
  xrlexample3_LDADD = ../src/libxrlf03.la
  xrlexample3_CPPFLAGS = -I../src
  xrlexample3_FFLAGS = ${xrlexample3_CPPFLAGS} $(ARCHFLAGS)
  xrlexample3_FCFLAGS = ${xrlexample3_CPPFLAGS}
endif

if ENABLE_CXX
  xrlexample6_SOURCES = xrlexample6.cpp
  xrlexample6_LDADD = ../src/libxrl.la
endif



if ENABLE_CROSS
TESTS_ENVIRONMENT = path="${top_builddir}/src/.libs" $(WINE)
endif

TESTS = $(check_PROGRAMS) $(check_SCRIPTS)

EXTRA_DIST = xrlexample4.pro xrlexample2.pl xrlexample5.py xrlexample7.java


#test the idl bindings using this script
xrlexample4.sh: 
	echo "Testing IDL bindings"
	echo "unset IDL_STARTUP" > xrlexample4.sh
if !ABS_SRC_BUILD_EQUAL
	echo "ln -s ${top_srcdir}/src/libxrlidl.dlm ${top_builddir}/src/libxrlidl.dlm" >> xrlexample4.sh
endif
	echo "LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" IDL_PATH=\"${top_srcdir}/example:${top_srcdir}/idl:<IDL_DEFAULT>\" IDL_DLM_PATH=\"${top_builddir}/src:<IDL_DEFAULT>\" @RSIIDL_BINDIR@/idl xrlexample4" >> xrlexample4.sh
if !ABS_SRC_BUILD_EQUAL
	echo "rm -f ${top_builddir}/src/libxrlidl.dlm" >> xrlexample4.sh
endif
	chmod +x xrlexample4.sh	

#test the perl bindings
xrlexample2.sh:
	echo "Testing perl bindings"
#	echo "unset PERL5LIB" > xrlexample2.sh
#	echo "unset PERLLIB" >> xrlexample2.sh
	echo "LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" PERL5LIB=\"${top_builddir}/src/perl/blib/lib:${top_builddir}/src/perl/blib/arch\" $(PERL) ${top_srcdir}/example/xrlexample2.pl" >> xrlexample2.sh
	chmod +x xrlexample2.sh

#test the python bindings
xrlexample5.sh:
	echo "Testing python bindings"
	echo "LD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs\" PYTHONPATH=\"${top_builddir}/python\" $(PYTHON) ${top_srcdir}/example/xrlexample5.py" >> xrlexample5.sh
	chmod +x xrlexample5.sh

#test the java bindings
xrlexample7.sh:
	echo "Testing java bindings"
	echo "$(JAVAC) -classpath \"${top_builddir}/java:.:..\" -d \"${builddir}\" ${top_srcdir}/example/xrlexample7.java" >> xrlexample7.sh
	echo "LD_LIBRARY_PATH=\"${top_builddir}/src/.libs:${top_builddir}/java/\" DYLD_LIBRARY_PATH=\"${top_builddir}/src/.libs:${top_builddir}/java/\" $(JAVA) -Djava.library.path=\"${top_builddir}/java\" -classpath \"${top_builddir}/java:.:..\" xrlexample7" >> xrlexample7.sh
	chmod +x xrlexample7.sh



clean-local:
	rm -rf xrlexample4.sh xrlexample2.sh xrlexample5.sh xrlexample7.sh xrlexample7.class
