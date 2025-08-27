#Copyright (c) 2010-2015, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

SUBDIRS = . tests

java_source = \
	Xraylib.java \
	compoundData.java \
	compoundDataBase.java \
	compoundDataNIST.java \
	radioNuclideData.java \
	Crystal_Struct.java \
	Crystal_Atom.java

if ENABLE_JAVA

AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include -I${top_srcdir}/src

#java bindings for xraylib
xrljavadir = $(prefix)/share/xraylib/java
xrljava_DATA = xraylib-$(VERSION).jar xraylib-$(VERSION)-javadoc.jar xraylib-$(VERSION)-sources.jar

noinst_PROGRAMS = prdata_java
prdata_java_SOURCES = pr_data_java.c
prdata_java_LDADD = ../src/libprdata.la
prdata_java_CFLAGS = $(AM_CFLAGS) $(ARCHFLAGS) $(WSTRICT_CFLAGS)

all-local: xraylib-$(VERSION).jar xraylib-$(VERSION)-javadoc.jar xraylib-$(VERSION)-sources.jar

xraylib-$(VERSION)-javadoc.jar: xraylib-$(VERSION).jar
	
xraylib-$(VERSION)-sources.jar: xraylib-$(VERSION).jar

xraylib-$(VERSION).jar: xraylib.dat $(java_source)
	$(srcdir)/gradlew build sourcesJar javadocJar -x test

xraylib.dat: prdata_java$(EXEEXT)
	$(AM_V_GEN) $(WINE) ./prdata_java$(EXEEXT) ${top_srcdir}

clean-local:
	rm -rf xraylib-$(VERSION).jar xraylib-$(VERSION)-javadoc.jar xraylib-$(VERSION)-sources.jar  xraylib.dat com build .gradle META-INF xrlexample7.class prdata_java.dSYM

check-local:
	$(srcdir)/gradlew test

#rm -rf build/test-results

publishToMavenLocal:
	$(srcdir)/gradlew publishToMavenLocal

publish:
	$(srcdir)/gradlew publishToMavenCentral

endif

EXTRA_DIST = \
	$(java_source) \
	gradlew \
	gradle
