#Copyright (c) 2010, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


if ENABLE_JAVA

#java bindings for xraylib
xrljavadir = $(prefix)/share/xraylib/java
xrljava_JAVA = xraylib.java xraylibConstants.java xraylibJNI.java 
if OS_WINDOWS
xrljava_LTLIBRARIES = xraylib.la 
else
xrljava_LTLIBRARIES = libxraylib.la 
endif
libxraylib_la_CFLAGS = -I${top_srcdir}/include $(JAVACFLAGS) $(JAVAINC)
libxraylib_la_LIBADD = ../src/libxrl.la
nodist_libxraylib_la_SOURCES = xraylib_wrap.c
libxraylib_la_LDFLAGS = -avoid-version -module -shared -export-dynamic -shrext $(JAVASO)

xraylib_la_CFLAGS = -I${top_srcdir}/include $(JAVACFLAGS) $(JAVAINC)
xraylib_la_LIBADD = ../src/libxrl.la
nodist_xraylib_la_SOURCES = xraylib_wrap.c
xraylib_la_LDFLAGS = -avoid-version -module -shared -export-dynamic -shrext $(JAVASO)




xraylib_wrap.c: ${top_srcdir}/src/xraylib.i ${top_srcdir}/include/xraylib*h
	$(SWIG) -I${top_srcdir}/include -includeall -java -o xraylib_wrap.c ${top_srcdir}/src/xraylib.i

xraylib.java: xraylib_wrap.c
xraylibConstants.java: xraylib_wrap.c
xraylibJNI.java: xraylib_wrap.c

clean-local:
	rm -rf *.java xraylib_wrap.c

endif

