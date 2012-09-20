
#Copyright (c) 2009, 2010, 2011, 2012, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

xraylibincludedir=${includedir}/xraylib

AM_CPPFLAGS = -I${top_srcdir}/include -I${top_builddir}/include -I.. -I${top_srcdir}/src

if ENABLE_IDL
all:
	$(MAKE) idlxrl


#I'm not sure how portable this .libs link location is...
#Maybe this could be rewritten using libtool...
idlxrl:	xraylib_idl.c ../src/libxrl.la
	$(CC) -c $(AM_CPPFLAGS) -I@RSIIDL_DIR@/external/include $(CFLAGS) @IDL_CFLAGS@ ${srcdir}/xraylib_idl.c
	gcc @IDL_LD_FLAGS@ -o libxrlidl.so xraylib_idl.o -L${top_builddir}/src/.libs -lxrl


#install the IDL bindings specific files
install-exec-local:
	$(MKDIR_P) $(DESTDIR)$(prefix)/dlm
	$(INSTALL) libxrlidl.so ${srcdir}/libxrlidl.dlm $(DESTDIR)$(prefix)/dlm
	$(MKDIR_P) $(DESTDIR)$(prefix)/pro	
	$(INSTALL) $(top_srcdir)/idl/xraylib_auger.pro $(top_srcdir)/idl/xraylib_shells.pro $(top_srcdir)/idl/xraylib_help.pro $(top_srcdir)/idl/xraylib_lines.pro $(top_srcdir)/idl/xraylib.pro $(DESTDIR)$(prefix)/pro

uninstall-local:
	rm -f $(DESTDIR)$(prefix)/dlm/libxrlidl.so $(DESTDIR)$(prefix)/dlm/libxrlidl.dlm
	rm -f $(DESTDIR)$(prefix)/pro/xraylib_auger.pro $(DESTDIR)$(prefix)/pro/xraylib_shells.pro $(DESTDIR)$(prefix)/pro/xraylib_help.pro  $(DESTDIR)$(prefix)/pro/xraylib_lines.pro $(DESTDIR)$(prefix)/pro/xraylib.pro


clean-local:
	rm -f libxrlidl.so 
	rm -f xraylib_idl.o 
	rm -rf libxrlidl.so.dSYM
endif

EXTRA_DIST = xraylib_idl.c libxrlidl.dlm xraylib_auger.pro xraylib_shells.pro xraylib_help.pro xraylib_lines.pro xraylib.pro


