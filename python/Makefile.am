
#Copyright (c) 2009, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



#python bindings for xraylib



EXTRA_DIST = xraylib


if ENABLE_PYTHON
all:
	$(MAKE) pythonxrl



pythonxrl: ${top_builddir}/src/xraylib.i
	$(SWIG) -Wall -includeall -o xraylib_wrap.c -python ${top_builddir}/src/xraylib.i
	CC=$(CC) CFLAGS="$(ARCHFLAGS)" $(PYTHON) xrlsetup.py build_ext --inplace




#I like to use my own commands for the install: although xrlsetup.py can handle this, I want to do it myself. Libtool would also be a possibility as is detailed in the automake manual.
#MKDIR_P's are necessary for distcheck
install-exec-local:
	$(MKDIR_P) $(DESTDIR)$(pythondir)
	$(MKDIR_P) $(DESTDIR)$(pyexecdir)
	$(MKDIR_P) $(DESTDIR)$(bindir)
	$(INSTALL) xraylib.py xrayhelp.py xraymessages.py $(DESTDIR)$(pythondir)
	$(INSTALL) _xraylib.* $(DESTDIR)$(pyexecdir)
	$(INSTALL) ${top_srcdir}/python/xraylib $(DESTDIR)$(bindir)

uninstall-local:
	rm -f $(DESTDIR)$(pythondir)/xraylib.py $(DESTDIR)$(pythondir)/xrayhelp.py $(DESTDIR)$(pythondir)/xraymessages.py
	rm -f $(DESTDIR)$(pyexecdir)/_xraylib.*
	rm -f $(DESTDIR)$(bindir)/xraylib

clean-local:
	rm -rf xraylib_wrap.c build _xraylib.* xraylib.py





endif
