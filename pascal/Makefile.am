#Copyright (c) 2015, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


if ENABLE_PASCAL

all-local: xraylib.ppu

xraylib.ppu: xraylib.pas
	$(FPC) -FE$(builddir) -Mdelphi $(srcdir)/xraylib.pas


install-exec-local: xraylib.ppu
	$(MKDIR_P) $(DESTDIR)$(FPC_DIR)
	$(INSTALL) xraylib.o xraylib.ppu $(DESTDIR)$(FPC_DIR)

uninstall-local:
	rm -rf $(DESTDIR)$(FPC_DIR)/xraylib.o
	rm -rf $(DESTDIR)$(FPC_DIR)/xraylib.ppu

clean-local:
	rm -rf xraylib.ppu xraylib.o libimpxraylib.a

SUBDIRS = tests
endif

EXTRA_DIST = xraylib.pas xraylib_impl.pas xraylib_iface.pas xraylib_const.pas meson.build
