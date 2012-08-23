#Copyright (c) 2009, 2010, 2011, 2012, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if ENABLE_LUA
all:
	$(MAKE) luaxrl



luaxrl: ${top_srcdir}/src/xraylib.i
	$(SWIG) -I${top_srcdir}/include -includeall -o xraylib_wrap.c -lua ${top_srcdir}/src/xraylib.i
	$(CC) -I${top_srcdir}/include -Wall $(LUA_INCLUDE) -c xraylib_wrap.c -o xraylib_wrap.o
	$(CC) -Wall $(LUA_LD_FLAGS) -o xraylib.so xraylib_wrap.o -L../src/.libs -lxrl $(LUA_LIB) 


install-exec-local: xraylib.so
	$(MKDIR_P) $(DESTDIR)$(libdir)/lua/xraylib
	$(INSTALL) xraylib.so $(DESTDIR)$(libdir)/lua/xraylib

uninstall-local:
	rm -f $(DESTDIR)$(libdir)/lua/xraylib/xraylib.so

clean-local:
	rm -f xraylib.so xraylib_wrap.c xraylib_wrap.o
endif

