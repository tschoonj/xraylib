#Copyright (c) 2009, 2010, 2011, 2012, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if ENABLE_RUBY
all:
	$(MAKE) rubyxrl


#perl line is necessary to avoid C++ comments in wrapper... bug in SWIG
rubyxrl: ${top_srcdir}/src/xraylib.i
	$(SWIG) -I${top_srcdir}/include -includeall -o xraylib_wrap.c -ruby ${top_srcdir}/src/xraylib.i
	perl -pi -e 's!^\/\/.+$$!!' xraylib_wrap.c
	$(RUBY_EXT_CC) $(RUBY_EXT_CPPFLAGS) -DHAVE_SIGNBIT -DHAVE_STRUCT_TIMEVAL -DHAVE_STRUCT_TIMESPEC -I${top_srcdir}/include -I$(RUBY_EXT_INC) -c xraylib_wrap.c -o xraylib_wrap.o
	$(RUBY_EXT_CC) $(RUBY_EXT_LDFLAGS) -o xraylib$(RUBY_EXT_DLEXT) xraylib_wrap.o -L../src/.libs -lxrl

install-exec-local: xraylib$(RUBY_EXT_DLEXT)
	$(MKDIR_P) $(DESTDIR)$(RUBYDIR)
	$(INSTALL) xraylib$(RUBY_EXT_DLEXT) $(DESTDIR)$(RUBYDIR)/

uninstall-local:
	rm -f $(DESTDIR)$(RUBYDIR)/xraylib$(RUBY_EXT_DLEXT)
 


clean-local:
	rm -rf xraylib_wrap.* xraylib.bundle
endif
