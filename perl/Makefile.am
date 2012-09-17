#Copyright (c) 2009, 2010, 2011, 2012, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


if ENABLE_PERL
all:
	$(MAKE) perlxrl


perlxrl: ${top_srcdir}/src/xraylib.i
	$(SWIG) -I${top_srcdir}/include -includeall -o xraylib_wrap.c -perl ${top_srcdir}/src/xraylib.i
	rm -rf Makefile2 blib pm_to_blib Makefile.PL
	@echo "#!/usr/bin/perl -w" > Makefile.PL
	@echo "use ExtUtils::MakeMaker 6.31;" >> Makefile.PL
	@echo "WriteMakefile(" >> Makefile.PL
	@echo "'NAME'=> 'xraylib'," >> Makefile.PL
	@echo "'OBJECT'=> 'xraylib_wrap.o'," >> Makefile.PL
	@echo "'FIRST_MAKEFILE'=> 'Makefile2'" >> Makefile.PL
	@echo ");" >> Makefile.PL
	$(PERL) Makefile.PL CCDLFLAGS="$(ARCHFLAGS)" LDDLFLAGS="$(ARCHFLAGS)" INC="-I${top_srcdir}/include" CCFLAGS="$(ARCHFLAGS)" INSTALL_BASE=$(DESTDIR)/${prefix} VERSION=$(VERSION) LIBS="-L../src/.libs -lxrl"
	$(MAKE) $(AM_MAKEFLAGS)	INSTALL_BASE=$(DESTDIR)/$(prefix) VERSION=$(VERSION) -f Makefile2


install-exec-local:
	$(MAKE) $(AM_MAKEFLAGS) -f Makefile2 install

uninstall-local:
#	perl uninstall doesn't work... it's unsafe according to them.
#	So we'll have to do it ourselves
#	let's use the .packlist file for this.
	find $(DESTDIR)$(prefix) -name '.packlist' | grep xraylib | xargs cat | xargs rm -vf
	find $(DESTDIR)$(prefix) -name '.packlist' | grep xraylib | xargs rm -fv
	find $(DESTDIR)$(prefix) -name 'perllocal.pod' | xargs rm -fv

clean-local:
	rm -rf blib Makefile.PL Makefile2 pm_to_blib xraylib.bs xraylib.pm xraylib_wrap.*

endif
