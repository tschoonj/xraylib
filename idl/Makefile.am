if ENABLE_IDL
#install the IDL bindings specific files
install-exec-local:
	$(MKDIR_P) $(DESTDIR)$(prefix)/pro	
	$(INSTALL) $(top_srcdir)/idl/xraylib_shells.pro $(top_srcdir)/idl/xraylib_help.pro $(top_srcdir)/idl/xraylib_lines.pro $(top_srcdir)/idl/xraylib.pro $(DESTDIR)$(prefix)/pro

uninstall-local:
	rm -f $(DESTDIR)$(prefix)/pro/xraylib_shells.pro $(DESTDIR)$(prefix)/pro/xraylib_help.pro  $(DESTDIR)$(prefix)/pro/xraylib_lines.pro $(DESTDIR)$(prefix)/pro/xraylib.pro
endif
