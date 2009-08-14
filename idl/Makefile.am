if ENABLE_IDL
#install the IDL bindings specific files
install-exec-local:
	$(MKDIR_P) $(prefix)/pro	
	$(INSTALL) $(top_srcdir)/idl/xraylib_shells.pro $(top_srcdir)/idl/xraylib_help.pro $(top_srcdir)/idl/xraylib_lines.pro $(top_srcdir)/idl/xraylib.pro $(prefix)/pro

uninstall-local:
	rm -f $(prefix)/pro/xraylib_shells.pro $(prefix)/pro/xraylib_help.pro $(prefix)/pro/xraylib_lines.pro $(prefix)/pro/xraylib.pro
endif
