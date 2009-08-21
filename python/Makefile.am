#python bindings for xraylib



EXTRA_DIST = xraylib


if ENABLE_PYTHON
all:
	$(MAKE) pythonxrl



pythonxrl: ${top_builddir}/src/xraylib.i
	$(SWIG) -o xraylib_wrap.c -python ${top_builddir}/src/xraylib.i
	$(PYTHON) xrlsetup.py build_ext --inplace




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
	rm -rf xraylib_wrap.c build xraylib _xraylib.* xraylib.py





endif
