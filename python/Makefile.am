#python bindings for xraylib

if ENABLE_PYTHON
all:
	make pythonxrl



pythonxrl: ${top_builddir}/src/xraylib.i
	$(SWIG) -o xraylib_wrap.c -python ${top_builddir}/src/xraylib.i
	$(PYTHON) xrlsetup.py build_ext --inplace




#I like to use my own commands for the install: although xrlsetup.py can handle this, I want to do it myself. Libtool would also be a possibility as is detailed in the automake manual.
#MKDIR_P's are necessary for distcheck
install-exec-local:
	$(MKDIR_P) $(DESTDIR)$(pythondir)
	$(MKDIR_P) $(DESTDIR)$(pyexecdir)
	$(INSTALL) xraylib.py $(DESTDIR)$(pythondir)
	$(INSTALL) _xraylib.* $(DESTDIR)$(pyexecdir)

uninstall-local:
	rm -f $(DESTDIR)$(pythondir)/xraylib.py
	rm -f $(DESTDIR)$(pyexecdir)/_xraylib.*

clean-local:
	rm -rf xraylib_wrap.c build





endif
