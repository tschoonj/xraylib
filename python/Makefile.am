#python bindings for xraylib

if ENABLE_PYTHON
all:
	make pythonxrl



pythonxrl: ${top_builddir}/src/xraylib.i
	$(SWIG) -o xraylib_wrap.c -python ${top_builddir}/src/xraylib.i
	$(PYTHON) xrlsetup.py build_ext --inplace












endif
