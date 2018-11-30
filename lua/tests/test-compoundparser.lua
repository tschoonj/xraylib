require("xraylib")
luaunit = require("luaunit")

TestCompoundParser = {}
	function TestCompoundParser:test_good_compounds()
		luaunit.assertIsTable(xraylib.CompoundParser("C19H29COOH"))
		luaunit.assertIsTable(xraylib.CompoundParser("C12H10"))
		luaunit.assertIsTable(xraylib.CompoundParser("C12H6O2"))
		luaunit.assertIsTable(xraylib.CompoundParser("C6H5Br"))
		luaunit.assertIsTable(xraylib.CompoundParser("C3H4OH(COOH)3"))
		luaunit.assertIsTable(xraylib.CompoundParser("HOCH2CH2OH"))
		luaunit.assertIsTable(xraylib.CompoundParser("C5H11NO2"))
		luaunit.assertIsTable(xraylib.CompoundParser("CH3CH(CH3)CH3"))
		luaunit.assertIsTable(xraylib.CompoundParser("NH2CH(C4H5N2)COOH"))
		luaunit.assertIsTable(xraylib.CompoundParser("H2O"))
		luaunit.assertIsTable(xraylib.CompoundParser("Ca5(PO4)3F"))
		luaunit.assertIsTable(xraylib.CompoundParser("Ca5(PO4)3OH"))
		luaunit.assertIsTable(xraylib.CompoundParser("Ca5.522(PO4.48)3OH"))
		luaunit.assertIsTable(xraylib.CompoundParser("Ca5.522(PO.448)3OH"))
	end

	function TestCompoundParser:test_bad_compounds()
		luaunit.assertError(xraylib.CompoundParser, "CuI2ww")
		luaunit.assertError(xraylib.CompoundParser, "0C")
		luaunit.assertError(xraylib.CompoundParser, "2O")
		luaunit.assertError(xraylib.CompoundParser, "13Li")
		luaunit.assertError(xraylib.CompoundParser, "2(NO3)")
		luaunit.assertError(xraylib.CompoundParser, "H(2)")
		luaunit.assertError(xraylib.CompoundParser, "Ba(12)")
		luaunit.assertError(xraylib.CompoundParser, "Cr(5)3")
		luaunit.assertError(xraylib.CompoundParser, "Pb(13)2")
		luaunit.assertError(xraylib.CompoundParser, "Au(22)11")
		luaunit.assertError(xraylib.CompoundParser, "Au11(H3PO4)2)")
		luaunit.assertError(xraylib.CompoundParser, "Au11(H3PO4))2")
		luaunit.assertError(xraylib.CompoundParser, "Au(11(H3PO4))2")
		luaunit.assertError(xraylib.CompoundParser, "Ca5.522(PO.44.8)3OH")
		luaunit.assertError(xraylib.CompoundParser, "Ba[12]")
		luaunit.assertError(xraylib.CompoundParser, "Auu1")
		luaunit.assertError(xraylib.CompoundParser, "AuL1")
		luaunit.assertError(xraylib.CompoundParser, nil)
		luaunit.assertError(xraylib.CompoundParser, "  ")
		luaunit.assertError(xraylib.CompoundParser, "\t")
		luaunit.assertError(xraylib.CompoundParser, "\n")
		luaunit.assertError(xraylib.CompoundParser, "Au L1")
		luaunit.assertError(xraylib.CompoundParser, "Au\tFe")
		luaunit.assertError(xraylib.CompoundParser, 26)
	end

	function TestCompoundParser:test_H2SO4()
		cd = xraylib.CompoundParser("H2SO4")
		luaunit.assertEquals(cd['nElements'], 3)
		luaunit.assertAlmostEquals(cd['molarMass'], 98.09)
		luaunit.assertAlmostEquals(cd['nAtomsAll'], 7.0)
		luaunit.assertEquals(cd['Elements'], {1, 8, 16})
		luaunit.assertAlmostEquals(cd['massFractions'][1], 0.02059333265368539, 1E-6)
		luaunit.assertAlmostEquals(cd['massFractions'][2], 0.6524620246712203, 1E-6)
		luaunit.assertAlmostEquals(cd['massFractions'][3], 0.32694464267509427, 1E-6)
		luaunit.assertEquals(cd['nAtoms'], {2.0, 4.0, 1.0})
	end

TestSymbolToAtomicNumber = {}
	function TestSymbolToAtomicNumber:test_Fe()
		luaunit.assertEquals(xraylib.SymbolToAtomicNumber('Fe'), 26)
	end

	function TestSymbolToAtomicNumber:test_bad_symbol()
		luaunit.assertError(xraylib.SymbolToAtomicNumber, 'Uu')
	end

	function TestSymbolToAtomicNumber:test_bad_type()
		luaunit.assertError(xraylib.SymbolToAtomicNumber, 26)
	end

TestAtomicNumberToSymbol = {}
	function TestAtomicNumberToSymbol:test_Fe()
		luaunit.assertEquals(xraylib.AtomicNumberToSymbol(26), 'Fe')
		luaunit.assertEquals(xraylib.AtomicNumberToSymbol("26"), 'Fe')
	end

	function TestAtomicNumberToSymbol:test_bad_symbol()
		luaunit.assertError(xraylib.AtomicNumberToSymbol, -2)
		luaunit.assertError(xraylib.AtomicNumberToSymbol, 108)
	end

	function TestAtomicNumberToSymbol:test_bad_type()
		luaunit.assertError(xraylib.AtomicNumberToSymbol, "Fe")
		luaunit.assertError(xraylib.AtomicNumberToSymbol, nil)
	end

TestCrossValidation = {}
	function TestCrossValidation:test()
		for Z = 1, 107 do
			symbol = xraylib.AtomicNumberToSymbol(Z)
			luaunit.assertEquals(xraylib.SymbolToAtomicNumber(symbol), Z)
		end
	end

os.exit(luaunit.LuaUnit.run('-v', '-o', 'tap'))
