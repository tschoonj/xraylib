require "xraylib"
require "test/unit"

class TestCompoundParser < Test::Unit::TestCase
	def test_good_compounds
		assert_instance_of(Hash, Xraylib.CompoundParser("C19H29COOH"))
		assert_instance_of(Hash, Xraylib.CompoundParser("C12H10"))
		assert_instance_of(Hash, Xraylib.CompoundParser("C12H6O2"))
		assert_instance_of(Hash, Xraylib.CompoundParser("C6H5Br"))
		assert_instance_of(Hash, Xraylib.CompoundParser("C3H4OH(COOH)3"))
		assert_instance_of(Hash, Xraylib.CompoundParser("HOCH2CH2OH"))
		assert_instance_of(Hash, Xraylib.CompoundParser("C5H11NO2"))
		assert_instance_of(Hash, Xraylib.CompoundParser("CH3CH(CH3)CH3"))
		assert_instance_of(Hash, Xraylib.CompoundParser("NH2CH(C4H5N2)COOH"))
		assert_instance_of(Hash, Xraylib.CompoundParser("H2O"))
		assert_instance_of(Hash, Xraylib.CompoundParser("Ca5(PO4)3F"))
		assert_instance_of(Hash, Xraylib.CompoundParser("Ca5(PO4)3OH"))
		assert_instance_of(Hash, Xraylib.CompoundParser("Ca5.522(PO4.48)3OH"))
		assert_instance_of(Hash, Xraylib.CompoundParser("Ca5.522(PO.448)3OH"))
	end

	def test_bad_compounds
		bad_compounds = [
			"CuI2ww",
			"0C",
			"2O",
			"13Li",
			"2(NO3)",
			"H(2)",
			"Ba(12)",
			"Cr(5)3",
			"Pb(13)2",
			"Au(22)11",
			"Au11(H3PO4)2)",
			"Au11(H3PO4))2",
			"Au(11(H3PO4))2",
			"Ca5.522(PO.44.8)3OH",
			"Ba[12]",
			"Auu1",
			"AuL1",
			nil,
			"  ",
			"\t",
			"\n",
			"Au L1",
			"Au\tFe"
		]
		bad_compounds.each {|compound| assert_raise(ArgumentError) {Xraylib.CompoundParser(compound)}}
		assert_raise(TypeError) {
			Xraylib.CompoundParser(26)
		}
		assert_raise(ArgumentError) {
			Xraylib.CompoundParser()
		}
		assert_raise(ArgumentError) {
			Xraylib.CompoundParser("H2O", "H2SO4")
		}
	end

	def test_H2SO4
		cd = Xraylib.CompoundParser("H2SO4")
		assert_equal(cd['nElements'], 3)
		assert_in_delta(cd['molarMass'], 98.09, 1E-6)
		assert_in_delta(cd['nAtomsAll'], 7.0, 1E-6)
		assert_equal(cd['Elements'], [1, 8, 16])
		[0.02059333265368539, 0.6524620246712203, 0.32694464267509427].each_with_index do |v, i|
			assert_in_delta(cd['massFractions'][i], v, 1E-6)
		end
		[2.0, 4.0, 1.0].each_with_index do |v, i|
			assert_in_delta(cd['nAtoms'][i], v, 1E-6)
		end
	end
end

class TestSymbolToAtomicNumber < Test::Unit::TestCase
	def test_Fe
		assert_equal(Xraylib.SymbolToAtomicNumber("Fe"), 26)
	end

	def test_bad_symbol
		assert_raise(ArgumentError) {
			Xraylib.SymbolToAtomicNumber("Uu")
		}
		assert_raise(TypeError) {
			Xraylib.SymbolToAtomicNumber(26)
		}
		assert_raise(ArgumentError) {
			Xraylib.SymbolToAtomicNumber(nil)
		}
		assert_raise(ArgumentError) {
			Xraylib.SymbolToAtomicNumber()
		}
		assert_raise(ArgumentError) {
			Xraylib.SymbolToAtomicNumber("Cl", "Fe")
		}
	end
end

class TestAtomicNumberToSymbol < Test::Unit::TestCase
	def test_Fe
		assert_equal(Xraylib.AtomicNumberToSymbol(26), "Fe")
	end

	def test_bad_symbol
		assert_raise(ArgumentError) {
			Xraylib.AtomicNumberToSymbol(-2)
		}
		assert_raise(ArgumentError) {
			Xraylib.AtomicNumberToSymbol(108)
		}
		assert_raise(TypeError) {
			Xraylib.AtomicNumberToSymbol("Fe")
		}
		assert_raise(TypeError) {
			Xraylib.AtomicNumberToSymbol(nil)
		}
		assert_raise(ArgumentError) {
			Xraylib.AtomicNumberToSymbol(26, 52)
		}
	end
end

class TestCrossValidation < Test::Unit::TestCase
	def test
		(1..107).each {|z| assert_equal(Xraylib.SymbolToAtomicNumber(Xraylib.AtomicNumberToSymbol(z)), z)}
	end
end
