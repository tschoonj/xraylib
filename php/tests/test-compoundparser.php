<? 
include("xrltest.php");
include("xraylib.php");

function test_bad_compound($compound) {
	try {
		$cd = CompoundParser($compound);
	} catch (Exception $exception) {
		assertEqual($exception->getCode(), ValueError);	
		return;
	}
	throw new Exception("CompoundParser should have failed!");
}

const bad_compounds = array(
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
       NULL,
       "  ",
       "\t",
       "\n",
       "Au L1",
       "Au\tFe",
       26
      	);

class TestCompoundParser extends XrlTest {
	function test_good_compounds() {
		assertTrue(is_array(CompoundParser("C19H29COOH")));
		assertTrue(is_array(CompoundParser("C12H10")));
		assertTrue(is_array(CompoundParser("C12H6O2")));
		assertTrue(is_array(CompoundParser("C6H5Br")));
		assertTrue(is_array(CompoundParser("C3H4OH(COOH)3")));
		assertTrue(is_array(CompoundParser("HOCH2CH2OH")));
		assertTrue(is_array(CompoundParser("C5H11NO2")));
		assertTrue(is_array(CompoundParser("CH3CH(CH3)CH3")));
		assertTrue(is_array(CompoundParser("NH2CH(C4H5N2)COOH")));
		assertTrue(is_array(CompoundParser("H2O")));
		assertTrue(is_array(CompoundParser("Ca5(PO4)3F")));
		assertTrue(is_array(CompoundParser("Ca5(PO4)3OH")));
		assertTrue(is_array(CompoundParser("Ca5.522(PO4.48)3OH")));
		assertTrue(is_array(CompoundParser("Ca5.522(PO.448)3OH")));
	}
	function test_bad_compounds() {
		foreach (bad_compounds as $compound) {
			test_bad_compound($compound);
		}
	}
	function test_H2SO4() {
		$cd = CompoundParser("H2SO4");
		assertEqual($cd['nElements'], 3);
		assertAlmostEqual($cd['molarMass'], 98.09);
		assertAlmostEqual($cd['nAtomsAll'], 7.0);
		assertEqual($cd['Elements'], array(1, 8, 16));
		assertAlmostEqual($cd['massFractions'], array(0.02059333265368539, 0.6524620246712203, 0.32694464267509427));
		assertEqual($cd['nAtoms'], array(2.0, 4.0, 1.0));
	}
}

class TestSymbolToAtomicNumber extends XrlTest {
	function test_Fe() {
		assertEqual(SymbolToAtomicNumber("Fe"), 26);
	}
	function test_bad_symbol() {
		assertException(ValueError, "SymbolToAtomicNumber", "Uu");
		assertException(ValueError, "SymbolToAtomicNumber", 26);
		assertException(ValueError, "SymbolToAtomicNumber", NULL);
	}
}

class TestAtomicNumberToSymbol extends XrlTest {
	function test_Fe() {
		assertEqual(AtomicNumberToSymbol(26), "Fe");
	}
	function test_bad_symbol() {
		assertException(ValueError, "AtomicNumberToSymbol", -2);
		assertException(ValueError, "AtomicNumberToSymbol", 108);
		assertException(ValueError, "AtomicNumberToSymbol", "Fe");
		assertException(ValueError, "AtomicNumberToSymbol", NULL);
	}
}

class TestCrossValidation extends XrlTest {
	function test() {
		foreach (range(1, 107) as $Z) {
			assertEqual(SymbolToAtomicNumber(AtomicNumberToSymbol($Z)), $Z);
		}
	}
}

$suite = new XrlTestSuite();
$suite->append(new TestCompoundParser());
$suite->append(new TestSymbolToAtomicNumber());
$suite->append(new TestAtomicNumberToSymbol());
$suite->append(new TestCrossValidation());
$suite->run();

?>


