import unittest
import xraylib

class TestCompoundParser(unittest.TestCase):
    def test_good_compounds(self):
        self.assertIsInstance(xraylib.CompoundParser("C19H29COOH"), dict)
        self.assertIsInstance(xraylib.CompoundParser("C12H10"), dict)
        self.assertIsInstance(xraylib.CompoundParser("C12H6O2"), dict)
        self.assertIsInstance(xraylib.CompoundParser("C6H5Br"), dict)
        self.assertIsInstance(xraylib.CompoundParser("C3H4OH(COOH)3"), dict)
        self.assertIsInstance(xraylib.CompoundParser("HOCH2CH2OH"), dict)
        self.assertIsInstance(xraylib.CompoundParser("C5H11NO2"), dict)
        self.assertIsInstance(xraylib.CompoundParser("CH3CH(CH3)CH3"), dict)
        self.assertIsInstance(xraylib.CompoundParser("NH2CH(C4H5N2)COOH"), dict)
        self.assertIsInstance(xraylib.CompoundParser("H2O"), dict)
        self.assertIsInstance(xraylib.CompoundParser("Ca5(PO4)3F"), dict)
        self.assertIsInstance(xraylib.CompoundParser("Ca5(PO4)3OH"), dict)
        self.assertIsInstance(xraylib.CompoundParser("Ca5.522(PO4.48)3OH"), dict)
        self.assertIsInstance(xraylib.CompoundParser("Ca5.522(PO.448)3OH"), dict)

    def test_bad_compounds(self):
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("CuI2ww")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("0C")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("2O")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("13Li")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("2(NO3)")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("H(2)")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("Ba(12)")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("Cr(5)3")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("Pb(13)2")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("Au(22)11")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("Au11(H3PO4)2)")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("Au11(H3PO4))2")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("Au(11(H3PO4))2")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("Ca5.522(PO.44.8)3OH")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("Ba[12]")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("Auu1")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("AuL1")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser(None)
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("  ")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("\t")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("\n")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("Au L1")
        with self.assertRaises(ValueError):
            xraylib.CompoundParser("Au\tFe")
        with self.assertRaises(TypeError):
            xraylib.CompoundParser(26)

class TestSymbolToAtomicNumber(unittest.TestCase):
    def test_Fe(self):
        self.assertEqual(xraylib.SymbolToAtomicNumber('Fe'), 26)

    def test_bad_symbol(self):
        with self.assertRaises(ValueError):
            xraylib.SymbolToAtomicNumber('Uu')

    def test_bad_type(self):
        with self.assertRaises(TypeError):
            xraylib.SymbolToAtomicNumber(26)

class TestAtomicNumberToSymbol(unittest.TestCase):
    def test_Fe(self):
        self.assertEqual(xraylib.AtomicNumberToSymbol(26), 'Fe')

    def test_bad_symbol(self):
        with self.assertRaises(ValueError):
            xraylib.AtomicNumberToSymbol(-2)
        with self.assertRaises(ValueError):
            xraylib.AtomicNumberToSymbol(108)

    def test_bad_type(self):
        with self.assertRaises(TypeError):
            xraylib.AtomicNumberToSymbol("26")
        with self.assertRaises(TypeError):
            xraylib.AtomicNumberToSymbol("Fe")
        with self.assertRaises(TypeError):
            xraylib.AtomicNumberToSymbol(None)

class TestCrossValidation(unittest.TestCase):
    def test(self):
        for Z in range(1, 108):
            symbol = xraylib.AtomicNumberToSymbol(Z)
            self.assertEqual(xraylib.SymbolToAtomicNumber(symbol), Z)

if __name__ == '__main__':
    unittest.main(verbosity=2)
