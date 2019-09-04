import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
 
import com.github.tschoonj.xraylib.Xraylib;
import com.github.tschoonj.xraylib.compoundData;

public class TestCompoundParser {

	@ParameterizedTest(name="test_good_compounds {index} -> {arguments}")
	@ValueSource(strings = {
		"C19H29COOH",
		"C12H10",
		"C12H6O2",
		"C6H5Br",
		"C3H4OH(COOH)3",
		"HOCH2CH2OH",
		"C5H11NO2",
		"CH3CH(CH3)CH3",
		"NH2CH(C4H5N2)COOH",
		"H2O",
		"Ca5(PO4)3F",
		"Ca5(PO4)3OH",
		"Ca5.522(PO4.48)3OH",
		"Ca5.522(PO.448)3OH"
	})
	public void test_good_compounds(String compound) {
		compoundData cd = Xraylib.CompoundParser(compound);	
	}

	@ParameterizedTest(name="test_bad_compounds {index} -> {arguments}")
	@NullAndEmptySource
	@ValueSource(strings = {
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
		"  ",
		"\t",
		"\n",
		"Au L1",
		"Au\tFe"
	})
	public void test_bad_compounds(String compound) {
		assertThrows(IllegalArgumentException.class, () -> {
			compoundData cd = Xraylib.CompoundParser(compound);
		});
	}

	@Test
	public void test_H2SO4() {
		compoundData cd = Xraylib.CompoundParser("H2SO4");
		assertEquals(cd.nElements, 3);
		assertEquals(cd.molarMass, 98.09);
		assertEquals(cd.nAtomsAll, 7.0);
		assertArrayEquals(cd.Elements, new int[]{1, 8, 16});
		assertArrayEquals(cd.massFractions, new double[]{0.02059333265368539, 0.6524620246712203, 0.32694464267509427}, 1E-8);
		assertArrayEquals(cd.nAtoms, new double[]{2.0, 4.0, 1.0});
	}

	@Test
	public void test_SymbolToAtomicNumber_Fe() {
		assertEquals(Xraylib.SymbolToAtomicNumber("Fe"), 26);
	}

	@Test
	public void test_SymbolToAtomicNumber_bad_symbol() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			int Z = Xraylib.SymbolToAtomicNumber("Uu");
		});
		assertEquals(exc.getMessage(), "Invalid chemical symbol");

		 assertThrows(IllegalArgumentException.class, () -> {
			int Z = Xraylib.SymbolToAtomicNumber(null);
		});
		assertEquals(exc.getMessage(), "Invalid chemical symbol");
	}

	@Test
	public void test_AtomicNumberToSymbol_Fe() {
		assertEquals(Xraylib.AtomicNumberToSymbol(26), "Fe");
	}

	@Test
	public void test_AtomicNumberToSymbol_bad_symbol() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			String symbol = Xraylib.AtomicNumberToSymbol(-2);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			String symbol = Xraylib.AtomicNumberToSymbol(108);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);
	}

	static IntStream generateStreamOfZ() {
		return IntStream.range(1, 108);
	}

	@ParameterizedTest(name="test_cross_validation {arguments}")
	@MethodSource("generateStreamOfZ")
	public void test_cross_validation(int Z) {
		String symbol = Xraylib.AtomicNumberToSymbol(Z);
		assertEquals(Xraylib.SymbolToAtomicNumber(symbol), Z);
	}
}
