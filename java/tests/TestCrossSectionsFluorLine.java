import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.Arguments;
import static org.junit.jupiter.params.provider.Arguments.arguments;
 
import com.github.tschoonj.xraylib.Xraylib;

public class TestCrossSectionsFluorLine {

	static Stream<Arguments> goodValuesProvider() {
		return Stream.of(
			arguments(29, Xraylib.L3M5_LINE, 10.0, 0.13198670698075143),
			arguments(29, Xraylib.L1M5_LINE, 10.0, 7.723944209880828e-06),
			arguments(29, Xraylib.L1M2_LINE, 10.0, 0.0018343168459245755),
			arguments(29, Xraylib.KL3_LINE, 10.0, 49.21901698835919),
			arguments(26, Xraylib.L2M4_LINE, 10.0, 0.0200667),
			arguments(26, Xraylib.L1M2_LINE, 10.0, 0.000830915),
			arguments(29, Xraylib.KL3_LINE, 999.0, 1.1827911388054846E-4)
		);
	}

	@ParameterizedTest(name="test_good_values {index} -> {0} {1} {2}")
	@MethodSource("goodValuesProvider")
	public void test_good_values(int Z, int line, double energy, double expected) {
		double cs = Xraylib.CS_FluorLine(Z, line, energy);
		assertEquals(cs, expected, 1E-6);
	}

	static Stream<Arguments> badValuesProvider() {
		return Stream.of(
			arguments(26, Xraylib.L3M5_LINE, 10.0, Xraylib.UNAVAILABLE_CK),
			arguments(0, Xraylib.KL3_LINE, 10.0, Xraylib.Z_OUT_OF_RANGE),
			arguments(Xraylib.ZMAX, Xraylib.KL3_LINE, 10.0, Xraylib.INVALID_LINE),
			arguments(Xraylib.ZMAX + 1, Xraylib.KL3_LINE, 10.0, Xraylib.Z_OUT_OF_RANGE),
			arguments(1, Xraylib.KL3_LINE, 10.0, Xraylib.INVALID_LINE),
			arguments(92, Xraylib.M5N7_LINE, 10.0, Xraylib.INVALID_LINE),
			arguments(26, Xraylib.KL3_LINE, 0.0, Xraylib.NEGATIVE_ENERGY),
			arguments(92, Xraylib.L3M5_LINE, 10.0, Xraylib.TOO_LOW_EXCITATION_ENERGY),
			arguments(26, Xraylib.KL3_LINE, 1001.0, Xraylib.SPLINT_X_TOO_HIGH)
		);
	}

	@ParameterizedTest(name="test_bad_values {index} -> {0} {1} {2}")
	@MethodSource("badValuesProvider")
	public void test_bad_values(int Z, int line, double energy, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = Xraylib.CS_FluorLine(Z, line, energy);
		}, message);
		assertEquals(exc.getMessage(), message);
	}

	@Test
	public void test_deprecated_siegbahn_macros() {

		// KA_LINE
		double cs = Xraylib.CS_FluorLine(26, Xraylib.KL3_LINE, 10.0);
		cs += Xraylib.CS_FluorLine(26, Xraylib.KL2_LINE, 10.0);
		assertEquals(cs, Xraylib.CS_FluorLine(26, Xraylib.KA_LINE, 10.0), 1E-6);

		// LA_LINE
		cs = Xraylib.CS_FluorLine(92, Xraylib.L3M5_LINE, 30.0);
		cs += Xraylib.CS_FluorLine(92, Xraylib.L3M4_LINE, 30.0);
		assertEquals(cs, Xraylib.CS_FluorLine(92, Xraylib.LA_LINE, 30.0), 1E-6);

		// LB_LINE
		final int[] lb_line_macros = new int[]{
			Xraylib.LB1_LINE, Xraylib.LB2_LINE, Xraylib.LB3_LINE, Xraylib.LB4_LINE,
			Xraylib.LB5_LINE, Xraylib.LB6_LINE, Xraylib.LB7_LINE, Xraylib.LB9_LINE,
			Xraylib.LB10_LINE, Xraylib.LB15_LINE, Xraylib.LB17_LINE, Xraylib.L3N6_LINE,
			Xraylib.L3N7_LINE
		};

		cs = 0.0;
		for (int line: lb_line_macros) {
			cs += Xraylib.CS_FluorLine(92, line, 30.0);
		}
		assertEquals(cs, Xraylib.CS_FluorLine(92, Xraylib.LB_LINE, 30.0), 1E-6);
	}
}
