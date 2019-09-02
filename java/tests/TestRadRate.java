import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.Arguments;
import static org.junit.jupiter.params.provider.Arguments.arguments;
 
import com.github.tschoonj.xraylib.Xraylib;

public class TestRadRate {

	private static double RadRateNoThrow(int Z, int line) {
		try {
			return Xraylib.RadRate(Z, line);
		} catch (IllegalArgumentException e) {
			return 0.0;
		}
	}

	static Stream<Arguments> goodValuesProvider() {
		return Stream.of(
			arguments(26, Xraylib.KL3_LINE, 0.58357),
			arguments(92, Xraylib.KN2_LINE, 0.01452),
			arguments(56, Xraylib.L3M1_LINE, 0.031965),
			arguments(82, Xraylib.M5N7_LINE, 0.86638),
			arguments(109, Xraylib.KL3_LINE, 0.4698),
			arguments(26, Xraylib.KA_LINE, 0.88156),
			arguments(26, Xraylib.KA_LINE, RadRateNoThrow(26, Xraylib.KL1_LINE) + RadRateNoThrow(26, Xraylib.KL2_LINE) + RadRateNoThrow(26, Xraylib.KL3_LINE)),
			arguments(26, Xraylib.KB_LINE, 1.0 - Xraylib.RadRate(26, Xraylib.KA_LINE)),
			arguments(10, Xraylib.KA_LINE, 1.0),
			arguments(56, Xraylib.LA_LINE, 0.828176),
			arguments(56, Xraylib.LA_LINE, Xraylib.RadRate(56, Xraylib.L3M4_LINE) + Xraylib.RadRate(56, Xraylib.L3M5_LINE))
		);
	}

	@ParameterizedTest(name="test_good_values {index} -> {0} {1}")
	@MethodSource("goodValuesProvider")
	public void test_good_values(int Z, int line, double expected) {
		double rr = Xraylib.RadRate(Z, line);
		assertEquals(rr, expected, 1E-6);
	}

	static Stream<Arguments> badValuesProvider() {
		return Stream.of(
			arguments(0, Xraylib.KL3_LINE, Xraylib.Z_OUT_OF_RANGE),
			arguments(-1, Xraylib.KL3_LINE, Xraylib.Z_OUT_OF_RANGE),
			arguments(Xraylib.ZMAX + 1, Xraylib.KL3_LINE, Xraylib.Z_OUT_OF_RANGE),
			arguments(Xraylib.ZMAX, Xraylib.KL3_LINE, Xraylib.INVALID_LINE),
			arguments(110, Xraylib.KL3_LINE, Xraylib.INVALID_LINE),
			arguments(26, 1000, Xraylib.UNKNOWN_LINE),
			arguments(26, Xraylib.M5N7_LINE, Xraylib.INVALID_LINE),
			arguments(10, Xraylib.KB_LINE, Xraylib.INVALID_LINE),
			arguments(56, Xraylib.LB_LINE, Xraylib.INVALID_LINE)
		);
	}

	@ParameterizedTest(name="test_bad_values {index} -> {0} {1}")
	@MethodSource("badValuesProvider")
	public void test_bad_values(int Z, int line, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double rr = Xraylib.RadRate(Z, line);
		});
		assertEquals(exc.getMessage(), message);
	}

}
