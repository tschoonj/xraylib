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

public class TestFii {

	static Stream<Arguments> goodValuesProvider() {
		return Stream.of(
			arguments(26, 1.0, -12.11016272606279),
			arguments(92, 10.0, -9.812520269504303),
			arguments(56, 100.0, -0.6440539604345072),
			arguments(100, 10.0, -13.413095197631547),
			arguments(59, 0.0011, 0.0),
			arguments(59, 9999, -0.0016150230148230047)
		);
	}

	@ParameterizedTest(name="test_good_values {index} -> {0} {1}")
	@MethodSource("goodValuesProvider")
	public void test_good_values(int Z, double energy, double expected) {
		double fii = Xraylib.Fii(Z, energy);
		assertEquals(fii, expected, 1E-6);
	}

	static Stream<Arguments> badValuesProvider() {
		return Stream.of(
			arguments(0, 10.0, Xraylib.Z_OUT_OF_RANGE),
			arguments(101, 10.0, Xraylib.Z_OUT_OF_RANGE),
			arguments(59, 0.0, Xraylib.NEGATIVE_ENERGY),
			arguments(59, -10.0, Xraylib.NEGATIVE_ENERGY),
			arguments(59, 0.0009, Xraylib.SPLINT_X_TOO_LOW),
			arguments(59, 10001, Xraylib.SPLINT_X_TOO_HIGH)
		);
	}

	@ParameterizedTest(name="test_bad_values {index} -> {0} {1}")
	@MethodSource("badValuesProvider")
	public void test_bad_values(int Z, double energy, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double fii = Xraylib.Fii(Z, energy);
		});
		assertEquals(exc.getMessage(), message);
	}
}
