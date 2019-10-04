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

public class TestFi {

	static Stream<Arguments> goodValuesProvider() {
		return Stream.of(
			arguments(26, 1.0, -3.6433341979174823),
			arguments(92, 10.0, -4.152309997030393),
			arguments(56, 100.0, -0.05092880640048588),
			arguments(100, 10.0, -4.657346364215495),
			arguments(59, 0.0011, -59.22135650916104),
			arguments(59, 9999, -0.45526897883478495)
		);
	}

	@ParameterizedTest(name="test_good_values {index} -> {0} {1}")
	@MethodSource("goodValuesProvider")
	public void test_good_values(int Z, double energy, double expected) {
		double fi = Xraylib.Fi(Z, energy);
		assertEquals(fi, expected, 1E-6);
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
			double fi = Xraylib.Fi(Z, energy);
		}, message);
		assertEquals(exc.getMessage(), message);
	}
}
