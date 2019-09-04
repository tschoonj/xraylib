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

public class TestEdges {

	static Stream<Arguments> goodValuesProvider() {
		return Stream.of(
			arguments(26, Xraylib.K_SHELL, 7.112),
			arguments(1, Xraylib.K_SHELL, 0.0136),
			arguments(92, Xraylib.K_SHELL, 115.602),
			arguments(92, Xraylib.N7_SHELL, 0.379),
			arguments(92, Xraylib.P5_SHELL, 0.0057)
		);
	}

	@ParameterizedTest(name="test_good_values {index} -> {0} {1}")
	@MethodSource("goodValuesProvider")
	public void test_good_values(int Z, int shell, double expected) {
		double edge = Xraylib.EdgeEnergy(Z, shell);
		assertEquals(edge, expected, 1E-6);
	}

	static Stream<Arguments> badValuesProvider() {
		return Stream.of(
			arguments(0, Xraylib.K_SHELL, Xraylib.Z_OUT_OF_RANGE),
			arguments(Xraylib.ZMAX + 1, Xraylib.K_SHELL, Xraylib.Z_OUT_OF_RANGE),
			arguments(26, -1, Xraylib.UNKNOWN_SHELL),
			arguments(26, Xraylib.SHELLNUM, Xraylib.UNKNOWN_SHELL),
			arguments(26, Xraylib.P5_SHELL, Xraylib.INVALID_SHELL)
		);
	}

	@ParameterizedTest(name="test_bad_values {index} -> {0} {1}")
	@MethodSource("badValuesProvider")
	public void test_bad_values(int Z, int shell, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double edge = Xraylib.EdgeEnergy(Z, shell);
		});
		assertEquals(exc.getMessage(), message);
	}
}
