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

public class TestJumpFactor {

	static Stream<Arguments> goodValuesProvider() {
		return Stream.of(
			arguments(82, Xraylib.K_SHELL, 4.731),
			arguments(82, Xraylib.M3_SHELL, 1.15887)
		);
	}

	@ParameterizedTest(name="test_good_values {index} -> {0} {1}")
	@MethodSource("goodValuesProvider")
	public void test_good_values(int Z, int shell, double expected) {
		double jf = Xraylib.JumpFactor(Z, shell);
		assertEquals(jf, expected, 1E-6);
	}

	static Stream<Arguments> badValuesProvider() {
		return Stream.of(
			arguments(-35, Xraylib.K_SHELL, Xraylib.Z_OUT_OF_RANGE),
			arguments(82, -1, Xraylib.UNKNOWN_SHELL),
			arguments(26, Xraylib.N1_SHELL, Xraylib.INVALID_SHELL)
		);
	}

	@ParameterizedTest(name="test_bad_values {index} -> {0} {1}")
	@MethodSource("badValuesProvider")
	public void test_bad_values(int Z, int shell, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double jf = Xraylib.JumpFactor(Z, shell);
		});
		assertEquals(exc.getMessage(), message);
	}
}
