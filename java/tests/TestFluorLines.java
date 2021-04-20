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

public class TestFluorLines {

	static Stream<Arguments> goodValuesProvider() {
		return Stream.of(
			arguments(26, Xraylib.KL3_LINE, 6.4039),
			arguments(92, Xraylib.KL1_LINE, 93.844),
			arguments(56, Xraylib.L3M1_LINE, 3.9542),
			arguments(82, Xraylib.M5N7_LINE, 2.3477),
			arguments(104, Xraylib.KL3_LINE, 133.381),
			arguments(26, Xraylib.KA_LINE, 6.399505664957576),
			arguments(26, Xraylib.KB_LINE, 7.058),
			arguments(26, Xraylib.LA_LINE, 0.7045),
			arguments(26, Xraylib.LB_LINE, 0.724378),
			arguments(92, Xraylib.L1N67_LINE, (Xraylib.LineEnergy(92, Xraylib.L1N6_LINE) + Xraylib.LineEnergy(92, Xraylib.L1N7_LINE)) / 2.0),
			arguments(13, Xraylib.LB_LINE, 0.112131),
			arguments(21, Xraylib.LA_LINE, 0.3956),
			arguments(48, Xraylib.KO_LINE, Xraylib.LineEnergy(48, Xraylib.KO1_LINE)),
			arguments(82, Xraylib.KP_LINE, Xraylib.LineEnergy(82, Xraylib.KP1_LINE))
		);
	}

	@ParameterizedTest(name="test_good_values {index} -> {0} {1}")
	@MethodSource("goodValuesProvider")
	public void test_good_values(int Z, int line, double expected) {
		double energy = Xraylib.LineEnergy(Z, line);
		assertEquals(energy, expected, 1E-6);
	}

	static Stream<Arguments> badValuesProvider() {
		return Stream.of(
			arguments(0, Xraylib.KL3_LINE, Xraylib.Z_OUT_OF_RANGE),
			arguments(-1, Xraylib.KL3_LINE, Xraylib.Z_OUT_OF_RANGE),
			arguments(Xraylib.ZMAX + 1, Xraylib.KL3_LINE, Xraylib.Z_OUT_OF_RANGE),
			arguments(Xraylib.ZMAX, Xraylib.KL3_LINE, Xraylib.INVALID_LINE),
			arguments(105, Xraylib.KL3_LINE, Xraylib.INVALID_LINE),
			arguments(26, 1000, Xraylib.UNKNOWN_LINE),
			arguments(26, Xraylib.M5N7_LINE, Xraylib.INVALID_LINE),
			arguments(1, Xraylib.KA_LINE, Xraylib.INVALID_LINE),
			arguments(0, Xraylib.KA_LINE, Xraylib.Z_OUT_OF_RANGE),
			arguments(12, Xraylib.LB_LINE, Xraylib.INVALID_LINE),
			arguments(20, Xraylib.LA_LINE, Xraylib.INVALID_LINE),
			arguments(47, Xraylib.KO_LINE, Xraylib.INVALID_LINE),
			arguments(81, Xraylib.KP_LINE, Xraylib.INVALID_LINE)
		);
	}

	@ParameterizedTest(name="test_bad_values {index} -> {0} {1}")
	@MethodSource("badValuesProvider")
	public void test_bad_values(int Z, int line, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double energy = Xraylib.LineEnergy(Z, line);
		});
		assertEquals(exc.getMessage(), message);
	}
}
