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

public class TestCosKron {

	static Stream<Arguments> goodValuesProvider() {
		return Stream.of(
			arguments(92, Xraylib.FL13_TRANS, 0.620),
			arguments(75, Xraylib.FL12_TRANS, 1.03E-1),
			arguments(51, Xraylib.FL23_TRANS, 1.24E-1),
			arguments(86, Xraylib.FM45_TRANS, 6E-2),
			arguments(109, Xraylib.FM45_TRANS, 1.02E-1),
			arguments(92, Xraylib.FM45_TRANS, 0.088)
		);
	}

	@ParameterizedTest(name="test_good_values {index} -> {0} {1}")
	@MethodSource("goodValuesProvider")
	public void test_good_values(int Z, int trans, double expected) {
		double coskron = Xraylib.CosKronTransProb(Z, trans);
		assertEquals(coskron, expected, 1E-6);
	}
	
	static Stream<Arguments> badValuesProvider() {
		return Stream.of(
			arguments(11, Xraylib.FL12_TRANS, Xraylib.INVALID_CK),
			arguments(110, Xraylib.FL12_TRANS, Xraylib.INVALID_CK),
			arguments(0, Xraylib.FL12_TRANS, Xraylib.Z_OUT_OF_RANGE),
			arguments(0, Xraylib.FL12_TRANS, Xraylib.Z_OUT_OF_RANGE),
			arguments(Xraylib.ZMAX + 1, Xraylib.FL12_TRANS, Xraylib.Z_OUT_OF_RANGE),
			arguments(26, 0, Xraylib.UNKNOWN_CK),
			arguments(92, Xraylib.FM45_TRANS + 1, Xraylib.UNKNOWN_CK)
		);
	}

	@ParameterizedTest(name="test_bad_values {index} -> {0} {1}")
	@MethodSource("badValuesProvider")
	public void test_good_values(int Z, int trans, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double coskron = Xraylib.CosKronTransProb(Z, trans);
		}, message);
		assertEquals(exc.getMessage(), message);
	}
	

	
	@Test
	public void test_consistency_L1() {
		double sum = Xraylib.FluorYield(92, Xraylib.L1_SHELL) +
				Xraylib.AugerYield(92, Xraylib.L1_SHELL) +
				Xraylib.CosKronTransProb(92, Xraylib.FL12_TRANS) + 
				Xraylib.CosKronTransProb(92, Xraylib.FL13_TRANS) + 
				Xraylib.CosKronTransProb(92, Xraylib.FLP13_TRANS);
		assertEquals(sum, 1.0, 1E-6);
	}

	@Test
	public void test_consistency_L2() {
		double sum = Xraylib.FluorYield(92, Xraylib.L2_SHELL) +
				Xraylib.AugerYield(92, Xraylib.L2_SHELL) +
				Xraylib.CosKronTransProb(92, Xraylib.FL23_TRANS);
		assertEquals(sum, 1.0, 1E-6);
	}

	@Test
	public void test_consistency_M1() {
		double sum = Xraylib.FluorYield(92, Xraylib.M1_SHELL) +
				Xraylib.AugerYield(92, Xraylib.M1_SHELL) +
				Xraylib.CosKronTransProb(92, Xraylib.FM12_TRANS) +
				Xraylib.CosKronTransProb(92, Xraylib.FM13_TRANS) +
				Xraylib.CosKronTransProb(92, Xraylib.FM14_TRANS) +
				Xraylib.CosKronTransProb(92, Xraylib.FM15_TRANS);
		assertEquals(sum, 1.0, 1E-6);
	}

	@Test
	public void test_consistency_M2() {
		double sum = Xraylib.FluorYield(92, Xraylib.M2_SHELL) +
				Xraylib.AugerYield(92, Xraylib.M2_SHELL) +
				Xraylib.CosKronTransProb(92, Xraylib.FM23_TRANS) +
				Xraylib.CosKronTransProb(92, Xraylib.FM24_TRANS) +
				Xraylib.CosKronTransProb(92, Xraylib.FM25_TRANS);
		assertEquals(sum, 1.0, 1E-6);
	}

	@Test
	public void test_consistency_M3() {
		double sum = Xraylib.FluorYield(92, Xraylib.M3_SHELL) +
				Xraylib.AugerYield(92, Xraylib.M3_SHELL) +
				Xraylib.CosKronTransProb(92, Xraylib.FM34_TRANS) +
				Xraylib.CosKronTransProb(92, Xraylib.FM35_TRANS);
		assertEquals(sum, 1.0, 1E-6);
	}

	@Test
	public void test_consistency_M4() {
		double sum = Xraylib.FluorYield(92, Xraylib.M4_SHELL) +
				Xraylib.AugerYield(92, Xraylib.M4_SHELL) +
				Xraylib.CosKronTransProb(92, Xraylib.FM45_TRANS);
		assertEquals(sum, 1.0, 1E-6);
	}
}
