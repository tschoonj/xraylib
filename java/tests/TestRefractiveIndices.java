import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

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
import org.apache.commons.numbers.complex.Complex;

public class TestRefractiveIndices {

	@FunctionalInterface
	private static interface RefractiveIndexWrapper {
		public Object execute(String compound, double energy, double density);
	}

	private static void myAssertEquals(Object actual, Object expected, double delta) {
		if (Double.class.isInstance(actual) && Double.class.isInstance(expected)) {
			assertEquals(Double.class.cast(actual), Double.class.cast(expected), delta);
		} else if (Complex.class.isInstance(actual) && Complex.class.isInstance(expected)) {
			Complex actualComplex = Complex.class.cast(actual);
			Complex expectedComplex = Complex.class.cast(expected);
			assertEquals(actualComplex.real(), expectedComplex.real(), delta);
			assertEquals(actualComplex.imag(), expectedComplex.imag(), delta);
		} else {
			fail("Unknown type for actual and/or expected");
		}
	}

	private static final String H2O = "H2O";
	private static final String AIR = "Air, Dry (near sea level)";

	static Stream<Arguments> goodValuesProvider() {
		return Stream.of(
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Re, H2O, 1.0, 1.0, 0.999763450676632),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Im, H2O, 1.0, 1.0, 4.021660592312145e-05),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index, H2O, 1.0, 1.0, Complex.ofCartesian(0.999763450676632, 4.021660592312145e-05)),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Re, AIR, 1.0, 1.0, 0.999782559048),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Im, AIR, 1.0, 1.0, 0.000035578193),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index, AIR, 1.0, 1.0, Complex.ofCartesian(0.999782559048, 0.000035578193)),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Re, AIR, 1.0, 0.0, 0.999999737984),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Im, AIR, 1.0, 0.0, 0.000000042872),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index, AIR, 1.0, 0.0, Complex.ofCartesian(0.999999737984, 0.000000042872)),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index, AIR, 1.0, -1.0, Complex.ofCartesian(0.999999737984, 0.000000042872))
		);
	}

	@ParameterizedTest(name="test_good_values {index} -> {0} {1}")
	@MethodSource("goodValuesProvider")
	public void test_good_values(RefractiveIndexWrapper wrapper, String compound, double energy, double density, Object expected) {
		Object rv = wrapper.execute(compound, energy, density);
		myAssertEquals(rv, expected, 1E-9);
	}

	static Stream<Arguments> badValuesProvider() {
		return Stream.of(
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Re, null, 1.0, 1.0, Xraylib.UNKNOWN_COMPOUND),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Im, null, 1.0, 1.0, Xraylib.UNKNOWN_COMPOUND),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index, null, 1.0, 1.0, Xraylib.UNKNOWN_COMPOUND),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Re, "", 1.0, 1.0, Xraylib.UNKNOWN_COMPOUND),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Im, "", 1.0, 1.0, Xraylib.UNKNOWN_COMPOUND),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index, "", 1.0, 1.0, Xraylib.UNKNOWN_COMPOUND),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Re, H2O, 0.0, 1.0, Xraylib.NEGATIVE_ENERGY),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Im, H2O, 0.0, 1.0, Xraylib.NEGATIVE_ENERGY),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index, H2O, 0.0, 1.0, Xraylib.NEGATIVE_ENERGY),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Re, H2O, 1.0, 0.0, Xraylib.NEGATIVE_DENSITY),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index_Im, H2O, 1.0, 0.0, Xraylib.NEGATIVE_DENSITY),
			arguments((RefractiveIndexWrapper) Xraylib::Refractive_Index, H2O, 1.0, 0.0, Xraylib.NEGATIVE_DENSITY)
		);
	}

	@ParameterizedTest(name="test_bad_values {index} -> {0} {1}")
	@MethodSource("badValuesProvider")
	public void test_bad_values(RefractiveIndexWrapper wrapper, String compound, double energy, double density, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			Object rv = wrapper.execute(compound, energy, density);
		});
		assertEquals(exc.getMessage(), message);
	}
}
