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

public class TestScattering {

	@FunctionalInterface
	private static interface IntDoubleWrapper {
		public double execute(int arg1, double arg2);
	}

	static Stream<Arguments> goodIntDoubleValuesProvider() {
		return Stream.of(
			arguments((IntDoubleWrapper) Xraylib::FF_Rayl, 26, 0.0, 26.0),
			arguments((IntDoubleWrapper) Xraylib::FF_Rayl, 92, 10.0, 2.1746),
			arguments((IntDoubleWrapper) Xraylib::FF_Rayl, 98, 10.0, 2.4621),
			arguments((IntDoubleWrapper) Xraylib::FF_Rayl, 9, 1E9 - 1, 4.630430625170564E-21),
			arguments((IntDoubleWrapper) Xraylib::SF_Compt, 26, 0.1, 2.891),
			arguments((IntDoubleWrapper) Xraylib::SF_Compt, 92, 10.0, 89.097),
			arguments((IntDoubleWrapper) Xraylib::SF_Compt, 98, 10.0, 94.631),
			arguments((IntDoubleWrapper) Xraylib::SF_Compt, 9, 1E9 - 1, 9.0)
		);
	}

	static Stream<Arguments> badIntDoubleValuesProvider() {
		return Stream.of(
			arguments((IntDoubleWrapper) Xraylib::FF_Rayl, 0, 10.0, Xraylib.Z_OUT_OF_RANGE),
			arguments((IntDoubleWrapper) Xraylib::FF_Rayl, 99, 10.0, Xraylib.Z_OUT_OF_RANGE),
			arguments((IntDoubleWrapper) Xraylib::FF_Rayl, 9, -10.0, Xraylib.NEGATIVE_Q),
			arguments((IntDoubleWrapper) Xraylib::FF_Rayl, 9, 1E9 + 1, Xraylib.SPLINT_X_TOO_HIGH),
			arguments((IntDoubleWrapper) Xraylib::SF_Compt, 0, 10.0, Xraylib.Z_OUT_OF_RANGE),
			arguments((IntDoubleWrapper) Xraylib::SF_Compt, 99, 10.0, Xraylib.Z_OUT_OF_RANGE),
			arguments((IntDoubleWrapper) Xraylib::SF_Compt, 9, 0.0, Xraylib.NEGATIVE_Q),
			arguments((IntDoubleWrapper) Xraylib::SF_Compt, 9, 1E9 + 1, Xraylib.SPLINT_X_TOO_HIGH)
		);
	}

	@ParameterizedTest(name="test_good_int_double_values {index} -> {0} {1} {2}")
	@MethodSource("goodIntDoubleValuesProvider")
	public void test_good_int_double_values(IntDoubleWrapper wrapper, int arg1, double arg2, double expected) {
		double rv = wrapper.execute(arg1, arg2);
		assertEquals(rv, expected, 1E-6);
	}

	@ParameterizedTest(name="test_bad_int_double_values {index} -> {0} {1} {2}")
	@MethodSource("badIntDoubleValuesProvider")
	public void test_bad_int_double_values(IntDoubleWrapper wrapper, int arg1, double arg2, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			Object rv = wrapper.execute(arg1, arg2);
		});
		assertEquals(exc.getMessage(), message);
	}

	@FunctionalInterface
	private static interface DoubleWrapper {
		public double execute(double arg);
	}

	static Stream<Arguments> goodDoubleValuesProvider() {
		return Stream.of(
			arguments((DoubleWrapper) Xraylib::DCS_Thoms, Math.PI / 4.0, 0.05955590775),
			arguments((DoubleWrapper) Xraylib::CS_KN, 10, 0.6404703229290962)
		);
	}

	static Stream<Arguments> badDoubleValuesProvider() {
		return Stream.of(
			arguments((DoubleWrapper) Xraylib::CS_KN, 0.0, Xraylib.NEGATIVE_ENERGY),
			arguments((DoubleWrapper) Xraylib::CS_KN, -10.0, Xraylib.NEGATIVE_ENERGY)
		);
	}

	@ParameterizedTest(name="test_good_double_values {index} -> {0} {1}")
	@MethodSource("goodDoubleValuesProvider")
	public void test_good_double_values(DoubleWrapper wrapper, double arg, double expected) {
		double rv = wrapper.execute(arg);
		assertEquals(rv, expected, 1E-6);
	}

	@ParameterizedTest(name="test_bad_double_values {index} -> {0} {1}")
	@MethodSource("badDoubleValuesProvider")
	public void test_bad_double_values(DoubleWrapper wrapper, double arg, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			Object rv = wrapper.execute(arg);
		});
		assertEquals(exc.getMessage(), message);
	}

	@FunctionalInterface
	private static interface DoubleDoubleWrapper {
		public double execute(double arg1, double arg2);
	}

	static Stream<Arguments> goodDoubleDoubleValuesProvider() {
		return Stream.of(
			arguments((DoubleDoubleWrapper) Xraylib::DCS_KN, 10.0, Math.PI / 4.0, 0.058880292827846535),
			arguments((DoubleDoubleWrapper) Xraylib::MomentTransf, 10.0, Math.PI, 0.8065544290795198),
			arguments((DoubleDoubleWrapper) Xraylib::ComptonEnergy, 10.0, Math.PI / 4.0, 9.943008884806082),
			arguments((DoubleDoubleWrapper) Xraylib::ComptonEnergy, 10.0, 0.0, 10.0)
		);
	}

	static Stream<Arguments> badDoubleDoubleValuesProvider() {
		return Stream.of(
			arguments((DoubleDoubleWrapper) Xraylib::DCS_KN, 0.0, 0.0, Xraylib.NEGATIVE_ENERGY),
			arguments((DoubleDoubleWrapper) Xraylib::DCS_KN, -10.0, 0.0, Xraylib.NEGATIVE_ENERGY),
			arguments((DoubleDoubleWrapper) Xraylib::MomentTransf, 0.0, Math.PI, Xraylib.NEGATIVE_ENERGY),
			arguments((DoubleDoubleWrapper) Xraylib::MomentTransf, -1.0, Math.PI, Xraylib.NEGATIVE_ENERGY),
			arguments((DoubleDoubleWrapper) Xraylib::ComptonEnergy, 0.0, Math.PI / 4.0, Xraylib.NEGATIVE_ENERGY)
		);
	}

	@ParameterizedTest(name="test_good_double_double_values {index} -> {0} {1} {2}")
	@MethodSource("goodDoubleDoubleValuesProvider")
	public void test_good_double_double_values(DoubleDoubleWrapper wrapper, double arg1, double arg2, double expected) {
		double rv = wrapper.execute(arg1, arg2);
		assertEquals(rv, expected, 1E-6);
	}

	@ParameterizedTest(name="test_bad_double_double_values {index} -> {0} {1} {2}")
	@MethodSource("badDoubleDoubleValuesProvider")
	public void test_bad_double_double_values(DoubleDoubleWrapper wrapper, double arg1, double arg2, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			Object rv = wrapper.execute(arg1, arg2);
		});
		assertEquals(exc.getMessage(), message);
	}

	@FunctionalInterface
	private static interface IntDoubleDoubleWrapper {
		public double execute(int arg1, double arg2, double arg3);
	}

	static Stream<Arguments> goodIntDoubleDoubleValuesProvider() {
		return Stream.of(
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Rayl, 26, 10.0, Math.PI / 4.0, 0.17394690792051704),
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Rayl, 26, 10.0, 0.0, 0.5788126901827545),
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Rayl, 98, 10.0, Math.PI / 4.0, 0.7582632962268532),
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Compt, 26, 10.0, Math.PI / 4.0, 0.005489497545806117),
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Compt, 92, 1.0, Math.PI / 3.0, 0.0002205553556181471),
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Compt, 98, 10.0, Math.PI / 4.0, 0.0026360563424557386)
		);
	}

	static Stream<Arguments> badIntDoubleDoubleValuesProvider() {
		return Stream.of(
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Rayl, 0, 10.0, Math.PI / 4.0, Xraylib.Z_OUT_OF_RANGE),
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Rayl, 99, 10.0, Math.PI / 4.0, Xraylib.Z_OUT_OF_RANGE),
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Rayl, 26, 0.0, Math.PI / 4.0, Xraylib.NEGATIVE_ENERGY),
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Compt, 0, 10.0, Math.PI / 4.0, Xraylib.Z_OUT_OF_RANGE),
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Compt, 99, 10.0, Math.PI / 4.0, Xraylib.Z_OUT_OF_RANGE),
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Compt, 26, 0.0, Math.PI / 4.0, Xraylib.NEGATIVE_ENERGY),
			arguments((IntDoubleDoubleWrapper) Xraylib::DCS_Compt, 26, 10.0, 0.0, Xraylib.NEGATIVE_Q)
		);
	}

	@ParameterizedTest(name="test_good_int_double_double_values {index} -> {0} {1} {2} {3}")
	@MethodSource("goodIntDoubleDoubleValuesProvider")
	public void test_good_int_double_double_values(IntDoubleDoubleWrapper wrapper, int arg0, double arg1, double arg2, double expected) {
		double rv = wrapper.execute(arg0, arg1, arg2);
		assertEquals(rv, expected, 1E-6);
	}

	@ParameterizedTest(name="test_bad_int_double_double_values {index} -> {0} {1} {2} {3}")
	@MethodSource("badIntDoubleDoubleValuesProvider")
	public void test_bad_int_double_double_values(IntDoubleDoubleWrapper wrapper, int arg0, double arg1, double arg2, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			Object rv = wrapper.execute(arg0, arg1, arg2);
		});
		assertEquals(exc.getMessage(), message);
	}
}
