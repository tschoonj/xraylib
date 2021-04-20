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

public class TestKisselPE {

	static Stream<Arguments> goodCSFluorLineKisselValuesProvider() {
		return Stream.of(
			arguments(29, Xraylib.L3M5_LINE, 10.0, 1.677692151560),
			arguments(29, Xraylib.L1M5_LINE, 10.0, 0.000126120244),
			arguments(29, Xraylib.L1M2_LINE, 10.0, 0.029951600106),
			arguments(29, Xraylib.KL3_LINE, 10.0, 49.51768761506201),
			arguments(82, Xraylib.M5N7_LINE, 30.0, 0.538227139546),
			arguments(82, Xraylib.M5N7_LINE, 100.0, 0.102639909656483),
			arguments(26, Xraylib.KL3_LINE, 300.0, 5.151152717634017E-4)
		);
	}

	@ParameterizedTest(name="test_cs_fluorline_kissel_good_values {index} -> {0} {1} {2}")
	@MethodSource("goodCSFluorLineKisselValuesProvider")
	public void test_cs_fluorline_kissel_good_values(int Z, int line, double energy, double expected) {
		double cs = Xraylib.CS_FluorLine_Kissel(Z, line, energy);
		assertEquals(cs, expected, 1E-3);
	}

	static Stream<Arguments> badCSFluorLineKisselValuesProvider() {
		return Stream.of(
			arguments(0, Xraylib.KL3_LINE, 10.0, Xraylib.Z_OUT_OF_RANGE),
			arguments(Xraylib.ZMAX, Xraylib.KL3_LINE, 10.0, Xraylib.INVALID_LINE),
			arguments(Xraylib.ZMAX + 1, Xraylib.KL3_LINE, 10.0, Xraylib.Z_OUT_OF_RANGE),
			arguments(Xraylib.ZMAX + 1, Xraylib.KL3_LINE, 10.0, Xraylib.Z_OUT_OF_RANGE),
			arguments(1, Xraylib.KL3_LINE, 10.0, Xraylib.INVALID_LINE),
			arguments(92, Xraylib.N1O3_LINE, 10.0, Xraylib.INVALID_LINE),
			arguments(26, Xraylib.KL3_LINE, 0.0, Xraylib.NEGATIVE_ENERGY),
			arguments(92, Xraylib.L3M5_LINE, 10.0, Xraylib.TOO_LOW_EXCITATION_ENERGY),
			arguments(26, Xraylib.KL3_LINE, 301.0, Xraylib.SPLINT_X_TOO_HIGH)
		);
	}

	@ParameterizedTest(name="test_cs_fluorline_kissel_bad_values {index} -> {0} {1} {2}")
	@MethodSource("badCSFluorLineKisselValuesProvider")
	public void test_cs_fluorline_kissel_bad_values(int Z, int line, double energy, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = Xraylib.CS_FluorLine_Kissel(Z, line, energy);
		});
		assertEquals(exc.getMessage(), message);
	}

	@Test
	public void test_cs_fluorline_kissel_deprecated_siegbahn_macros() {
		// KA_LINE
		double cs = Xraylib.CS_FluorLine_Kissel(26, Xraylib.KL3_LINE, 10.0);
		cs += Xraylib.CS_FluorLine_Kissel(26, Xraylib.KL2_LINE, 10.0);
		assertEquals(cs, Xraylib.CS_FluorLine_Kissel(26, Xraylib.KA_LINE, 10.0), 1E-6);

		// LA_LINE
		cs = Xraylib.CS_FluorLine_Kissel(92, Xraylib.L3M5_LINE, 30.0);
		cs += Xraylib.CS_FluorLine_Kissel(92, Xraylib.L3M4_LINE, 30.0);
		assertEquals(cs, Xraylib.CS_FluorLine_Kissel(92, Xraylib.LA_LINE, 30.0), 1E-6);

		// LB_LINE
		final int[] lb_line_macros = new int[]{
    		Xraylib.LB1_LINE,
    		Xraylib.LB2_LINE,
    		Xraylib.LB3_LINE,
    		Xraylib.LB4_LINE,
    		Xraylib.LB5_LINE,
    		Xraylib.LB6_LINE,
    		Xraylib.LB7_LINE,
    		Xraylib.LB9_LINE,
    		Xraylib.LB10_LINE,
    		Xraylib.LB15_LINE,
    		Xraylib.LB17_LINE,
    		Xraylib.L3N6_LINE,
    		Xraylib.L3N7_LINE,
		};

		cs = 0.0;
		for (int line: lb_line_macros) {
			try {
				cs += Xraylib.CS_FluorLine_Kissel(92, line, 30.0);
			} catch (IllegalArgumentException e) {
				continue;
			}
		}
		assertEquals(cs, Xraylib.CS_FluorLine_Kissel(92, Xraylib.LB_LINE, 30.0), 1E-6);

	}

	@Test
	public void test_cs_photo_partial_good() {
		double cs = Xraylib.CS_Photo_Partial(26, Xraylib.K_SHELL, 20.0);
		assertEquals(cs, 22.40452459077649, 1E-6);
		cs = Xraylib.CS_Photo_Partial(26, Xraylib.K_SHELL, 300.0);
		assertEquals(cs, 0.0024892741933504824, 1E-6);
	}

	static Stream<Arguments> badCSPhotoPartialsValuesProvider() {
		return Stream.of(
			arguments(26, Xraylib.K_SHELL, 6.0, Xraylib.TOO_LOW_EXCITATION_ENERGY),
			arguments(26, Xraylib.N5_SHELL, 16.0, Xraylib.INVALID_SHELL),
			arguments(26, Xraylib.SHELLNUM_K, 16.0, Xraylib.UNKNOWN_SHELL),
			arguments(26, Xraylib.K_SHELL, 0.0, Xraylib.NEGATIVE_ENERGY),
			arguments(0, Xraylib.K_SHELL, 0.0, Xraylib.Z_OUT_OF_RANGE),
			arguments(26, Xraylib.K_SHELL, 301.0, Xraylib.SPLINT_X_TOO_HIGH)
		);
	}

	@ParameterizedTest(name="test_cs_photo_partial_bad_values {index} -> {0} {1} {2}")
	@MethodSource("badCSPhotoPartialsValuesProvider")
	public void test_cs_photo_partial_bad_values(int Z, int shell, double energy, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = Xraylib.CS_Photo_Partial(Z, shell, energy);
		});
		assertEquals(exc.getMessage(), message);
	}

	@Test
	public void test_electron_config_good() {
		double ec = Xraylib.ElectronConfig(26, Xraylib.M5_SHELL);
		assertEquals(ec, 3.6, 1E-6);
	}

	static Stream<Arguments> badElectronConfigValuesProvider() {
		return Stream.of(
			arguments(26, Xraylib.N7_SHELL, Xraylib.INVALID_SHELL),
			arguments(26, Xraylib.SHELLNUM_K, Xraylib.UNKNOWN_SHELL),
			arguments(0, Xraylib.K_SHELL, Xraylib.Z_OUT_OF_RANGE)
		);
	}

	@ParameterizedTest(name="test_electron_config_bad_values {index} -> {0} {1}")
	@MethodSource("badElectronConfigValuesProvider")
	public void test_electron_config_bad_values(int Z, int shell, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double ec = Xraylib.ElectronConfig(Z, shell);
		});
		assertEquals(exc.getMessage(), message);
	}
}
