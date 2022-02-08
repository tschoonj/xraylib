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
		/* see https://github.com/tschoonj/xraylib/issues/187 */
		cs = Xraylib.CSb_Photo_Partial(47, Xraylib.L2_SHELL, 3.5282);
		assertEquals(cs, 1.569549E+04, 1E-2);
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
	
	static int test_lines[] ={
		Xraylib.KL3_LINE,
		Xraylib.L1M3_LINE,
		Xraylib.L2M4_LINE,
		Xraylib.L3M5_LINE,
		Xraylib.M1N3_LINE,
		Xraylib.M2N4_LINE,
		Xraylib.M3N5_LINE,
		Xraylib.M4N6_LINE,
		Xraylib.M5N7_LINE,
	};

	@FunctionalInterface
	private interface CS_FluorLine_base {
		public abstract double execute(int Z, int line, double energy);
	}

	@FunctionalInterface
	private interface CS_FluorShell_base {
		public abstract double execute(int Z, int shell, double energy);
	}

	static Stream<Arguments> goodCSFluorLineKisselAllValuesProvider() {
		return Stream.of(
			arguments((CS_FluorShell_base) Xraylib::CS_FluorShell_Kissel, (CS_FluorLine_base) Xraylib::CS_FluorLine_Kissel, new double[]{1.488296, 0.021101, 0.431313, 0.701276, 0.000200, 0.004753, 0.004467, 0.099232, 0.134301}),
			arguments((CS_FluorShell_base) Xraylib::CSb_FluorShell_Kissel, (CS_FluorLine_base) Xraylib::CSb_FluorLine_Kissel, new double[]{588.359681, 8.341700, 170.508401, 277.231590, 0.078971, 1.878953, 1.766078, 39.228801, 53.092598}),
			arguments((CS_FluorShell_base) Xraylib::CS_FluorShell_Kissel_Cascade, (CS_FluorLine_base) Xraylib::CS_FluorLine_Kissel_Cascade, new double[]{1.488296, 0.021101, 0.431313, 0.701276, 0.000200, 0.004753, 0.004467, 0.099232, 0.134301}),
			arguments((CS_FluorShell_base) Xraylib::CS_FluorShell_Kissel_Radiative_Cascade, (CS_FluorLine_base) Xraylib::CS_FluorLine_Kissel_Radiative_Cascade, new double[]{1.488296, 0.017568, 0.413908, 0.671135, 0.000092, 0.001906, 0.001758, 0.043009, 0.055921}),
			arguments((CS_FluorShell_base) Xraylib::CS_FluorShell_Kissel_Nonradiative_Cascade, (CS_FluorLine_base) Xraylib::CS_FluorLine_Kissel_Nonradiative_Cascade, new double[]{1.488296, 0.021101, 0.100474, 0.169412, 0.000104, 0.001204, 0.001106, 0.018358, 0.025685}),
			arguments((CS_FluorShell_base) Xraylib::CS_FluorShell_Kissel_no_Cascade, (CS_FluorLine_base) Xraylib::CS_FluorLine_Kissel_no_Cascade, new double[]{1.488296, 0.017568, 0.083069, 0.139271, 0.000053, 0.000417, 0.000327, 0.003360, 0.004457}),
			arguments((CS_FluorShell_base) Xraylib::CSb_FluorShell_Kissel_Cascade, (CS_FluorLine_base) Xraylib::CSb_FluorLine_Kissel_Cascade, new double[]{588.359681, 8.341700, 170.508401, 277.231590, 0.078971, 1.878953, 1.766078, 39.228801, 53.092598}),
			arguments((CS_FluorShell_base) Xraylib::CSb_FluorShell_Kissel_Radiative_Cascade, (CS_FluorLine_base) Xraylib::CSb_FluorLine_Kissel_Radiative_Cascade, new double[]{588.359681, 6.945250, 163.627802, 265.316234, 0.036434, 0.753313, 0.695153, 17.002549, 22.106758}),
			arguments((CS_FluorShell_base) Xraylib::CSb_FluorShell_Kissel_Nonradiative_Cascade, (CS_FluorLine_base) Xraylib::CSb_FluorLine_Kissel_Nonradiative_Cascade, new double[]{588.359681, 8.341700, 39.719983, 66.972714, 0.041157, 0.475948, 0.437288, 7.257269, 10.153862}),
			arguments((CS_FluorShell_base) Xraylib::CSb_FluorShell_Kissel_no_Cascade, (CS_FluorLine_base) Xraylib::CSb_FluorLine_Kissel_no_Cascade, new double[]{588.359681, 6.945250, 32.839384, 55.057358, 0.020941, 0.164658, 0.129253, 1.328221, 1.762099})
		);
	}

	private static class LineMapping {

	    public final int line_lower;
	    public final int line_upper;
	    public final int shell;

	    public LineMapping(final int line_lower, final int line_upper, final int shell) {
	      this.line_lower = line_lower;
	      this.line_upper = line_upper;
	      this.shell = shell;
	    }
	}

	private static final LineMapping[] line_mappings = new LineMapping[]{
	    new LineMapping(Xraylib.KN5_LINE, Xraylib.KB_LINE, Xraylib.K_SHELL),
	    new LineMapping(Xraylib.L1P5_LINE, Xraylib.L1L2_LINE, Xraylib.L1_SHELL),
	    new LineMapping(Xraylib.L2Q1_LINE, Xraylib.L2L3_LINE, Xraylib.L2_SHELL),
	    new LineMapping(Xraylib.L3Q1_LINE, Xraylib.L3M1_LINE, Xraylib.L3_SHELL),
    	new LineMapping(Xraylib.M1P5_LINE, Xraylib.M1N1_LINE, Xraylib.M1_SHELL),
    	new LineMapping(Xraylib.M2P5_LINE, Xraylib.M2N1_LINE, Xraylib.M2_SHELL),
    	new LineMapping(Xraylib.M3Q1_LINE, Xraylib.M3N1_LINE, Xraylib.M3_SHELL),
    	new LineMapping(Xraylib.M4P5_LINE, Xraylib.M4N1_LINE, Xraylib.M4_SHELL),
    	new LineMapping(Xraylib.M5P5_LINE, Xraylib.M5N1_LINE, Xraylib.M5_SHELL)
	};

	@ParameterizedTest(name="test_cs_fluorline_all {index} -> {0}")
	@MethodSource("goodCSFluorLineKisselAllValuesProvider")
	public void test_cs_fluorline_all(CS_FluorShell_base shell_func, CS_FluorLine_base line_func, double[] expected_values) {
		for (int i = 0 ; i < test_lines.length ; i++) {
			int line = test_lines[i];
			double cs = line_func.execute(92, line, 120.0);
			assertEquals(cs/expected_values[i], 1.0, 1E-2);
		}

		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double ec = shell_func.execute(0, Xraylib.K_SHELL, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double ec = shell_func.execute(Xraylib.ZMAX, Xraylib.K_SHELL, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.INVALID_SHELL);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double ec = shell_func.execute(Xraylib.ZMAX + 1, Xraylib.K_SHELL, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double ec = shell_func.execute(1, Xraylib.K_SHELL, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.INVALID_SHELL);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double ec = shell_func.execute(92, Xraylib.KL3_LINE, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.INVALID_SHELL);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double ec = shell_func.execute(26, Xraylib.K_SHELL, 0.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double ec = shell_func.execute(26, Xraylib.K_SHELL, 1001.0);
		});
		assertEquals(exc.getMessage(), Xraylib.SPLINT_X_TOO_HIGH);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double ec = shell_func.execute(26, Xraylib.K_SHELL, 5.0);
		});
		assertEquals(exc.getMessage(), Xraylib.TOO_LOW_EXCITATION_ENERGY);

		for (LineMapping mapping: line_mappings) {
			double cs = shell_func.execute(92, mapping.shell, 120.0);
			double cs2 = 0;
			double rr = 0;

			for (int j = mapping.line_lower ; j <= mapping.line_upper ; j++) {
				try {
					rr += Xraylib.RadRate(92, j);
					cs2 += line_func.execute(92, j, 120.0);
				} catch (IllegalArgumentException e) {
					continue;
				}
			}
			assertEquals(cs2, rr * cs, 1E-6);
		}

	}
}
