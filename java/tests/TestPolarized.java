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

public class TestPolarized {
	@FunctionalInterface
	private static interface CrossSectionZEnergyThetaPhiWrapper {
		public double execute(int Z, double energy, double theta, double phi);
	}

	static Stream<Arguments> goodCrossSectionZEnergyThetaPhiValuesProvider() {
		return Stream.of(
			arguments((CrossSectionZEnergyThetaPhiWrapper) Xraylib::DCSP_Rayl, Math.PI / 4.0, 0.17394690792051704),
			arguments((CrossSectionZEnergyThetaPhiWrapper) Xraylib::DCSP_Rayl, 0.0, 0.5788126901827545),
			arguments((CrossSectionZEnergyThetaPhiWrapper) Xraylib::DCSP_Compt, Math.PI / 4.0, 0.005489497545806118)
		);
	}

	@ParameterizedTest(name="test_cross_sections_z_energy_theta_phi_good_values {index} -> {0} {1}")
	@MethodSource("goodCrossSectionZEnergyThetaPhiValuesProvider")
	public void test_cross_sections_z_energy_theta_phi_good_values(CrossSectionZEnergyThetaPhiWrapper wrapper, double theta, double expected) {
		double cs = wrapper.execute(26, 10.0, theta, Math.PI / 4.0);
		assertEquals(cs, expected, 1E-4);
	}

	static Stream<Arguments> badCrossSectionZEnergyThetaPhiValuesProvider() {
		return Stream.of(
			arguments((CrossSectionZEnergyThetaPhiWrapper) Xraylib::DCSP_Rayl, 0, 10.0, Math.PI / 4.0, Xraylib.Z_OUT_OF_RANGE),
			arguments((CrossSectionZEnergyThetaPhiWrapper) Xraylib::DCSP_Rayl, Xraylib.ZMAX + 1, 10.0, Math.PI / 4.0, Xraylib.Z_OUT_OF_RANGE),
			arguments((CrossSectionZEnergyThetaPhiWrapper) Xraylib::DCSP_Rayl, 26, 0.0, Math.PI / 4.0, Xraylib.NEGATIVE_ENERGY),
			arguments((CrossSectionZEnergyThetaPhiWrapper) Xraylib::DCSP_Compt, 0, 10.0, Math.PI / 4.0, Xraylib.Z_OUT_OF_RANGE),
			arguments((CrossSectionZEnergyThetaPhiWrapper) Xraylib::DCSP_Compt, Xraylib.ZMAX + 1, 10.0, Math.PI / 4.0, Xraylib.Z_OUT_OF_RANGE),
			arguments((CrossSectionZEnergyThetaPhiWrapper) Xraylib::DCSP_Compt, 26, 0.0, Math.PI / 4.0, Xraylib.NEGATIVE_ENERGY),
			arguments((CrossSectionZEnergyThetaPhiWrapper) Xraylib::DCSP_Compt, 26, 10.0, 0.0, Xraylib.NEGATIVE_Q)
		);
	}

	@ParameterizedTest(name="test_cross_sections_z_energy_theta_phi_bad_values {index} -> {0} {1}")
	@MethodSource("badCrossSectionZEnergyThetaPhiValuesProvider")
	public void test_cross_sections_z_energy_theta_phi_bad_values(CrossSectionZEnergyThetaPhiWrapper wrapper, int Z, double energy, double theta, String message) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = wrapper.execute(Z, energy, theta, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), message);
	}

	@Test
	public void test_dcsp_kn_good() {
		double cs = Xraylib.DCSP_KN(10.0, Math.PI / 4, Math.PI / 4);
		assertEquals(cs, 0.05888029282784654, 1E-6);
	}

	@Test
	public void test_dcsp_kn_bad() {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = Xraylib.DCSP_KN(0.0, Math.PI / 4, Math.PI / 4);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);
	}

	@Test
	public void test_dcsp_thoms() {
		double cs = Xraylib.DCSP_Thoms(Math.PI / 4, Math.PI / 4);
		assertEquals(cs, 0.05955590775, 1E-6);
	}
}
