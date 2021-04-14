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

public class TestCrossSectionsBarns {

	@FunctionalInterface
	private static interface CrossSectionWrapperEnergy {
		public double execute(int Z, double energy);
	}

	static Stream<Arguments> energyValuesProvider() {
		return Stream.of(
			arguments((CrossSectionWrapperEnergy) Xraylib::CS_Total, (CrossSectionWrapperEnergy) Xraylib::CSb_Total),
			arguments((CrossSectionWrapperEnergy) Xraylib::CS_Rayl, (CrossSectionWrapperEnergy) Xraylib::CSb_Rayl),
			arguments((CrossSectionWrapperEnergy) Xraylib::CS_Compt, (CrossSectionWrapperEnergy) Xraylib::CSb_Compt),
			arguments((CrossSectionWrapperEnergy) Xraylib::CS_Photo, (CrossSectionWrapperEnergy) Xraylib::CSb_Photo)
		);
	}

	@ParameterizedTest(name="test_good_energy_values {index} -> {0} {1}")
	@MethodSource("energyValuesProvider")
	public void test_good_energy_values(CrossSectionWrapperEnergy base, CrossSectionWrapperEnergy barns) {
		double cs = base.execute(26, 10.0);
		double aw = Xraylib.AtomicWeight(26);
		assertEquals(barns.execute(26, 10.0), cs * aw / Xraylib.AVOGNUM);
	}

	@ParameterizedTest(name="test_bad_energy_values {index} -> {0} {1}")
	@MethodSource("energyValuesProvider")
	public void test_bad_energy_values(CrossSectionWrapperEnergy base, CrossSectionWrapperEnergy barns) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(-1, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(Xraylib.ZMAX, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(26, 0.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);
	}

	@FunctionalInterface
	private static interface CrossSectionWrapperLineEnergy {
		public double execute(int Z, int line, double energy);
	}

	static Stream<Arguments> lineEnergyValuesProvider() {
		return Stream.of(
			arguments((CrossSectionWrapperLineEnergy) Xraylib::CS_FluorLine, (CrossSectionWrapperLineEnergy) Xraylib::CSb_FluorLine)
		);
	}

	@ParameterizedTest(name="test_good_line_energy_values {index} -> {0} {1}")
	@MethodSource("lineEnergyValuesProvider")
	public void test_good_line_energy_values(CrossSectionWrapperLineEnergy base, CrossSectionWrapperLineEnergy barns) {
		double cs = base.execute(26, Xraylib.KL3_LINE, 10.0);
		double aw = Xraylib.AtomicWeight(26);
		assertEquals(barns.execute(26, Xraylib.KL3_LINE, 10.0), cs * aw / Xraylib.AVOGNUM);
	}

	@ParameterizedTest(name="test_bad_line_energy_values {index} -> {0} {1}")
	@MethodSource("lineEnergyValuesProvider")
	public void test_bad_line_energy_values(CrossSectionWrapperLineEnergy base, CrossSectionWrapperLineEnergy barns) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(-1, Xraylib.KL3_LINE, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(Xraylib.ZMAX, Xraylib.KL3_LINE, 10.0);
		});
		exc.printStackTrace();
		assertEquals(exc.getMessage(), Xraylib.INVALID_LINE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(26, -500, 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.INVALID_LINE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(26, Xraylib.KL3_LINE, 0.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);
	}

	@FunctionalInterface
	private static interface CrossSectionWrapperEnergyTheta {
		public double execute(int Z, double energy, double theta);
	}

	static Stream<Arguments> energyThetaValuesProvider() {
		return Stream.of(
			arguments((CrossSectionWrapperEnergyTheta) Xraylib::DCS_Rayl, (CrossSectionWrapperEnergyTheta) Xraylib::DCSb_Rayl),
			arguments((CrossSectionWrapperEnergyTheta) Xraylib::DCS_Compt, (CrossSectionWrapperEnergyTheta) Xraylib::DCSb_Compt)
		);
	}

	@ParameterizedTest(name="test_good_energy_theta_values {index} -> {0} {1}")
	@MethodSource("energyThetaValuesProvider")
	public void test_good_energy_theta_values(CrossSectionWrapperEnergyTheta base, CrossSectionWrapperEnergyTheta barns) {
		double cs = base.execute(26, 10.0, Math.PI / 4.0);
		double aw = Xraylib.AtomicWeight(26);
		assertEquals(barns.execute(26, 10.0, Math.PI / 4.0), cs * aw / Xraylib.AVOGNUM);
	}

	@ParameterizedTest(name="test_bad_energy_theta_values {index} -> {0} {1}")
	@MethodSource("energyThetaValuesProvider")
	public void test_bad_energy_theta_values(CrossSectionWrapperEnergyTheta base, CrossSectionWrapperEnergyTheta barns) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(-1, 10.0, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(Xraylib.ZMAX, 10.0, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(26, -10.0, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);
	}

	@FunctionalInterface
	private static interface CrossSectionWrapperEnergyThetaPhi {
		public double execute(int Z, double energy, double theta, double phi);
	}

	static Stream<Arguments> energyThetaPhiValuesProvider() {
		return Stream.of(
			arguments((CrossSectionWrapperEnergyThetaPhi) Xraylib::DCSP_Rayl, (CrossSectionWrapperEnergyThetaPhi) Xraylib::DCSPb_Rayl),
			arguments((CrossSectionWrapperEnergyThetaPhi) Xraylib::DCSP_Compt, (CrossSectionWrapperEnergyThetaPhi) Xraylib::DCSPb_Compt)
		);
	}

	@ParameterizedTest(name="test_good_energy_theta_phi_values {index} -> {0} {1}")
	@MethodSource("energyThetaPhiValuesProvider")
	public void test_good_energy_theta_phi_values(CrossSectionWrapperEnergyThetaPhi base, CrossSectionWrapperEnergyThetaPhi barns) {
		double cs = base.execute(26, 10.0, Math.PI / 4.0, Math.PI / 4.0);
		double aw = Xraylib.AtomicWeight(26);
		assertEquals(barns.execute(26, 10.0, Math.PI / 4.0, Math.PI / 4.0), cs * aw / Xraylib.AVOGNUM);
	}

	@ParameterizedTest(name="test_bad_energy_theta_phi_values {index} -> {0} {1}")
	@MethodSource("energyThetaPhiValuesProvider")
	public void test_bad_energy_theta_phi_values(CrossSectionWrapperEnergyThetaPhi base, CrossSectionWrapperEnergyThetaPhi barns) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(-1, 10.0, Math.PI / 4.0, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(Xraylib.ZMAX, 10.0, Math.PI / 4.0, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.Z_OUT_OF_RANGE);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = barns.execute(26, -10.0, Math.PI / 4.0, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);
	}
}
