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
import com.github.tschoonj.xraylib.compoundData;
import com.github.tschoonj.xraylib.compoundDataNIST;

public class TestCrossSectionsCompound {

	public final static String COMPOUND = "Ca5(PO4)3F"; /* Fluorapatite */
	public final static String NIST_COMPOUND = "Ferrous Sulfate Dosimeter Solution";

	private static compoundData cd;
	private static compoundDataNIST cdn;

	@BeforeAll
	public static void init() {
		cd = Xraylib.CompoundParser(COMPOUND);
		cdn = Xraylib.GetCompoundDataNISTByName(NIST_COMPOUND);
	}

	@FunctionalInterface
	private static interface CrossSectionWrapperEnergy {
		public double execute(int Z, double energy);
	}

	@FunctionalInterface
	private static interface CrossSectionCPWrapperEnergy {
		public double execute(String compound, double energy);
	}

	static Stream<Arguments> wrapperEnergyValuesProvider() {
		return Stream.of(
			arguments((CrossSectionWrapperEnergy) Xraylib::CS_Total, (CrossSectionCPWrapperEnergy) Xraylib::CS_Total_CP),
			arguments((CrossSectionWrapperEnergy) Xraylib::CS_Photo, (CrossSectionCPWrapperEnergy) Xraylib::CS_Photo_CP),
			arguments((CrossSectionWrapperEnergy) Xraylib::CS_Compt, (CrossSectionCPWrapperEnergy) Xraylib::CS_Compt_CP),
			arguments((CrossSectionWrapperEnergy) Xraylib::CS_Rayl, (CrossSectionCPWrapperEnergy) Xraylib::CS_Rayl_CP),
			arguments((CrossSectionWrapperEnergy) Xraylib::CS_Total_Kissel, (CrossSectionCPWrapperEnergy) Xraylib::CS_Total_Kissel_CP),
			arguments((CrossSectionWrapperEnergy) Xraylib::CS_Photo_Total, (CrossSectionCPWrapperEnergy) Xraylib::CS_Photo_Total_CP),
			arguments((CrossSectionWrapperEnergy) Xraylib::CSb_Total, (CrossSectionCPWrapperEnergy) Xraylib::CSb_Total_CP),
			arguments((CrossSectionWrapperEnergy) Xraylib::CSb_Photo, (CrossSectionCPWrapperEnergy) Xraylib::CSb_Photo_CP),
			arguments((CrossSectionWrapperEnergy) Xraylib::CSb_Compt, (CrossSectionCPWrapperEnergy) Xraylib::CSb_Compt_CP),
			arguments((CrossSectionWrapperEnergy) Xraylib::CSb_Rayl, (CrossSectionCPWrapperEnergy) Xraylib::CSb_Rayl_CP),
			arguments((CrossSectionWrapperEnergy) Xraylib::CSb_Total_Kissel, (CrossSectionCPWrapperEnergy) Xraylib::CSb_Total_Kissel_CP),
			arguments((CrossSectionWrapperEnergy) Xraylib::CSb_Photo_Total, (CrossSectionCPWrapperEnergy) Xraylib::CSb_Photo_Total_CP),
			arguments((CrossSectionWrapperEnergy) Xraylib::CS_Energy, (CrossSectionCPWrapperEnergy) Xraylib::CS_Energy_CP)
		);
	}

	@ParameterizedTest(name="test_chemical_compounds_energy_values {index} -> {0} {1}")
	@MethodSource("wrapperEnergyValuesProvider")
	public void test_chemical_compounds_energy_values(CrossSectionWrapperEnergy cs_base, CrossSectionCPWrapperEnergy cs_cp) {
		double sum = 0.0;
		for (int i = 0 ; i < cd.nElements ; i++)
			sum += cs_base.execute(cd.Elements[i], 10.0) * cd.massFractions[i];
		assertEquals(sum, cs_cp.execute(COMPOUND, 10.0), 1E-6);
	}
	
	@ParameterizedTest(name="test_nist_compounds_energy_values {index} -> {0} {1}")
	@MethodSource("wrapperEnergyValuesProvider")
	public void test_nist_compounds_energy_values(CrossSectionWrapperEnergy cs_base, CrossSectionCPWrapperEnergy cs_cp) {
		double sum = 0.0;
		for (int i = 0 ; i < cdn.nElements ; i++)
			sum += cs_base.execute(cdn.Elements[i], 10.0) * cdn.massFractions[i];
		assertEquals(sum, cs_cp.execute(NIST_COMPOUND, 10.0), 1E-6);
	}
	
	@ParameterizedTest(name="test_bad_compounds_energy_values {index} -> {1}")
	@MethodSource("wrapperEnergyValuesProvider")
	public void test_bad_compounds_energy_values(CrossSectionWrapperEnergy cs_base, CrossSectionCPWrapperEnergy cs_cp) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = cs_cp.execute("jpewffpfjpwf", 10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.UNKNOWN_COMPOUND);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = cs_cp.execute(COMPOUND, -10.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = cs_cp.execute(null, -10.0);
		}, Xraylib.UNKNOWN_COMPOUND);
		assertEquals(exc.getMessage(), Xraylib.UNKNOWN_COMPOUND);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = cs_cp.execute("", -10.0);
		}, Xraylib.UNKNOWN_COMPOUND);
		assertEquals(exc.getMessage(), Xraylib.UNKNOWN_COMPOUND);
	}
	
	@FunctionalInterface
	private static interface CrossSectionWrapperEnergyTheta {
		public double execute(int Z, double energy, double theta);
	}

	@FunctionalInterface
	private static interface CrossSectionCPWrapperEnergyTheta {
		public double execute(String compound, double energy, double theta);
	}

	static Stream<Arguments> wrapperEnergyThetaValuesProvider() {
		return Stream.of(
			arguments((CrossSectionWrapperEnergyTheta) Xraylib::DCS_Rayl, (CrossSectionCPWrapperEnergyTheta) Xraylib::DCS_Rayl_CP),
			arguments((CrossSectionWrapperEnergyTheta) Xraylib::DCS_Compt, (CrossSectionCPWrapperEnergyTheta) Xraylib::DCS_Compt_CP),
			arguments((CrossSectionWrapperEnergyTheta) Xraylib::DCSb_Rayl, (CrossSectionCPWrapperEnergyTheta) Xraylib::DCSb_Rayl_CP),
			arguments((CrossSectionWrapperEnergyTheta) Xraylib::DCSb_Compt, (CrossSectionCPWrapperEnergyTheta) Xraylib::DCSb_Compt_CP)
		);
	}

	@ParameterizedTest(name="test_chemical_compounds_energy_theta_values {index} -> {0} {1}")
	@MethodSource("wrapperEnergyThetaValuesProvider")
	public void test_chemical_compounds_energy_theta_values(CrossSectionWrapperEnergyTheta cs_base, CrossSectionCPWrapperEnergyTheta cs_cp) {
		double sum = 0.0;
		for (int i = 0 ; i < cd.nElements ; i++)
			sum += cs_base.execute(cd.Elements[i], 10.0, Math.PI / 4.0) * cd.massFractions[i];
		assertEquals(sum, cs_cp.execute(COMPOUND, 10.0, Math.PI / 4.0), 1E-6);
	}
	
	@ParameterizedTest(name="test_nist_compounds_energy_theta_values {index} -> {0} {1}")
	@MethodSource("wrapperEnergyThetaValuesProvider")
	public void test_nist_compounds_energy_theta_values(CrossSectionWrapperEnergyTheta cs_base, CrossSectionCPWrapperEnergyTheta cs_cp) {
		double sum = 0.0;
		for (int i = 0 ; i < cdn.nElements ; i++)
			sum += cs_base.execute(cdn.Elements[i], 10.0, Math.PI / 4.0) * cdn.massFractions[i];
		assertEquals(sum, cs_cp.execute(NIST_COMPOUND, 10.0, Math.PI / 4.0), 1E-6);
	}
	
	@ParameterizedTest(name="test_bad_compounds_energy_theta_values {index} -> {1}")
	@MethodSource("wrapperEnergyThetaValuesProvider")
	public void test_bad_compounds_energy_theta_values(CrossSectionWrapperEnergyTheta cs_base, CrossSectionCPWrapperEnergyTheta cs_cp) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = cs_cp.execute("jpewffpfjpwf", 10.0, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.UNKNOWN_COMPOUND);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = cs_cp.execute(COMPOUND, -10.0, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = cs_cp.execute(null, -10.0, Math.PI / 4.0);
		}, Xraylib.UNKNOWN_COMPOUND);
		assertEquals(exc.getMessage(), Xraylib.UNKNOWN_COMPOUND);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = cs_cp.execute("", -10.0, Math.PI / 4.0);
		}, Xraylib.UNKNOWN_COMPOUND);
		assertEquals(exc.getMessage(), Xraylib.UNKNOWN_COMPOUND);
	}
	
	@FunctionalInterface
	private static interface CrossSectionWrapperEnergyThetaPhi {
		public double execute(int Z, double energy, double theta, double phi);
	}

	@FunctionalInterface
	private static interface CrossSectionCPWrapperEnergyThetaPhi {
		public double execute(String compound, double energy, double theta, double phi);
	}

	static Stream<Arguments> wrapperEnergyThetaPhiValuesProvider() {
		return Stream.of(
			arguments((CrossSectionWrapperEnergyThetaPhi) Xraylib::DCSP_Rayl, (CrossSectionCPWrapperEnergyThetaPhi) Xraylib::DCSP_Rayl_CP),
			arguments((CrossSectionWrapperEnergyThetaPhi) Xraylib::DCSP_Compt, (CrossSectionCPWrapperEnergyThetaPhi) Xraylib::DCSP_Compt_CP),
			arguments((CrossSectionWrapperEnergyThetaPhi) Xraylib::DCSPb_Rayl, (CrossSectionCPWrapperEnergyThetaPhi) Xraylib::DCSPb_Rayl_CP),
			arguments((CrossSectionWrapperEnergyThetaPhi) Xraylib::DCSPb_Compt, (CrossSectionCPWrapperEnergyThetaPhi) Xraylib::DCSPb_Compt_CP)
		);
	}

	@ParameterizedTest(name="test_chemical_compounds_energy_theta_phi_values {index} -> {0} {1}")
	@MethodSource("wrapperEnergyThetaPhiValuesProvider")
	public void test_chemical_compounds_energy_theta_phi_values(CrossSectionWrapperEnergyThetaPhi cs_base, CrossSectionCPWrapperEnergyThetaPhi cs_cp) {
		double sum = 0.0;
		for (int i = 0 ; i < cd.nElements ; i++)
			sum += cs_base.execute(cd.Elements[i], 10.0, Math.PI / 4.0, Math.PI / 4.0) * cd.massFractions[i];
		assertEquals(sum, cs_cp.execute(COMPOUND, 10.0, Math.PI / 4.0, Math.PI / 4.0), 1E-6);
	}
	
	@ParameterizedTest(name="test_nist_compounds_energy_theta_phi_values {index} -> {0} {1}")
	@MethodSource("wrapperEnergyThetaPhiValuesProvider")
	public void test_nist_compounds_energy_theta_phi_values(CrossSectionWrapperEnergyThetaPhi cs_base, CrossSectionCPWrapperEnergyThetaPhi cs_cp) {
		double sum = 0.0;
		for (int i = 0 ; i < cdn.nElements ; i++)
			sum += cs_base.execute(cdn.Elements[i], 10.0, Math.PI / 4.0, Math.PI / 4.0) * cdn.massFractions[i];
		assertEquals(sum, cs_cp.execute(NIST_COMPOUND, 10.0, Math.PI / 4.0, Math.PI / 4.0), 1E-6);
	}
	
	@ParameterizedTest(name="test_bad_compounds_energy_theta_phi_values {index} -> {1}")
	@MethodSource("wrapperEnergyThetaPhiValuesProvider")
	public void test_bad_compounds_energy_theta_phi_values(CrossSectionWrapperEnergyThetaPhi cs_base, CrossSectionCPWrapperEnergyThetaPhi cs_cp) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = cs_cp.execute("jpewffpfjpwf", 10.0, Math.PI / 4.0, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.UNKNOWN_COMPOUND);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = cs_cp.execute(COMPOUND, -10.0, Math.PI / 4.0, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.NEGATIVE_ENERGY);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = cs_cp.execute(null, -10.0, Math.PI / 4.0, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.UNKNOWN_COMPOUND);

		exc = assertThrows(IllegalArgumentException.class, () -> {
			double cs = cs_cp.execute("", -10.0, Math.PI / 4.0, Math.PI / 4.0);
		});
		assertEquals(exc.getMessage(), Xraylib.UNKNOWN_COMPOUND);
	}
}
