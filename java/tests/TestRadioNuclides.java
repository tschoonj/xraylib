import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
 
import com.github.tschoonj.xraylib.Xraylib;
import com.github.tschoonj.xraylib.radioNuclideData;

public class TestRadioNuclides {

	@Test
	public void test_list_consistency() {
		String[] list = Xraylib.GetRadioNuclideDataList();
		assertEquals(list.length, 10);

		for (int i = 0 ; i < list.length ; i++) {
			radioNuclideData rnd = Xraylib.GetRadioNuclideDataByIndex(i);
			assertEquals(rnd.name, list[i]);
			rnd = Xraylib.GetRadioNuclideDataByName(list[i]);
			assertEquals(rnd.name, list[i]);
		}
	}

	@Test
	public void test_GetRadioNuclideDataByIndex_3() {
		radioNuclideData rnd = Xraylib.GetRadioNuclideDataByIndex(3);
		assertEquals(rnd.A, 125);
		assertArrayEquals(rnd.GammaEnergies, new double[]{35.4919});
		assertArrayEquals(rnd.GammaIntensities, new double[]{0.0668});
		assertEquals(rnd.N, 72);
		double[] XrayIntensities = new double[]{0.0023, 0.00112, 0.0063, 0.056, 0.035, 0.0042, 0.007, 0.00043, 0.0101, 0.0045, 0.00103, 0.0016, 3.24e-05, 0.406, 0.757, 0.0683, 0.132, 0.00121, 0.0381, 0.0058};
		assertArrayEquals(rnd.XrayIntensities, XrayIntensities);
		int[] XrayLines = new int[]{-86, -60, -89, -90, -63, -33, -34, -91, -95, -68, -38, -39, -1, -2, -3, -5, -6, -8, -11, -13};
		assertArrayEquals(rnd.XrayLines, XrayLines);
		assertEquals(rnd.Z, 53);
		assertEquals(rnd.Z_xray, 52);
		assertEquals(rnd.nGammas, 1);
		assertEquals(rnd.nXrays, 20);
		assertEquals(rnd.name, "125I");
	}

	@Test
	public void test_GetRadioNuclideDataByName_125I() {
		radioNuclideData rnd = Xraylib.GetRadioNuclideDataByName("125I");
		assertEquals(rnd.A, 125);
		assertArrayEquals(rnd.GammaEnergies, new double[]{35.4919});
		assertArrayEquals(rnd.GammaIntensities, new double[]{0.0668});
		assertEquals(rnd.N, 72);
		double[] XrayIntensities = new double[]{0.0023, 0.00112, 0.0063, 0.056, 0.035, 0.0042, 0.007, 0.00043, 0.0101, 0.0045, 0.00103, 0.0016, 3.24e-05, 0.406, 0.757, 0.0683, 0.132, 0.00121, 0.0381, 0.0058};
		assertArrayEquals(rnd.XrayIntensities, XrayIntensities);
		int[] XrayLines = new int[]{-86, -60, -89, -90, -63, -33, -34, -91, -95, -68, -38, -39, -1, -2, -3, -5, -6, -8, -11, -13};
		assertArrayEquals(rnd.XrayLines, XrayLines);
		assertEquals(rnd.Z, 53);
		assertEquals(rnd.Z_xray, 52);
		assertEquals(rnd.nGammas, 1);
		assertEquals(rnd.nXrays, 20);
		assertEquals(rnd.name, "125I");
	}


	@ParameterizedTest(name="test_bad_radionuclices_names {index} -> {arguments}")
	@NullAndEmptySource
	@ValueSource(strings = {"howehfofhwf"})
	public void test_bad_radionuclices_names(String nuclide) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			radioNuclideData rnd = Xraylib.GetRadioNuclideDataByName(nuclide);
		});
		assertEquals(exc.getMessage(), String.format("%s was not found in the radionuclide database", nuclide));
	}

	@ParameterizedTest(name="test_bad_radionuclides_indices {index} -> {arguments}")
	@ValueSource(ints = {-1, 10})
	public void test_bad_radionuclides_indices(int index) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			radioNuclideData rnd = Xraylib.GetRadioNuclideDataByIndex(index);
		});
		assertEquals(exc.getMessage(), String.format("%d is out of the range of indices covered by the radionuclide database", index));
	}
}
