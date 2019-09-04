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
import com.github.tschoonj.xraylib.compoundDataNIST;

public class TestNISTCompounds {

	@Test
	public void test_list_consistency() {
		String[] list = Xraylib.GetCompoundDataNISTList();
		assertEquals(list.length, 180);

		for (int i = 0 ; i < list.length ; i++) {
			compoundDataNIST cdn = Xraylib.GetCompoundDataNISTByIndex(i);
			assertEquals(cdn.name, list[i]);
			cdn = Xraylib.GetCompoundDataNISTByName(list[i]);
			assertEquals(cdn.name, list[i]);
		}
	}

	@Test
	public void test_GetCompountDataNISTByIndex_5() {
		compoundDataNIST cdn = Xraylib.GetCompoundDataNISTByIndex(5);
		assertEquals(cdn.nElements, 4);
		assertEquals(cdn.density, 0.001205);
		assertArrayEquals(cdn.Elements, new int[]{6, 7, 8, 18});
		assertArrayEquals(cdn.massFractions, new double[]{0.000124, 0.755267, 0.231781, 0.012827}, 1E-6);
		assertEquals(cdn.name, "Air, Dry (near sea level)");
	}

	@Test
	public void test_GetCompountDataNISTByName_Air() {
		compoundDataNIST cdn = Xraylib.GetCompoundDataNISTByName("Air, Dry (near sea level)");
		assertEquals(cdn.nElements, 4);
		assertEquals(cdn.density, 0.001205);
		assertArrayEquals(cdn.Elements, new int[]{6, 7, 8, 18});
		assertArrayEquals(cdn.massFractions, new double[]{0.000124, 0.755267, 0.231781, 0.012827}, 1E-6);
		assertEquals(cdn.name, "Air, Dry (near sea level)");
	}

	@ParameterizedTest(name="test_bad_NIST_compounds {index} -> {arguments}")
	@NullAndEmptySource
	@ValueSource(strings = {"howehfofhwf"})
	public void test_bad_NIST_compounds(String compound) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			compoundDataNIST cdn = Xraylib.GetCompoundDataNISTByName(compound);
		});
		assertEquals(exc.getMessage(), String.format("%s was not found in the NIST compound database", compound));
	}

	@ParameterizedTest(name="test_bad_NIST_indices {index} -> {arguments}")
	@ValueSource(ints = {-1, 180})
	public void test_bad_NIST_indices(int index) {
		IllegalArgumentException exc = assertThrows(IllegalArgumentException.class, () -> {
			compoundDataNIST cdn = Xraylib.GetCompoundDataNISTByIndex(index);
		});
		assertEquals(exc.getMessage(), String.format("%d is out of the range of indices covered by the NIST compound database", index));
	}
}
