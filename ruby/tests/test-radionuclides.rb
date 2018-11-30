require "xraylib"
require "test/unit"

class TestRadionuclides < Test::Unit::TestCase
	def test_good
		list = Xraylib.GetRadioNuclideDataList()
		assert_equal(list.length, 10)
		list.each_with_index do |v, i|
			rnd = Xraylib.GetRadioNuclideDataByIndex(i)
			assert_equal(rnd['name'], v)
			rnd = Xraylib.GetRadioNuclideDataByName(v)
			assert_equal(rnd['name'], v)
		end

		xray_intensities = [0.0023, 0.00112, 0.0063, 0.056, 0.035, 0.0042, 0.007, 0.00043, 0.0101, 0.0045, 0.00103, 0.0016, 3.24e-05, 0.406, 0.757, 0.0683, 0.132, 0.00121, 0.0381, 0.0058]
 		xray_lines = [-86, -60, -89, -90, -63, -33, -34, -91, -95, -68, -38, -39, -1, -2, -3, -5, -6, -8, -11, -13]

		rnd = Xraylib.GetRadioNuclideDataByIndex(3)
		assert_equal(rnd['A'], 125)
		assert_in_delta(rnd['GammaEnergies'][0], 35.4919, 1E-6)
		assert_in_delta(rnd['GammaIntensities'][0], 0.0668, 1E-6)
		assert_equal(rnd['N'], 72)
		xray_intensities.each_with_index do |v, i|
			assert_in_delta(rnd['XrayIntensities'][i], v, 1E-6)
		end
		assert_equal(rnd['XrayLines'], xray_lines)
		assert_equal(rnd['Z'], 53)
		assert_equal(rnd['Z_xray'], 52)
		assert_equal(rnd['nGammas'], 1)
		assert_equal(rnd['nXrays'], 20)
		assert_equal(rnd['name'], '125I')
		
		rnd = Xraylib.GetRadioNuclideDataByName("125I")
		assert_equal(rnd['A'], 125)
		assert_in_delta(rnd['GammaEnergies'][0], 35.4919, 1E-6)
		assert_in_delta(rnd['GammaIntensities'][0], 0.0668, 1E-6)
		assert_equal(rnd['N'], 72)
		xray_intensities.each_with_index do |v, i|
			assert_in_delta(rnd['XrayIntensities'][i], v, 1E-6)
		end
		assert_equal(rnd['XrayLines'], xray_lines)
		assert_equal(rnd['Z'], 53)
		assert_equal(rnd['Z_xray'], 52)
		assert_equal(rnd['nGammas'], 1)
		assert_equal(rnd['nXrays'], 20)
		assert_equal(rnd['name'], '125I')
	end

	def test_bad
        	assert_raise(ArgumentError) {
			Xraylib.GetRadioNuclideDataByName("jwjfpfj")
		}
        	assert_raise(TypeError) {
			Xraylib.GetRadioNuclideDataByName(0)
		}
        	assert_raise(ArgumentError) {
			Xraylib.GetRadioNuclideDataByName(nil)
		}
        	assert_raise(ArgumentError) {
			Xraylib.GetRadioNuclideDataByIndex(-1)
		}
        	assert_raise(ArgumentError) {
			Xraylib.GetRadioNuclideDataByIndex(10)
		}
        	assert_raise(TypeError) {
			Xraylib.GetRadioNuclideDataByIndex(nil)
		}
        	assert_raise(TypeError) {
			Xraylib.GetRadioNuclideDataByIndex("jpwejfp")
		}
	end
end

