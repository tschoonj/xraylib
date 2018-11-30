require "xraylib"
require "test/unit"

class TestNISTCompounds < Test::Unit::TestCase
	def test_good
		list = Xraylib.GetCompoundDataNISTList()
		assert_equal(list.length, 180)
		list.each_with_index do |v, i|
			cdn = Xraylib.GetCompoundDataNISTByIndex(i)
			assert_equal(cdn['name'], v)
			cdn = Xraylib.GetCompoundDataNISTByName(v)
			assert_equal(cdn['name'], v)
		end
		cdn = Xraylib.GetCompoundDataNISTByIndex(5)
		assert_equal(cdn['nElements'], 4)
		assert_in_delta(cdn['density'], 0.001205, 1E-6)
		assert_equal(cdn['Elements'], [6, 7, 8, 18])
		[0.000124, 0.755267, 0.231781, 0.012827].each_with_index do |v, i|
			assert_in_delta(cdn['massFractions'][i], v, 1E-6)
		end
		cdn = Xraylib.GetCompoundDataNISTByName('Air, Dry (near sea level)')
		assert_equal(cdn['nElements'], 4)
		assert_in_delta(cdn['density'], 0.001205, 1E-6)
		assert_equal(cdn['Elements'], [6, 7, 8, 18])
		[0.000124, 0.755267, 0.231781, 0.012827].each_with_index do |v, i|
			assert_in_delta(cdn['massFractions'][i], v, 1E-6)
		end
	end

	def test_bad
        	assert_raise(ArgumentError) {
			Xraylib.GetCompoundDataNISTByName("jwjfpfj")
		}
        	assert_raise(TypeError) {
			Xraylib.GetCompoundDataNISTByName(0)
		}
        	assert_raise(ArgumentError) {
			Xraylib.GetCompoundDataNISTByName(nil)
		}
        	assert_raise(ArgumentError) {
			Xraylib.GetCompoundDataNISTByIndex(-1)
		}
        	assert_raise(ArgumentError) {
			Xraylib.GetCompoundDataNISTByIndex(180)
		}
        	assert_raise(TypeError) {
			Xraylib.GetCompoundDataNISTByIndex(nil)
		}
        	assert_raise(TypeError) {
			Xraylib.GetCompoundDataNISTByIndex("jpwejfp")
		}
	end
end
