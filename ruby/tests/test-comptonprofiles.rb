require "xraylib"
require "test/unit"

class TestComptonProfiles < Test::Unit::TestCase
	def test_pz_0
		profile = Xraylib.ComptonProfile(26, 0.0)
		assert_in_delta(profile, 7.060, 1E-6)

		profile = Xraylib.ComptonProfile_Partial(26, Xraylib::N1_SHELL, 0.0)
		assert_in_delta(profile, 1.550, 1E-6)

		profile1 = Xraylib.ComptonProfile_Partial(26, Xraylib::L2_SHELL, 0.0)
		profile2 = Xraylib.ComptonProfile_Partial(26, Xraylib::L3_SHELL, 0.0)
		assert_in_delta(profile1, profile2, 1E-6)
		assert_in_delta(profile1, 0.065, 1E-6)
	end

	def test_pz_100
		profile = Xraylib.ComptonProfile(26, 100.0)
		assert_in_delta(profile, 1.8E-5, 1E-8)

		profile = Xraylib.ComptonProfile_Partial(26, Xraylib::N1_SHELL, 100.0)
		assert_in_delta(profile, 5.1E-9, 1E-12)

		profile1 = Xraylib.ComptonProfile_Partial(26, Xraylib::L2_SHELL, 100.0)
		profile2 = Xraylib.ComptonProfile_Partial(26, Xraylib::L3_SHELL, 100.0)
		assert_in_delta(profile1, profile2, 1E-10)
		assert_in_delta(profile1, 1.1E-8, 1E-10)
	end

	def test_pz_50
		profile = Xraylib.ComptonProfile(26, 50.0)
		assert_in_delta(profile, 0.0006843950273082384, 1E-8)

		profile = Xraylib.ComptonProfile_Partial(26, Xraylib::N1_SHELL, 50.0)
		assert_in_delta(profile, 2.4322755767709126e-07, 1E-10)

		profile1 = Xraylib.ComptonProfile_Partial(26, Xraylib::L2_SHELL, 50.0)
		profile2 = Xraylib.ComptonProfile_Partial(26, Xraylib::L3_SHELL, 50.0)
		assert_in_delta(profile1, profile2, 1E-10)
		assert_in_delta(profile1, 2.026953933016568e-06, 1E-10)
	end

	def test_bad_input
		assert_raise(ArgumentError) {
			Xraylib.ComptonProfile(0, 0.0)
		}
		Xraylib.ComptonProfile(102, 0.0)
		assert_raise(ArgumentError) {
			Xraylib.ComptonProfile(103, 0.0)
		}
		assert_raise(ArgumentError) {
			Xraylib.ComptonProfile(26, -1.0)
		}
		assert_raise(ArgumentError) {
			Xraylib.ComptonProfile_Partial(0, Xraylib::K_SHELL, 0.0)
		}
		Xraylib.ComptonProfile_Partial(102, Xraylib::K_SHELL, 0.0)
		assert_raise(ArgumentError) {
			Xraylib.ComptonProfile_Partial(103, Xraylib::K_SHELL, 0.0)
		}
		assert_raise(ArgumentError) {
			Xraylib.ComptonProfile_Partial(26, Xraylib::K_SHELL, -1.0)
		}
		assert_raise(ArgumentError) {
			Xraylib.ComptonProfile_Partial(26, -1, 0.0)
		}
		assert_raise(ArgumentError) {
			Xraylib.ComptonProfile_Partial(26, Xraylib::N2_SHELL, 0.0)
		}
		assert_raise(ArgumentError) {
			Xraylib.ComptonProfile_Partial()
		}
		assert_raise(TypeError) {
			Xraylib.ComptonProfile_Partial("26", Xraylib::N2_SHELL, 0.0)
		}
	end
end

