require("xraylib")
luaunit = require("luaunit")

TestComptonProfiles = {}
	function TestComptonProfiles:test_pz_0()
		local profile, profile1, profile2

		profile = xraylib.ComptonProfile(26, 0.0)
		luaunit.assertAlmostEquals(profile, 7.060, 1E-6)

		profile = xraylib.ComptonProfile_Partial(26, xraylib.N1_SHELL, 0.0)
		luaunit.assertAlmostEquals(profile, 1.550, 1E-6)

		profile1 = xraylib.ComptonProfile_Partial(26, xraylib.L2_SHELL, 0.0)
		profile2 = xraylib.ComptonProfile_Partial(26, xraylib.L3_SHELL, 0.0)
		luaunit.assertAlmostEquals(profile1, profile2, 1E-6)
		luaunit.assertAlmostEquals(profile1, 0.065, 1E-6)
	end

	function TestComptonProfiles:test_pz_100()
		local profile, profile1, profile2

		profile = xraylib.ComptonProfile(26, 100.0)
		luaunit.assertAlmostEquals(profile, 1.8E-5, 1E-8)

		profile = xraylib.ComptonProfile_Partial(26, xraylib.N1_SHELL, 100.0)
		luaunit.assertAlmostEquals(profile, 5.1E-9, 1E-12)

		profile1 = xraylib.ComptonProfile_Partial(26, xraylib.L2_SHELL, 100.0)
		profile2 = xraylib.ComptonProfile_Partial(26, xraylib.L3_SHELL, 100.0)
		luaunit.assertAlmostEquals(profile1, profile2, 1E-10)
		luaunit.assertAlmostEquals(profile1, 1.1E-8, 1E-10)
	end

	function TestComptonProfiles:test_pz_50()
		local profile, profile1, profile2

		profile = xraylib.ComptonProfile(26, 50.0)
		luaunit.assertAlmostEquals(profile, 0.0006843950273082384, 1E-8)

		profile = xraylib.ComptonProfile_Partial(26, xraylib.N1_SHELL, 50.0)
		luaunit.assertAlmostEquals(profile, 2.4322755767709126e-07, 1E-10)

		profile1 = xraylib.ComptonProfile_Partial(26, xraylib.L2_SHELL, 50.0)
		profile2 = xraylib.ComptonProfile_Partial(26, xraylib.L3_SHELL, 50.0)
		luaunit.assertAlmostEquals(profile1, profile2, 1E-10)
		luaunit.assertAlmostEquals(profile1, 2.026953933016568e-06, 1E-10)
	end

	function TestComptonProfiles:test_bad_input()
		luaunit.assertError(xraylib.ComptonProfile, 0, 0.0)
		xraylib.ComptonProfile(102, 0.0)
		luaunit.assertError(xraylib.ComptonProfile, 103, 0.0)
		luaunit.assertError(xraylib.ComptonProfile, 26, -1.0)

		luaunit.assertError(xraylib.ComptonProfile_Partial, 0, xraylib.K_SHELL, 0.0)
		xraylib.ComptonProfile_Partial( 102, xraylib.K_SHELL, 0.0)
		luaunit.assertError(xraylib.ComptonProfile_Partial, 103, xraylib.K_SHELL, 0.0)
		luaunit.assertError(xraylib.ComptonProfile_Partial, 26, xraylib.K_SHELL, -1.0)
		luaunit.assertError(xraylib.ComptonProfile_Partial, 26, -1, 0.0)
		luaunit.assertError(xraylib.ComptonProfile_Partial, 26, xraylib.N2_SHELL, 0.0)
	end

os.exit(luaunit.LuaUnit.run('-v', '-o', 'tap'))
