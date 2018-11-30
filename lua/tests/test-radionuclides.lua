require("xraylib")
luaunit = require("luaunit")

TestRadionuclides = {}
	function TestRadionuclides:test_good()
		list = xraylib.GetRadioNuclideDataList()
		luaunit.assertEquals(#list, 10)
		for i,v in ipairs(list) do
			rnd = xraylib.GetRadioNuclideDataByIndex(i-1)
			luaunit.assertEquals(rnd['name'], v)
			rnd = xraylib.GetRadioNuclideDataByName(v)
			luaunit.assertEquals(rnd['name'], v)
		end

		rnd = xraylib.GetRadioNuclideDataByIndex(3)
		luaunit.assertEquals(rnd['A'], 125)
		luaunit.assertAlmostEquals(rnd['GammaEnergies'][1], 35.4919, 1E-6)
		luaunit.assertAlmostEquals(rnd['GammaIntensities'][1], 0.0668, 1E-6)
		luaunit.assertEquals(rnd['N'], 72)
		XrayIntensities = {0.0023, 0.00112, 0.0063, 0.056, 0.035, 0.0042, 0.007, 0.00043, 0.0101, 0.0045, 0.00103, 0.0016, 3.24e-05, 0.406, 0.757, 0.0683, 0.132, 0.00121, 0.0381, 0.0058}
		for i,v in ipairs(rnd['XrayIntensities']) do
			luaunit.assertAlmostEquals(v, XrayIntensities[i])
		end
		XrayLines = {-86, -60, -89, -90, -63, -33, -34, -91, -95, -68, -38, -39, -1, -2, -3, -5, -6, -8, -11, -13}
		luaunit.assertEquals(rnd['XrayLines'], XrayLines)
		luaunit.assertEquals(rnd['Z'], 53)
		luaunit.assertEquals(rnd['Z_xray'], 52)
		luaunit.assertEquals(rnd['nGammas'], 1)
		luaunit.assertEquals(rnd['nXrays'], 20)
		luaunit.assertEquals(rnd['name'], '125I')

		rnd = xraylib.GetRadioNuclideDataByName('125I')
		luaunit.assertEquals(rnd['A'], 125)
		luaunit.assertAlmostEquals(rnd['GammaEnergies'][1], 35.4919, 1E-6)
		luaunit.assertAlmostEquals(rnd['GammaIntensities'][1], 0.0668, 1E-6)
		luaunit.assertEquals(rnd['N'], 72)
		XrayIntensities = {0.0023, 0.00112, 0.0063, 0.056, 0.035, 0.0042, 0.007, 0.00043, 0.0101, 0.0045, 0.00103, 0.0016, 3.24e-05, 0.406, 0.757, 0.0683, 0.132, 0.00121, 0.0381, 0.0058}
		for i,v in ipairs(rnd['XrayIntensities']) do
			luaunit.assertAlmostEquals(v, XrayIntensities[i])
		end
		luaunit.assertEquals(rnd['Z'], 53)
		luaunit.assertEquals(rnd['Z_xray'], 52)
		luaunit.assertEquals(rnd['nGammas'], 1)
		luaunit.assertEquals(rnd['nXrays'], 20)
		luaunit.assertEquals(rnd['name'], '125I')
	end

	function TestRadionuclides:test_bad()
		luaunit.assertError(xraylib.GetRadioNuclideDataByName, "sjalala")
		luaunit.assertError(xraylib.GetRadioNuclideDataByName, nil)
		luaunit.assertError(xraylib.GetRadioNuclideDataByName, 2)
		luaunit.assertError(xraylib.GetRadioNuclideDataByName, {'125I'})
		luaunit.assertError(xraylib.GetRadioNuclideDataByIndex, {1})
		luaunit.assertError(xraylib.GetRadioNuclideDataByIndex, -1)
		luaunit.assertError(xraylib.GetRadioNuclideDataByIndex, nil)
		luaunit.assertError(xraylib.GetRadioNuclideDataByIndex, "jhopewjffj")
	end
os.exit(luaunit.LuaUnit.run('-v', '-o', 'tap'))
