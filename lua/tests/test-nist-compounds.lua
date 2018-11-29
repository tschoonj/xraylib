require("xraylib")
luaunit = require("luaunit")

TestNISTCompounds = {}
	function TestNISTCompounds:test_good()
		list = xraylib.GetCompoundDataNISTList()
		luaunit.assertEquals(#list, 180)
		for i,v in ipairs(list) do
			cdn = xraylib.GetCompoundDataNISTByIndex(i-1)
			luaunit.assertEquals(cdn['name'], v)
			cdn = xraylib.GetCompoundDataNISTByName(v)
			luaunit.assertEquals(cdn['name'], v)
		end
		cdn = xraylib.GetCompoundDataNISTByIndex(5)
		luaunit.assertAlmostEquals(cdn['density'], 0.001205, 1E-6)
		luaunit.assertEquals(cdn['nElements'], 4)
		luaunit.assertEquals(cdn['Elements'], {6, 7, 8, 18})
		luaunit.assertAlmostEquals(cdn['massFractions'][1], 0.000124, 1E-6)
		luaunit.assertAlmostEquals(cdn['massFractions'][2], 0.755267, 1E-6)
		luaunit.assertAlmostEquals(cdn['massFractions'][3], 0.231781, 1E-6)
		luaunit.assertAlmostEquals(cdn['massFractions'][4], 0.012827, 1E-6)
		cdn = xraylib.GetCompoundDataNISTByName('Air, Dry (near sea level)')
		luaunit.assertAlmostEquals(cdn['density'], 0.001205, 1E-6)
		luaunit.assertEquals(cdn['nElements'], 4)
		luaunit.assertEquals(cdn['Elements'], {6, 7, 8, 18})
		luaunit.assertAlmostEquals(cdn['massFractions'][1], 0.000124, 1E-6)
		luaunit.assertAlmostEquals(cdn['massFractions'][2], 0.755267, 1E-6)
		luaunit.assertAlmostEquals(cdn['massFractions'][3], 0.231781, 1E-6)
		luaunit.assertAlmostEquals(cdn['massFractions'][4], 0.012827, 1E-6)
	end

	function TestNISTCompounds:test_bad()
		luaunit.assertError(xraylib.GetCompoundDataNISTByIndex, -1)
		luaunit.assertError(xraylib.GetCompoundDataNISTByIndex, 180)
		luaunit.assertError(xraylib.GetCompoundDataNISTByIndex, nil)
		luaunit.assertError(xraylib.GetCompoundDataNISTByIndex, "jpjffpjwf")
		luaunit.assertError(xraylib.GetCompoundDataNISTByIndex, {2, 3})
		luaunit.assertError(xraylib.GetCompoundDataNISTByName, "non-existent-compound")
		luaunit.assertError(xraylib.GetCompoundDataNISTByName, nil)
		luaunit.assertError(xraylib.GetCompoundDataNISTByName, 2)
		luaunit.assertError(xraylib.GetCompoundDataNISTByName, {"Air, Dry (near sea level)"})
	end

os.exit(luaunit.LuaUnit.run('-v', '-o', 'tap'))
