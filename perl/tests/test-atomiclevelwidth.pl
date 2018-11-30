use Test::More;
use xraylib;
use xrltest;

ok(xrltest::almost_equal(xraylib::AtomicLevelWidth(26, $xraylib::K_SHELL), 1.19E-3), "TestAtomicLevelWidth::test_Fe_K");
ok(xrltest::almost_equal(xraylib::AtomicLevelWidth(92, $xraylib::N7_SHELL), 0.31E-3), "TestAtomicLevelWidth::test_U_N7");
eval {
	xraylib::AtomicLevelWidth(185, $xraylib::K_SHELL);
};
like($@, qr/^ValueError/ , "TestAtomicLevelWidth::test_bad_Z");
eval {
	xraylib::AtomicLevelWidth(26, -5);
};
like($@, qr/^ValueError/ , "TestAtomicLevelWidth::test_bad_shell");
eval {
	xraylib::AtomicLevelWidth(26, $xraylib::N3_SHELL);
};
like($@, qr/^ValueError/ , "TestAtomicLevelWidth::test_invalid_shell");

done_testing();
