use Test::More;
use xraylib;
use xrltest;

ok(xrltest::almost_equal(xraylib::AtomicWeight(26), 55.850), "TestAtomicWeight::test_Fe");
ok(xrltest::almost_equal(xraylib::AtomicWeight(92), 238.070), "TestAtomicWeight::test_U");
eval {
	xraylib::AtomicWeight(185);
};
like($@, qr/^ValueError/ , "TestAtomicWeight::test_bad_Z");

done_testing();

