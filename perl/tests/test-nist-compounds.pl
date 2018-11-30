use strict;
use Test::More;
use xraylib;
use xrltest;

my $list = xraylib::GetCompoundDataNISTList();
ok(scalar @{$list} eq 180, "TestNISTCompounds:test_good::list");
for my $i (0..179) {
	my $cd = xraylib::GetCompoundDataNISTByIndex($i);
	ok($cd->{name} eq $list->[$i], "TestNISTCompounds:test_good::index");
	$cd = xraylib::GetCompoundDataNISTByName($list->[$i]);
	ok($cd->{name} eq $list->[$i], "TestNISTCompounds:test_good::name");
}

my $cd = xraylib::GetCompoundDataNISTByIndex(5);
ok($cd->{'nElements'} eq 4, "TestNISTCompounds::test_good_nr5::nElements");
ok($cd->{'Elements'} ~~ [6, 7, 8, 18], "TestNISTCompounds::test_good_nr5::Elements");
ok($cd->{'massFractions'} ~~ [0.000124, 0.755267, 0.231781, 0.012827], "TestNISTCompounds::test_good_nr5::massFractions");
ok(xrltest::almost_equal($cd->{'density'}, 0.001205,), "TestNISTCompounds::test_good_nr5::density");
ok($cd->{name} eq "Air, Dry (near sea level)", "TestNISTCompounds::test_good_nr5::name");

$cd = xraylib::GetCompoundDataNISTByName("Air, Dry (near sea level)");
ok($cd->{'nElements'} eq 4, "TestNISTCompounds::test_good_Air::nElements");
ok($cd->{'Elements'} ~~ [6, 7, 8, 18], "TestNISTCompounds::test_good_Air::Elements");
ok($cd->{'massFractions'} ~~ [0.000124, 0.755267, 0.231781, 0.012827], "TestNISTCompounds::test_good_Air::massFractions");
ok(xrltest::almost_equal($cd->{'density'}, 0.001205,), "TestNISTCompounds::test_good_Air::density");
ok($cd->{name} eq "Air, Dry (near sea level)", "TestNISTCompounds::test_good_Air::name");

eval {
	xraylib::GetCompoundDataNISTByName('Uu');
};
like($@, qr/^ValueError/ , "TestNISTCompounds::test_bad::name1");

eval {
	xraylib::GetCompoundDataNISTByName(undef);
};
like($@, qr/^ValueError/ , "TestNISTCompounds::test_bad::name2");
eval {
	xraylib::GetCompoundDataNISTByIndex('Uu');
};
like($@, qr/^TypeError/ , "TestNISTCompounds::test_bad::index1");
xraylib::GetCompoundDataNISTByIndex(undef); # undef gets translated to zero apparently!
eval {
	xraylib::GetCompoundDataNISTByIndex(-1);
};
like($@, qr/^ValueError/ , "TestNISTCompounds::test_bad::index3");
eval {
	xraylib::GetCompoundDataNISTByIndex(180);
};
like($@, qr/^ValueError/ , "TestNISTCompounds::test_bad::index4");


done_testing();
