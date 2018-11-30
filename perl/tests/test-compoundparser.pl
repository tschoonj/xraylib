use strict;
use Test::More;
use xraylib;
use xrltest;

my @good_compounds = qw/
        C19H29COOH
        C12H10
        C12H6O2
        C6H5Br
        C3H4OH(COOH)3
        HOCH2CH2OH
        C5H11NO2
        CH3CH(CH3)CH3
        NH2CH(C4H5N2)COOH
        H2O
        Ca5(PO4)3F
        Ca5(PO4)3OH
        Ca5.522(PO4.48)3OH
        Ca5.522(PO.448)3OH
/;

my @bad_compounds = (
       "CuI2ww",
       "0C",
       "2O",
       "13Li",
       "2(NO3)",
       "H(2)",
       "Ba(12)",
       "Cr(5)3",
       "Pb(13)2",
       "Au(22)11",
       "Au11(H3PO4)2)",
       "Au11(H3PO4))2",
       "Au(11(H3PO4))2",
       "Ca5.522(PO.44.8)3OH",
       "Ba[12]",
       "Auu1",
       "AuL1",
       undef,
       "  ",
       "\t",
       "\n",
       "Au L1",
       "Au\tFe",
       26
      );

for my $compound (@good_compounds) {
	ok(ref(xraylib::CompoundParser($compound)) eq 'HASH', "TestCompoundParser::test_good_compound ".$compound);
}

for my $compound (@bad_compounds) {
	eval {
		xraylib::CompoundParser($compound);
	};
	like($@, qr/Error/, "TestCompoundParser::test_bad_compound ".$compound);
}

my $cd = xraylib::CompoundParser("H2SO4");
ok($cd->{'nElements'} eq 3, "TestCompoundParser::test_H2SO4::nElements");
ok(xrltest::almost_equal($cd->{'molarMass'}, 98.09), "TestCompoundParser::test_H2SO4::molarMass");
ok(xrltest::almost_equal($cd->{'nAtomsAll'}, 7.0), "TestCompoundParser::test_H2SO4::nAtomsAll");
ok($cd->{'Elements'} ~~ [1, 8, 16], "TestCompoundParser::test_H2SO4::Elements");
ok($cd->{'massFractions'} ~~ [0.02059333265368539, 0.6524620246712203, 0.32694464267509427], "TestCompoundParser::test_H2SO4::massFractions");
ok($cd->{'nAtoms'} ~~ [2.0, 4.0, 1.0], "TestCompoundParser::test_H2SO4::nAtoms");



ok(xraylib::SymbolToAtomicNumber("Fe") eq 26, "TestSymbolToAtomicNumber::test_Fe");
eval {
	xraylib::SymbolToAtomicNumber('Uu');
};
like($@, qr/^ValueError/ , "TestSymbolToAtomicNumber::test_bad_symbol");
eval {
	xraylib::SymbolToAtomicNumber(26);
};
like($@, qr/^TypeError/ , "TestSymbolToAtomicNumber::test_bad_type 26 as int");
eval {
	xraylib::SymbolToAtomicNumber(undef);
};
like($@, qr/^ValueError/ , "TestSymbolToAtomicNumber::test_bad_type undef");

ok(xraylib::AtomicNumberToSymbol(26) eq "Fe", "TestAtomicNumberToSymbol::test_Fe");
eval {
	xraylib::AtomicNumberToSymbol(-2);
};
like($@, qr/^ValueError/ , "TestAtomicNumberToSymbol::test_bad_symbol -2");
eval {
	xraylib::AtomicNumberToSymbol(108);
};
like($@, qr/^ValueError/ , "TestAtomicNumberToSymbol::test_bad_symbol 108");
eval {
	xraylib::AtomicNumberToSymbol("Fe");
};
like($@, qr/^TypeError/ , "TestAtomicNumberToSymbol::test_bad_type Fe");
eval {
	xraylib::AtomicNumberToSymbol(undef);
};
like($@, qr/^ValueError/ , "TestAtomicNumberToSymbol::test_bad_type undef");

for my $Z (1..107) {
	ok(xraylib::SymbolToAtomicNumber(xraylib::AtomicNumberToSymbol($Z)) eq $Z, "TestCrossValidation::test $Z");
}

done_testing();
