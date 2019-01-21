program test_compoundparser;

{$APPTYPE CONSOLE}
{$mode objfpc}
{$h+}

uses xraylib, xrltest, Classes, SysUtils, fpcunit, testreport, testregistry;

type
	TestCompoundParser = class(TTestCase)
	private
		procedure _test_bad_compound(compound: string);
	published
		procedure test_good_compounds;
		procedure test_bad_compounds;
	end;

type
	TestSymbolToAtomicNumber = class(TTestCase)
	private
		procedure _test_bad_symbol;
	published
		procedure test_Fe;
		procedure test_bad_symbol;
	end;

type
	TestAtomicNumberToSymbol = class(TTestCase)
	private
		procedure _test_bad_symbol1;
		procedure _test_bad_symbol2;
	published
		procedure test_Fe;
		procedure test_bad_symbol;
	end;

type
	TestCrossValidation = class(TTestCase)
	published
		procedure test;
	end;

procedure TestCompoundParser.test_good_compounds;
begin
	CompoundParser('C19H29COOH');
	CompoundParser('C12H10');
	CompoundParser('C12H6O2');
	CompoundParser('C6H5Br');
	CompoundParser('C3H4OH(COOH)3');
	CompoundParser('HOCH2CH2OH');
	CompoundParser('C5H11NO2');
	CompoundParser('CH3CH(CH3)CH3');
	CompoundParser('NH2CH(C4H5N2)COOH');
	CompoundParser('H2O');
	CompoundParser('Ca5(PO4)3F');
	CompoundParser('Ca5(PO4)3OH');
	CompoundParser('Ca5.522(PO4.48)3OH');
	CompoundParser('Ca5.522(PO.448)3OH');
end;

procedure TestCompoundParser.test_bad_compounds;
begin
	_test_bad_compound('0C');
	_test_bad_compound('2O');
	_test_bad_compound('13Li');
	_test_bad_compound('2(NO3)');
	_test_bad_compound('H(2)');
	_test_bad_compound('Ba(12)');
	_test_bad_compound('Cr(5)3');
	_test_bad_compound('Pb(13)2');
	_test_bad_compound('Au(22)11');
	_test_bad_compound('Au11(H3PO4)2)');
	_test_bad_compound('Au11(H3PO4))2');
	_test_bad_compound('Au(11(H3PO4))2');
	_test_bad_compound('Ca5.522(PO.44.8)3OH');
	_test_bad_compound('Ba[12]');
	_test_bad_compound('Auu1');
	_test_bad_compound('AuL1');
	_test_bad_compound('  ');
	_test_bad_compound('\t');
	_test_bad_compound('\n');
	_test_bad_compound('Au L1');
	_test_bad_compound('Au\tFe');
	_test_bad_compound('CuI2ww');
end;

procedure TestCompoundParser._test_bad_compound(compound: string);
begin
	try
		CompoundParser(compound);
	except
		on E: EArgumentException do
			begin
				exit;
			end;
	end;
	Fail('Expected exception was not raised for ' + compound);
end;

procedure TestSymbolToAtomicNumber.test_Fe;
begin
	AssertEquals(SymbolToAtomicNumber('Fe'), 26);
end;

procedure TestSymbolToAtomicNumber._test_bad_symbol;
begin
	SymbolToAtomicNumber('Uu');
end;

procedure TestSymbolToAtomicNumber.test_bad_symbol;
begin
	AssertException(EArgumentException, @_test_bad_symbol);
end;

procedure TestAtomicNumberToSymbol.test_Fe;
begin
	AssertEquals(AtomicNumberToSymbol(26), 'Fe');
end;

procedure TestAtomicNumberToSymbol._test_bad_symbol1;
begin
	AtomicNumberToSymbol(-2);
end;

procedure TestAtomicNumberToSymbol._test_bad_symbol2;
begin
	AtomicNumberToSymbol(108);
end;

procedure TestAtomicNumberToSymbol.test_bad_symbol;
begin
	AssertException(EArgumentException, @_test_bad_symbol1);
	AssertException(EArgumentException, @_test_bad_symbol2);
end;

procedure TestCrossValidation.test;
var
	Z: integer;
begin
	for Z := 1 to 107 do
	begin
		AssertEquals(SymbolToAtomicNumber(AtomicNumberToSymbol(Z)), Z);
	end;
end;

var
	App: TestRunner;
begin
	RegisterTest(TestCompoundParser);
	RegisterTest(TestSymbolToAtomicNumber);
	RegisterTest(TestAtomicNumberToSymbol);
	RegisterTest(TestCrossValidation);
	App := TestRunner.Create(nil);
	App.Initialize;
	App.Run;
	App.Free;
end.
