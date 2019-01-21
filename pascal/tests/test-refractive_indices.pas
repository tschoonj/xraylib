program test_refractive_indices;

{$APPTYPE CONSOLE}
{$mode objfpc}
{$h+}

uses xraylib, xrltest, Classes, SysUtils, fpcunit, testreport, testregistry;

type
	TestRefractiveIndices = class(TTestCase)
	private
		procedure _test_bad_input_0;
		procedure _test_bad_input_1;
		procedure _test_bad_input_2;
		procedure _test_bad_input_3;
		procedure _test_bad_input_4;
		procedure _test_bad_input_5;
		procedure _test_bad_input_6;
		procedure _test_bad_input_7;
		procedure _test_bad_input_8;
	published
		procedure test_chemical_formulas;
		procedure test_nist_compounds;
		procedure test_bad_input;
	end;

procedure TestRefractiveIndices.test_chemical_formulas;
var
	cmplx: xrlComplex;
begin
	AssertEquals(Refractive_Index_Re('H2O', 1.0, 1.0), 0.999763450676632, 1.0e-9);
	AssertEquals(Refractive_Index_Im('H2O', 1.0, 1.0), 4.021660592312145e-05, 1.0e-9);
	cmplx := Refractive_Index('H2O', 1.0, 1.0);
	AssertEquals(cmplx.re, 0.999763450676632, 1.0e-9);
	AssertEquals(cmplx.im, 4.021660592312145e-05, 1.0e-9);
end;

procedure TestRefractiveIndices.test_nist_compounds;
var
	cmplx: xrlComplex;
begin
	AssertEquals(Refractive_Index_Re('Air, Dry (near sea level)', 1.0, 1.0), 0.999782559048, 1.0E-12);
	AssertEquals(Refractive_Index_Im('Air, Dry (near sea level)', 1.0, 1.0), 0.000035578193, 1.0E-12);
	cmplx := Refractive_Index('Air, Dry (near sea level)', 1.0, 1.0);
	AssertEquals(cmplx.re, 0.999782559048, 1.0E-12);
	AssertEquals(cmplx.im, 0.000035578193, 1.0E-12);

	AssertEquals(Refractive_Index_Re('Air, Dry (near sea level)', 1.0, 0.0), 0.999999737984, 1.0E-12);
	AssertEquals(Refractive_Index_Im('Air, Dry (near sea level)', 1.0, 0.0), 0.000000042872, 1.0E-12);
	cmplx := Refractive_Index('Air, Dry (near sea level)', 1.0, 0.0);
	AssertEquals(cmplx.re, 0.999999737984, 1.0E-12);
	AssertEquals(cmplx.im, 0.000000042872, 1.0E-12);


	cmplx := Refractive_Index('Air, Dry (near sea level)', 1.0, -1.0);
	AssertEquals(cmplx.re, 0.999999737984, 1.0E-12);
	AssertEquals(cmplx.im, 0.000000042872, 1.0E-12);

end;

procedure TestRefractiveIndices.test_bad_input;
begin
	AssertException(EArgumentException, @_test_bad_input_0);
	AssertException(EArgumentException, @_test_bad_input_1);
	AssertException(EArgumentException, @_test_bad_input_2);
	AssertException(EArgumentException, @_test_bad_input_3);
	AssertException(EArgumentException, @_test_bad_input_4);
	AssertException(EArgumentException, @_test_bad_input_5);
	AssertException(EArgumentException, @_test_bad_input_6);
	AssertException(EArgumentException, @_test_bad_input_7);
	AssertException(EArgumentException, @_test_bad_input_8);
end;

procedure TestRefractiveIndices._test_bad_input_0;
begin
	Refractive_Index_Re('', 1.0, 1.0);
end;
	
procedure TestRefractiveIndices._test_bad_input_1;
begin
	Refractive_Index_Im('', 1.0, 1.0);
end;

procedure TestRefractiveIndices._test_bad_input_2;
begin
	Refractive_Index('', 1.0, 1.0);
end;

procedure TestRefractiveIndices._test_bad_input_3;
begin
	Refractive_Index_Re('H2O', 0.0, 1.0);
end;
	
procedure TestRefractiveIndices._test_bad_input_4;
begin
	Refractive_Index_Im('H2O', 0.0, 1.0);
end;

procedure TestRefractiveIndices._test_bad_input_5;
begin
	Refractive_Index('H2O', 0.0, 1.0);
end;

procedure TestRefractiveIndices._test_bad_input_6;
begin
	Refractive_Index_Re('H2O', 1.0, 0.0);
end;
	
procedure TestRefractiveIndices._test_bad_input_7;
begin
	Refractive_Index_Im('H2O', 1.0, 0.0);
end;

procedure TestRefractiveIndices._test_bad_input_8;
begin
	Refractive_Index('H2O', 1.0, 0.0);
end;

var
	App: TestRunner;
begin
	RegisterTest(TestRefractiveIndices);
	App := TestRunner.Create(nil);
	App.Initialize;
	App.Run;
	App.Free;
end.
