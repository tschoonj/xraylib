{
Copyright (c) 2015-2019, Tom Schoonjans & Matthew Wormington
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
}

{$IFDEF FPC}
{$MODE DELPHI}
{$ENDIF}

unit xraylib;

interface

uses
  SysUtils;

const
  {$IFDEF MACOS}
    {$DEFINE DARWIN}
  {$ENDIF}
  {$IFDEF DARWIN}
    External_library = 'libxrl.11.dylib'; {Setup as you need}
  {$ENDIF}
  {$IFDEF LINUX}
    External_library = 'libxrl.so.11'; {Setup as you need}
  {$ENDIF}
  {$IFDEF MSWINDOWS}
    External_library = 'libxrl-11.dll'; {Setup as you need}
  {$ENDIF}

{$IFNDEF windows}
  {$LINKLIB libxrl}
{$ENDIF windows}

{$IFDEF FPC}
  {$PACKRECORDS C}
{$ENDIF}

// From xraylib.h
const
  XRAYLIB_MAJOR = 4;
  XRAYLIB_MINOR = 1;
  XRAYLIB_MICRO = 1;

procedure XRayInit;cdecl;external External_library name 'XRayInit';

procedure SetHardExit(hard_exit:longint);cdecl;external External_library name 'SetHardExit';

procedure SetExitStatus(exit_status:longint);cdecl;external External_library name 'SetExitStatus';

function GetExitStatus:longint;cdecl;external External_library name 'GetExitStatus';

procedure SetErrorMessages(status:longint);cdecl;external External_library name 'SetErrorMessages';

function GetErrorMessages:longint;cdecl;external External_library name 'GetErrorMessages';

{$I xraylib_const.pas }
{$I xraylib_iface.pas }

type
  xrlComplex = record
    re : double;
    im : double;
  end;

function c_abs(x:xrlComplex):double;cdecl;external External_library name 'c_abs';

function c_mul(x:xrlComplex; y:xrlComplex):xrlComplex;

function Refractive_Index(compound:string; E:double; density:double):xrlComplex;

type
  PCrystalAtom = ^TCrystalAtom;
  TCrystalAtom = record
    Zatom : longint;
    fraction : double;
    x : double;
    y : double;
    z : double;
  end;

  PCrystalStruct  = ^TCrystalStruct;
  TCrystalStruct = record
    name : PAnsiChar;
    a : double;
    b : double;
    c : double;
    alpha : double;
    beta : double;
    gamma : double;
    volume : double;
    n_atom : longint;
    atom : array of TCrystalAtom;
  end;

  PCompoundData  = ^TCompoundData;
  TCompoundData = record
    nElements : longint;
    nAtomsAll : double;
    Elements : array of longint;
    massFractions : array of double;
    nAtoms : array of double;
    molarMass : double;
  end;

  PCompoundDataNIST  = ^TCompoundDataNIST;
  TCompoundDataNIST = record
    name : PAnsiChar;
    nElements : longint;
    Elements : array of longint;
    massFractions : array of double;
    density : double;
  end;

  TStringArray = array of string;

  PRadioNuclideData  = ^TRadioNuclideData;
  TRadioNuclideData = record
    name : PAnsiChar;
    Z : longint;
    A : longint;
    N : longint;
    Z_xray : longint;
    nXrays : longint;
    XrayLines : array of longint;
    XrayIntensities : array of double;
    nGammas : longint;
    GammaEnergies : array of double;
    GammaIntensities : array of double;
  end;

  function AtomicNumberToSymbol(Z:longint):string;
  function SymbolToAtomicNumber(symbol:string):longint;
  function CompoundParser(compound:string):PCompoundData;
  procedure FreeCompoundData(data:PCompoundData);cdecl;external External_library name 'FreeCompoundData';

  function GetCompoundDataNISTByName(compoundString:string):PCompoundDataNIST;
  function GetCompoundDataNISTByIndex(compoundIndex:longint):PCompoundDataNIST;cdecl;external External_library name 'GetCompoundDataNISTByIndex';
  procedure FreeCompoundDataNIST(compoundData:PCompoundDataNIST);cdecl;external External_library name 'FreeCompoundDataNIST';
  function GetCompoundDataNISTList:TStringArray;

  function GetRadioNuclideDataByName(radioNuclideString:string):PRadioNuclideData;
  function GetRadioNuclideDataByIndex(radioNuclideIndex:longint):PRadioNuclideData;cdecl;external External_library name 'GetRadioNuclideDataByIndex';
  function GetRadioNuclideDataList():TStringArray;
  procedure FreeRadioNuclideData(rnd:PRadioNuclideData);cdecl;external External_library name 'FreeRadioNuclideData';

  function Crystal_GetCrystal(material:string):PCrystalStruct;
  procedure Crystal_Free(cryst:PCrystalStruct);cdecl;external External_library name 'Crystal_Free';
  procedure Atomic_Factors(Z:longint; energy:double; q:double; debye_factor:double; var f0, f_primep, f_prime2:double);cdecl;external External_library name 'Atomic_Factors';
  function Bragg_angle(crystal:PCrystalStruct; energy:double; i_miller:longint; j_miller:longint; k_miller:longint):double;cdecl;external External_library name 'Bragg_angle';
  function Q_scattering_amplitude(crystal:PCrystalStruct; energy:double; i_miller:longint; j_miller:longint; k_miller:longint;rel_angle:double):double;cdecl;external External_library name 'Q_scattering_amplitude';
  function Crystal_F_H_StructureFactor(crystal:PCrystalStruct; energy:double; i_miller:longint; j_miller:longint; k_miller:longint;debye_factor:double; rel_angle:double):xrlComplex;
  function Crystal_F_H_StructureFactor_Partial(crystal:PCrystalStruct; energy:double; i_miller:longint; j_miller:longint; k_miller:longint;debye_factor:double; rel_angle:double; f0_flag:longint; f_prime_flag:longint; f_prime2_flag:longint):xrlComplex;
  function Crystal_UnitCellVolume(crystal:PCrystalStruct):double;cdecl;external External_library name 'Crystal_UnitCellVolume';
  function Crystal_dSpacing(crystal:PCrystalStruct; i_miller:longint; j_miller:longint; k_miller:longint):double;cdecl;external External_library name 'Crystal_dSpacing';
  function Crystal_GetCrystalsList():TStringArray;

implementation

type
xrl_error_code = (XRL_ERROR_MEMORY, XRL_ERROR_INVALID_ARGUMENT,
	XRL_ERROR_IO, XRL_ERROR_TYPE, XRL_ERROR_UNSUPPORTED, XRL_ERROR_RUNTIME);

type
  PPxrl_error = ^Pxrl_error;
  Pxrl_error = ^xrl_error;
  xrl_error = record
    code : xrl_error_code;
    message : PAnsiChar;
  end;

procedure xrl_error_free(error:Pxrl_error);cdecl;external External_library name 'xrl_error_free';

procedure process_error(error:Pxrl_error);
var
  msg: string;
  code: xrl_error_code;
begin
  if (error = nil) then
  begin
	  Exit;
  end;

  msg := string(error.message);
  code := error.code;
  xrl_error_free(error);

  case code of
    XRL_ERROR_MEMORY: raise EHeapException.create(msg);
    XRL_ERROR_INVALID_ARGUMENT: raise EArgumentException.create(msg);
    XRL_ERROR_IO: raise EInOutError.create(msg);
    XRL_ERROR_TYPE: raise EConvertError.create(msg);
    XRL_ERROR_UNSUPPORTED: raise ENotImplemented.create(msg);
    XRL_ERROR_RUNTIME: raise Exception.create(msg);
  else
    raise Exception.create('Unknown exception type raised!');
  end;
end;

{$I xraylib_impl.pas }

function CompoundParser_C(compoundString:PAnsiChar; error:PPxrl_error):PCompoundData;cdecl;external External_library name 'CompoundParser';

function AtomicNumberToSymbol_C(Z:longint; error:PPxrl_error):PAnsiChar;cdecl;external External_library name 'AtomicNumberToSymbol';

function SymbolToAtomicNumber_C(symbol:PAnsiChar; error:PPxrl_error):longint;cdecl;external External_library name 'SymbolToAtomicNumber';

procedure xrlFree(p:pointer);cdecl;external External_library name 'xrlFree';

function AtomicNumberToSymbol(Z:longint):string;
var
  error: Pxrl_error;
  symbol: PAnsiChar;
  rv : string;
begin
  error := nil;
  symbol := AtomicNumberToSymbol_C(Z, @error);
  process_error(error);
  rv := string(PAnsiChar(symbol));
  xrlFree(symbol);
  Result := rv
end;

function SymbolToAtomicNumber(symbol:string):longint;
var
  error: Pxrl_error;
  temp: PAnsiChar;
begin
  error := nil;
  temp := PAnsiChar(AnsiString(symbol));
  Result := SymbolToAtomicNumber_C(temp, @error);
  process_error(error);
end;

function CompoundParser(compound:string):PCompoundData;
var
  error: Pxrl_error;
  temp: PAnsiChar;
begin
  error := nil;
  temp := PAnsiChar(AnsiString(compound));
  Result := CompoundParser_C(temp, @error);
  process_error(error);
end;

function Refractive_Index_C(compound:PAnsiChar; E:double; density:double; error:PPxrl_error):xrlComplex;cdecl;external External_library name 'Refractive_Index';

function Refractive_Index(compound:string; E:double; density:double):xrlComplex;
var
  error: Pxrl_error;
  temp:PAnsiChar;
begin
  error := nil;
  temp := PAnsiChar(AnsiString(compound));
  result := Refractive_Index_C(temp, E, density, @error);
  process_error(error);
end;

function GetCompoundDataNISTList_C(var n:longint; error:PPxrl_error): PPAnsiChar;cdecl;external External_library name 'GetCompoundDataNISTList';

function GetCompoundDataNISTByName_C(compoundString:PAnsiChar; error:PPxrl_error):PCompoundDataNIST;cdecl;external External_library name 'GetCompoundDataNISTByName';

function GetCompoundDataNISTByName(compoundString:string):PCompoundDataNIST;
var
  error: Pxrl_error;
  temp: PAnsiChar;
begin
  error := nil;
  temp := PAnsiChar(AnsiString(compoundString));
  Result := GetCompoundDataNISTByName_C(temp, @error);
  process_error(error);
end;

function GetCompoundDataNISTList(): TStringArray;
var
  error: Pxrl_error;
  list_C, temp: PPAnsiChar;
  nCompounds:longint;
  i:longint;
begin
  error := nil;
  list_C := GetCompoundDataNISTList_C(nCompounds, @error);
  process_error(error);
  SetLength(Result, nCompounds);
  temp := list_C;

  for i := 0 to nCompounds-1 do
  begin
    Result[i] := string(PAnsiChar(temp^));
    xrlFree(temp^);
    inc(temp);
  end;
  xrlFree(list_C);
end;

function GetRadioNuclideDataByName_C(radioNuclideString:PAnsiChar; error:PPxrl_error):PRadioNuclideData;cdecl;external External_library name 'GetRadioNuclideDataByName';

function GetRadioNuclideDataList_C(var nRadioNuclides:longint; error:PPxrl_error):PPAnsiChar;cdecl;external External_library name 'GetRadioNuclideDataList';

function GetRadioNuclideDataByName(radioNuclideString:string):PRadioNuclideData;
var
  error: Pxrl_error;
  temp: PAnsiChar;
begin
  error := nil;
  temp := PAnsiChar(AnsiString(radioNuclideString));
  Result := GetRadioNuclideDataByName_C(temp, @error);
  process_error(error);
end;

function GetRadioNuclideDataList():TStringArray;
var
  error: Pxrl_error;
  list_C, temp: PPAnsiChar;
  nRadioNuclides:longint;
  i:longint;
begin
  error := nil;
  list_C := GetRadioNuclideDataList_C(nRadioNuclides, @error);
  process_error(error);
  SetLength(Result, nRadioNuclides);
  temp := list_C;

  for i := 0 to nRadioNuclides-1 do
  begin
    Result[i] := string(PAnsiChar(temp^));
    xrlFree(temp^);
    inc(temp);
  end;
  xrlFree(list_C);
end;

function Crystal_GetCrystal_C(material:PAnsiChar; p:Pointer; error:PPxrl_error):PCrystalStruct;cdecl;external External_library name 'Crystal_GetCrystal';
function Crystal_GetCrystalsList_C(c_array:Pointer; var nCrystals:longint; error:PPxrl_error):PPAnsiChar;cdecl;external External_library name 'Crystal_GetCrystalsList';

function Crystal_GetCrystalsList():TStringArray;
var
  error: Pxrl_error;
  list_C, temp: PPAnsiChar;
  nCrystals:longint;
  i:longint;
begin
  error := nil;
  list_C := Crystal_GetCrystalsList_C(nil, nCrystals, @error);
  process_error(error);
  SetLength(Result, nCrystals);
  temp := list_C;

  for i := 0 to nCrystals-1 do
  begin
    Result[i] := string(PAnsiChar(temp^));
    xrlFree(temp^);
    inc(temp);
  end;
  xrlFree(list_C);
end;

function Crystal_F_H_StructureFactor_C(crystal:PCrystalStruct; energy:double; i_miller:longint; j_miller:longint; k_miller:longint;debye_factor:double; rel_angle:double; error:PPxrl_error):xrlComplex;cdecl;external External_library name 'Crystal_F_H_StructureFactor';

function Crystal_F_H_StructureFactor_Partial_C(crystal:PCrystalStruct; energy:double; i_miller:longint; j_miller:longint; k_miller:longint;debye_factor:double; rel_angle:double; f0_flag:longint; f_prime_flag:longint; f_prime2_flag:longint; error:PPxrl_error):xrlComplex;cdecl;external External_library name 'Crystal_F_H_StructureFactor_Partial';

function Crystal_GetCrystal(material:string):PCrystalStruct;
var
  error: Pxrl_error;
  temp: PAnsiChar;
begin
  error := nil;
  temp := PAnsiChar(AnsiString(material));
  Result := Crystal_GetCrystal_C(temp, nil, @error);
  process_error(error);
end;

function Crystal_F_H_StructureFactor(crystal:PCrystalStruct; energy:double; i_miller:longint; j_miller:longint; k_miller:longint;debye_factor:double; rel_angle:double):xrlComplex;
var
  error: Pxrl_error;
begin
  error := nil;
  Result := Crystal_F_H_StructureFactor_C(crystal, energy, i_miller, j_miller, k_miller, debye_factor, rel_angle, @error);
  process_error(error);
end;

function Crystal_F_H_StructureFactor_Partial(crystal:PCrystalStruct; energy:double; i_miller:longint; j_miller:longint; k_miller:longint;debye_factor:double; rel_angle:double; f0_flag:longint; f_prime_flag:longint; f_prime2_flag:longint):xrlComplex;
var
  error: Pxrl_error;
begin
  error := nil;
  Result := Crystal_F_H_StructureFactor_Partial_C(crystal, energy, i_miller, j_miller, k_miller, debye_factor, rel_angle, f0_flag, f_prime_flag, f_prime2_flag, @error);
  process_error(error);
end;

function c_mul(x:xrlComplex; y:xrlComplex):xrlComplex;
begin
  Result.re := x.re*y.re - x.im*y.im;
  Result.im := x.re*y.im + x.im*y.re;
end;

end.
