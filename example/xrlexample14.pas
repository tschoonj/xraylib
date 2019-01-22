program xrlexample14;

{$APPTYPE CONSOLE}
uses SysUtils, xraylib;

var
  i: Integer;
  cd: PCompoundData;
  cdn: PCompoundDataNIST;
  nistCompounds: TStringArray;
  radioNuclides: TStringArray;
  crystals: TStringArray;
  cryst: PCrystalStruct;
	atom: PCrystalAtom;
	energy : double = 8.0;
	debye_temp_factor : double = 1.0;
	rel_angle : double = 1.0;
  bragg, q, dw : double;
	f0 , fp, fpp : double;
	FbigH, Fbig0, FbigHbar : xrlComplex;
  rnd: PRadioNuclideData;
begin
  try
    XrayInit();

    WriteLn(Format('XrayLib v%d.%d', [XRAYLIB_MAJOR,XRAYLIB_MINOR]));
    WriteLn('Example Delphi program using XrayLib');

    WriteLn;
    WriteLn('Atomic weight of Al: ', AtomicWeight(13),' g/mol');
    WriteLn('Density of pure Al: ', ElementDensity(13),' g/cm3');
    WriteLn('Ca K-alpha Fluorescence Line Energy: ', LineEnergy(20,KA_LINE));
    WriteLn('Fe partial photoionization cs of L3 at 6.0 keV: ', CS_Photo_Partial(26,L3_SHELL,6.0));
    WriteLn('Zr L1 edge energy: ', EdgeEnergy(40,L1_SHELL));
    WriteLn('Pb Lalpha XRF production cs at 20.0 keV (jump approx): ', CS_FluorLine(82,LA_LINE,20.0));
    WriteLn('Pb Lalpha XRF production cs at 20.0 keV (Kissel): ', CS_FluorLine_Kissel(82,LA_LINE,20.0));
    WriteLn('Bi M1N2 radiative rate: ', RadRate(83,M1N2_LINE));
    WriteLn('U M3O3 Fluorescence Line Energy: ', LineEnergy(92,M3O3_LINE));

    // Parser test for Ca(HCO3)2 (calcium bicarbonate)
    WriteLn;
    cd := CompoundParser('Ca(HCO3)2');
    WriteLn('Ca(HCO3)2 contains ', cd^.nAtomsAll, ' atoms, ', cd^.nElements,' elements and has a molar mass of ', cd^.molarMass, ' g/mol');
    for  i := 0 to cd^.nElements-1 do
      WriteLn('Element ', cd^.Elements[i], ': ', cd^.massFractions[i]*100.0, ' % and ', cd^.nAtoms[i], ' atoms');
    FreeCompoundData(cd);

    // parser test for SiO2 (quartz)
    WriteLn;
    cd := CompoundParser('SiO2');
    WriteLn('SiO2 contains ', cd^.nAtomsAll, ' atoms, ', cd^.nElements,' elements and has a molar mass of ', cd^.molarMass, ' g/mol');
    for  i := 0 to cd^.nElements-1 do
      WriteLn('Element ', cd^.Elements[i], ': ', cd^.massFractions[i]*100.0, ' % and ', cd^.nAtoms[i], ' atoms');
    FreeCompoundData(cd);

    WriteLn;
    WriteLn('Ca(HCO3)2 Rayleigh cs at 10.0 keV: ',CS_Rayl_CP('Ca(HCO3)2',10.0));
    WriteLn('CS2 Refractive Index at 10.0 keV: ',Refractive_Index_Re('CS2',10.0,1.261),' - ',Refractive_Index_Im('CS2', 10.0, 1.261), ' i');
    WriteLn('C16H14O3 Refractive Index at 1 keV: ',Refractive_Index_Re('C16H14O3', 1.0, 1.2),' - ',Refractive_Index_Im('C16H14O3', 1.0, 1.2),' i');
    WriteLn('SiO2 Refractive Index at 5 keV: ',Refractive_Index_Re('SiO2', 5.0, 2.65),' - ',Refractive_Index_Im('SiO2',5.0, 2.65),' i');

    WriteLn('Compton profile for Fe at pz = 1.1: ',ComptonProfile(26,1.1));
    WriteLn('M5 Compton profile for Fe at pz = 1.1: ',ComptonProfile_Partial(26,M5_SHELL,1.1));
    WriteLn('M1->M5 Coster-Kronig transition probability for Au: ',CosKronTransProb(79,FM15_TRANS));
    WriteLn('L1->L3 Coster-Kronig transition probability for Fe: ',CosKronTransProb(26,FL13_TRANS));
    WriteLn('Au Ma1 XRF production cs at 10.0 keV (Kissel): ', CS_FluorLine_Kissel(79,MA1_LINE,10.0));
    WriteLn('Au Mb XRF production cs at 10.0 keV (Kissel): ', CS_FluorLine_Kissel(79,MB_LINE,10.0));
    WriteLn('Au Mg XRF production cs at 10.0 keV (Kissel): ', CS_FluorLine_Kissel(79,MG_LINE,10.0));

    WriteLn('K atomic level width for Fe: ', AtomicLevelWidth(26,K_SHELL));
    WriteLn('Bi L2-M5M5 Auger non-radiative rate: ',AugerRate(86,L2_M5M5_AUGER));
    WriteLn('Bi L3 Auger yield: ', AugerYield(86, L3_SHELL));

    WriteLn('Sr anomalous scattering factor Fi at 10.0 keV: ', Fi(38, 10.0));
    WriteLn('Sr anomalous scattering factor Fii at 10.0 keV: ', Fii(38, 10.0));

    WriteLn('Symbol of element 26 is: ',AtomicNumberToSymbol(26));
    WriteLn('Number of element Fe is: ',SymbolToAtomicNumber('Fe'));

    WriteLn('Pb Malpha XRF production cs at 20.0 keV with cascade effect: ',CS_FluorLine_Kissel(82,MA1_LINE,20.0));
    WriteLn('Pb Malpha XRF production cs at 20.0 keV with radiative cascade effect: ',CS_FluorLine_Kissel_Radiative_Cascade(82,MA1_LINE,20.0));
    WriteLn('Pb Malpha XRF production cs at 20.0 keV with non-radiative cascade effect: ',CS_FluorLine_Kissel_Nonradiative_Cascade(82,MA1_LINE,20.0));
    WriteLn('Pb Malpha XRF production cs at 20.0 keV without cascade effect: ',CS_FluorLine_Kissel_no_Cascade(82,MA1_LINE,20.0));

    WriteLn('Al mass energy-absorption cs at 20.0 keV: ', CS_Energy(13, 20.0));
    WriteLn('Pb mass energy-absorption cs at 40.0 keV: ', CS_Energy(82, 40.0));
    WriteLn('CdTe mass energy-absorption cs at 40.0 keV: ', CS_Energy_CP('CdTe', 40.0));

  	// Si Crystal structure
    WriteLn;
  	cryst := Crystal_GetCrystal('Si');
  	WriteLn('Si unit cell dimensions are ', cryst^.a, cryst^.b, cryst^.c);
  	WriteLn('Si unit cell angles are ', cryst^.alpha, cryst^.beta, cryst^.gamma);
  	WriteLn('Si unit cell volume is ', cryst^.volume);
  	WriteLn('Si atoms at:');
  	WriteLn('   Z  fraction    X        Y        Z');
    for i := 0 to cryst^.n_atom-1 do
    begin
      atom := @cryst^.atom[i];
      WriteLn('   ', atom^.Zatom, atom^.fraction, atom^.x, atom^.y, atom^.z);
    end;

    // Si diffraction parameters
    WriteLn;
    WriteLn('Si111 at 8 KeV. Incidence at the Bragg angle:');

    bragg := Bragg_angle(cryst, energy, 1, 1, 1);
    WriteLn('  Bragg angle: Rad: ', bragg, ' Deg: ', bragg*180/PI);

    q := Q_scattering_amplitude (cryst, energy, 1, 1, 1, rel_angle);
    WriteLn('  Q Scattering amplitude: ', q);

    Atomic_Factors (14, energy, q, debye_temp_factor, f0, fp, fpp);
    WriteLn('  Atomic factors (Z = 14) f0, fp, fpp: ' , f0, ', ', fp, ', i*', fpp);

    FbigH := Crystal_F_H_StructureFactor (cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle);
    WriteLn('  FH(1,1,1) structure factor: (', FbigH.re, ', ', FbigH.im, ')');

    Fbig0 := Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
    WriteLn('  F0=FH(0,0,0) structure factor: (', Fbig0.re, ', ', Fbig0.im, ')');

    // Diamond diffraction parameters
    WriteLn;
    cryst := Crystal_GetCrystal('Diamond');

    WriteLn('Diamond 111 at 8 KeV. Incidence at the Bragg angle:');

    bragg := Bragg_angle (cryst, energy, 1, 1, 1);
    WriteLn('  Bragg angle: Rad: ', bragg, ' Deg: ', bragg*180/PI);

    q := Q_scattering_amplitude (cryst, energy, 1, 1, 1, rel_angle);
    WriteLn('  Q Scattering amplitude: ', q);

    Atomic_Factors (6, energy, q, debye_temp_factor, f0, fp, fpp);
    WriteLn('  Atomic factors (Z = 6) f0, fp, fpp: ' , f0, ', ', fp, ', i*', fpp);

    FbigH := Crystal_F_H_StructureFactor (cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle);
    WriteLn('  FH(1,1,1) structure factor: (', FbigH.re, ', ', FbigH.im, ')');

    Fbig0 := Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
    WriteLn('  F0=FH(0,0,0) structure factor: (', Fbig0.re, ', ', Fbig0.im, ')');

    FbigHbar := Crystal_F_H_StructureFactor (cryst, energy, -1, -1, -1, debye_temp_factor, rel_angle);
    dw := 1E10 * 2 * (R_E / cryst^.volume) * (KEV2ANGST * KEV2ANGST/ (energy * energy)) *
                                                  sqrt(c_abs(c_mul(FbigH, FbigHbar))) / PI / sin(2*bragg);
    WriteLn('  Darwin width: ', 1e6*dw,' micro-radians');

    // Alpha Quartz diffraction parameters
    WriteLn;
  	cryst := Crystal_GetCrystal('AlphaQuartz');
  	WriteLn('Alpha Quartz 020 at 8 KeV. Incidence at the Bragg angle:');

  	bragg := Bragg_angle (cryst, energy, 0, 2, 0);
  	WriteLn('  Bragg angle: Rad: ', bragg, ' Deg: ', bragg*180/PI);

	  q := Q_scattering_amplitude (cryst, energy, 0, 2, 0, rel_angle);
  	WriteLn('  Q Scattering amplitude: ', q);

  	Atomic_Factors (8, energy, q, debye_temp_factor, f0, fp, fpp);
  	WriteLn('  Atomic factors (Z = 8) f0, fp, fpp: ' , f0, ', ', fp, ', i*', fpp);

  	FbigH := Crystal_F_H_StructureFactor (cryst, energy, 0, 2, 0, debye_temp_factor, rel_angle);
  	WriteLn('  FH(0,2,0) structure factor: (', FbigH.re, ', ', FbigH.im, ')');

  	Fbig0 := Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
  	WriteLn('  F0=FH(0,0,0) structure factor: (', Fbig0.re, ', ', Fbig0.im, ')');

  	{ Muscovite diffraction parameters }
    WriteLn;
  	cryst := Crystal_GetCrystal('Muscovite');
  	WriteLn('Muscovite 331 at 8 KeV. Incidence at the Bragg angle:');

  	bragg := Bragg_angle (cryst, energy, 3, 3, 1);
  	WriteLn('  Bragg angle: Rad: ', bragg, ' Deg: ', bragg*180/PI);

  	q := Q_scattering_amplitude (cryst, energy, 3, 3, 1, rel_angle);
  	WriteLn('  Q Scattering amplitude: ', q);

  	Atomic_Factors (19, energy, q, debye_temp_factor, f0, fp, fpp);
  	WriteLn('  Atomic factors (Z = 19) f0, fp, fpp: ' , f0, ', ', fp, ', i*', fpp);

  	FbigH := Crystal_F_H_StructureFactor (cryst, energy, 3, 3, 1, debye_temp_factor, rel_angle);
  	WriteLn('  FH(3,3,1) structure factor: (', FbigH.re, ', ', FbigH.im, ')');

  	Fbig0 := Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
  	WriteLn('  F0=FH(0,0,0) structure factor: (', Fbig0.re, ', ', Fbig0.im, ')');

    crystals:= Crystal_GetCrystalsList();
    WriteLn('List of available crystals:');
    for  i := 0 to Length(crystals)-1 do
      WriteLn('  Crystal ',i,': ', crystals[i]);

    // CompoundDataNIST tests
    WriteLn;
    cdn := GetCompoundDataNISTByName('Uranium Monocarbide');
    WriteLn('Uranium Monocarbide');
    WriteLn('  Name: ', cdn^.name);
    WriteLn('  Density: ',cdn^.density ,' g/cm3');
    for  i := 0 to cdn^.nElements-1 do
      WriteLn('  Element ', cdn^.Elements[i], ': ', cdn^.massFractions[i]*100.0, ' %');
    FreeCompoundDataNIST(cdn);

    WriteLn;
    cdn := GetCompoundDataNISTByIndex(NIST_COMPOUND_BRAIN_ICRP);
    WriteLn('NIST_COMPOUND_BRAIN_ICRP');
    WriteLn('  Name: ', cdn^.name);
    WriteLn('  Density: ',cdn^.density ,' g/cm3');
    for  i := 0 to cdn^.nElements-1 do
      WriteLn('  Element ', cdn^.Elements[i], ': ', cdn^.massFractions[i]*100.0, ' %');
    FreeCompoundDataNIST(cdn);

    WriteLn;
    nistCompounds := GetCompoundDataNISTList();
    WriteLn('List of available NIST compounds:');
    for  i := 0 to Length(nistCompounds)-1 do
      WriteLn('  Compound ',i,': ', nistCompounds[i]);

    // RadioNuclideData tests
    WriteLn;
    rnd := GetRadioNuclideDataByName('109Cd');
    WriteLn('109Cd');
    WriteLn('  Name: ', rnd^.name);
    WriteLn('  Z: ', rnd^.Z);
    WriteLn('  A: ', rnd^.A);
    WriteLn('  N: ', rnd^.N);
    WriteLn('  Z_xray: ', rnd^.Z_xray);
    WriteLn('  X-rays:');
    for  i := 0 to rnd^.nXrays-1 do
      WriteLn('  ', LineEnergy(rnd^.Z_xray, rnd^.XrayLines[i]), ' keV -> ', rnd^.XrayIntensities[i]);
    WriteLn('  Gamma rays:');
    for  i := 0 to rnd^.nGammas-1 do
      WriteLn('  ', rnd^.GammaEnergies[i], ' keV -> ', rnd^.GammaIntensities[i]);
    FreeRadioNuclideData(rnd);

    WriteLn;
    rnd := GetRadioNuclideDataByIndex(RADIO_NUCLIDE_125I);
    WriteLn('RADIO_NUCLIDE_125I');
    WriteLn('  Name: ', rnd^.name);
    WriteLn('  Z: ', rnd^.Z);
    WriteLn('  A: ', rnd^.A);
    WriteLn('  N: ', rnd^.N);
    WriteLn('  Z_xray: ', rnd^.Z_xray);
    WriteLn('  X-rays:');
    for  i := 0 to rnd^.nXrays-1 do
      WriteLn('  ', LineEnergy(rnd^.Z_xray, rnd^.XrayLines[i]), ' keV -> ', rnd^.XrayIntensities[i]);
    WriteLn('  Gamma rays:');
    for  i := 0 to rnd^.nGammas-1 do
      WriteLn('  ', rnd^.GammaEnergies[i], ' keV -> ', rnd^.GammaIntensities[i]);
    FreeRadioNuclideData(rnd);

    WriteLn;
    radioNuclides := GetRadioNuclideDataList();
    WriteLn('List of available radionuclides:');
    for  i := 0 to Length(radioNuclides)-1 do
      WriteLn('  Radionuclide ',i,': ', radioNuclides[i]);

  except
    on E: Exception do
    begin
      Writeln(E.ClassName, ': ', E.Message);
      Halt(1);
    end;
  end;
end.
