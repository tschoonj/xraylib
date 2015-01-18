program xraylibtest;
{$linklib libxrl}
uses xraylib;
var
	cdtest : PcompoundData;
	mystring : string;
	i : longint;
	cdn : PcompoundDataNIST;
	nistCompounds : stringArray;
	rnd : PradioNuclideData;
	radioNuclides : stringArray;
	cryst : PCrystal_Struct;
	atom : PCrystal_Atom;
	energy : double = 8.0;
	debye_temp_factor : double = 1.0;
	rel_angle : double = 1.0;
	bragg, q, dw : double;
	f0 , fp, fpp : double;
	FbigH, Fbig0, FbigHbar : xrlComplex;

begin

	SetErrorMessages(0);

        writeln('Example of Pascal program using xraylib');
  	writeln('Density of pure Al: ', ElementDensity(13),' g/cm3');
  	writeln('Ca K-alpha Fluorescence Line Energy: ', LineEnergy(20,KA_LINE));
	writeln('Fe partial photoionization cs of L3 at 6.0 keV: ',CS_Photo_Partial(26,L3_SHELL,6.0));
	writeln('Zr L1 edge energy: ',EdgeEnergy(40,L1_SHELL));
	writeln('Pb Lalpha XRF production cs at 20.0 keV (jump approx): ',CS_FluorLine(82,LA_LINE,20.0));
	writeln('Pb Lalpha XRF production cs at 20.0 keV (Kissel): ',CS_FluorLine_Kissel(82,LA_LINE,20.0));
	writeln('Bi M1N2 radiative rate: ',RadRate(83,M1N2_LINE));
	writeln('U M3O3 Fluorescence Line Energy: ',LineEnergy(92,M3O3_LINE));
	{parser test for Ca(HCO3)2 (calcium bicarbonate)}
	cdtest := CompoundParser('Ca(HCO3)2');
	if (cdtest = nil) then
	begin
		Halt(1)
	end;
  	writeln('Ca(HCO3)2 contains ',cdtest^.nAtomsAll, ' atoms and ', cdtest^.nElements,' elements');
	for  i := 0 to cdtest^.nElements-1 do
	begin
		writeln('Element ', cdtest^.Elements[i], ' : ', cdtest^.massFractions[i]*100.0, ' %');
	end;
	Dispose(cdtest);


	{parser test for SiO2 (quartz)}
	cdtest := CompoundParser('SiO2');
	if (cdtest = nil) then
	begin
		Halt(1)
	end;
  	writeln('SiO2 contains ',cdtest^.nAtomsAll, ' atoms and ', cdtest^.nElements,' elements');
	for  i := 0 to cdtest^.nElements-1 do
	begin
		writeln('Element ', cdtest^.Elements[i], ' : ', cdtest^.massFractions[i]*100.0, ' %');
	end;
	Dispose(cdtest);
	writeln('Ca(HCO3)2 Rayleigh cs at 10.0 keV: ',CS_Rayl_CP('Ca(HCO3)2',10.0));

	writeln('CS2 Refractive Index at 10.0 keV : ',Refractive_Index_Re('CS2',10.0,1.261),' - ',Refractive_Index_Im('CS2', 10.0, 1.261), ' i');
	writeln('C16H14O3 Refractive Index at 1 keV : ',Refractive_Index_Re('C16H14O3', 1.0, 1.2),' - ',Refractive_Index_Im('C16H14O3', 1.0, 1.2),' i');
	writeln('SiO2 Refractive Index at 5 keV : ',Refractive_Index_Re('SiO2', 5.0, 2.65),' - ',Refractive_Index_Im('SiO2',5.0, 2.65),' i');

	writeln('Compton profile for Fe at pz = 1.1 : ',ComptonProfile(26,1.1));
	writeln('M5 Compton profile for Fe at pz = 1.1 : ',ComptonProfile_Partial(26,M5_SHELL,1.1));
	writeln('M1->M5 Coster-Kronig transition probability for Au : ',CosKronTransProb(79,FM15_TRANS));
	writeln('L1->L3 Coster-Kronig transition probability for Fe : ',CosKronTransProb(26,FL13_TRANS));
	writeln('Au Ma1 XRF production cs at 10.0 keV (Kissel): ', CS_FluorLine_Kissel(79,MA1_LINE,10.0));
	writeln('Au Mb XRF production cs at 10.0 keV (Kissel): ', CS_FluorLine_Kissel(79,MB_LINE,10.0));
	writeln('Au Mg XRF production cs at 10.0 keV (Kissel): ', CS_FluorLine_Kissel(79,MG_LINE,10.0));

	writeln('K atomic level width for Fe: ', AtomicLevelWidth(26,K_SHELL));
	writeln('Bi L2-M5M5 Auger non-radiative rate: ',AugerRate(86,L2_M5M5_AUGER));
	writeln('Bi L3 Auger yield: ', AugerYield(86, L3_SHELL));
	writeln('Symbol of element 26 is: ',AtomicNumberToSymbol(26));
	writeln('Number of element Fe is: ',SymbolToAtomicNumber('Fe'));

	writeln('Pb Malpha XRF production cs at 20.0 keV with cascade effect: ',CS_FluorLine_Kissel(82,MA1_LINE,20.0));
	writeln('Pb Malpha XRF production cs at 20.0 keV with radiative cascade effect: ',CS_FluorLine_Kissel_Radiative_Cascade(82,MA1_LINE,20.0));
	writeln('Pb Malpha XRF production cs at 20.0 keV with non-radiative cascade effect: ',CS_FluorLine_Kissel_Nonradiative_Cascade(82,MA1_LINE,20.0));
	writeln('Pb Malpha XRF production cs at 20.0 keV without cascade effect: ',CS_FluorLine_Kissel_no_Cascade(82,MA1_LINE,20.0));

	writeln('Al mass energy-absorption cs at 20.0 keV: ', CS_Energy(13, 20.0));
	writeln('Pb mass energy-absorption cs at 40.0 keV: ', CS_Energy(82, 40.0));
	mystring := 'CdTe';
	writeln('CdTe mass energy-absorption cs at 40.0 keV: ', CS_Energy_CP(mystring, 40.0));

  	{ Si Crystal structure }

  	cryst := Crystal_GetCrystal('Si');
  	if (cryst = nil) then
	begin
		Halt(1)
	end;
  	writeln('Si unit cell dimensions are ', cryst^.a, cryst^.b, cryst^.c);
  	writeln('Si unit cell angles are ', cryst^.alpha, cryst^.beta, cryst^.gamma);
  	writeln('Si unit cell volume is ', cryst^.volume);
  	writeln('Si atoms at:');
  	writeln('   Z  fraction    X        Y        Z');
	for i := 0 to cryst^.n_atom-1 do
	begin
    		atom := @cryst^.atom[i];
    		writeln(atom^.Zatom, atom^.fraction, atom^.x, atom^.y, atom^.z);
	end;

  	{ Si diffraction parameters }
	writeln('');
  	writeln('Si111 at 8 KeV. Incidence at the Bragg angle:');

  	bragg := Bragg_angle(cryst, energy, 1, 1, 1);
  	writeln('  Bragg angle: Rad: ', bragg, ' Deg: ', bragg*180/PI);

  	q := Q_scattering_amplitude (cryst, energy, 1, 1, 1, rel_angle);
  	writeln('  Q Scattering amplitude: ', q);

  	Atomic_Factors (14, energy, q, debye_temp_factor, @f0, @fp, @fpp);
  	writeln('  Atomic factors (Z = 14) f0, fp, fpp: ' , f0, ', ', fp, ', i*', fpp);

  	FbigH := Crystal_F_H_StructureFactor (cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle);
  	writeln('  FH(1,1,1) structure factor: (', FbigH.re, ', ', FbigH.im, ')');

  	Fbig0 := Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
  	writeln('  F0=FH(0,0,0) structure factor: (', Fbig0.re, ', ', Fbig0.im, ')');

 	{ Diamond diffraction parameters }

  	cryst := Crystal_GetCrystal('Diamond');
	writeln('');
  	writeln('Diamond 111 at 8 KeV. Incidence at the Bragg angle:');

  	bragg := Bragg_angle (cryst, energy, 1, 1, 1);
  	writeln('  Bragg angle: Rad: ', bragg, ' Deg: ', bragg*180/PI);

  	q := Q_scattering_amplitude (cryst, energy, 1, 1, 1, rel_angle);
  	writeln('  Q Scattering amplitude: ', q);

  	Atomic_Factors (6, energy, q, debye_temp_factor, @f0, @fp, @fpp);
  	writeln('  Atomic factors (Z = 6) f0, fp, fpp: ' , f0, ', ', fp, ', i*', fpp);

  	FbigH := Crystal_F_H_StructureFactor (cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle);
  	writeln('  FH(1,1,1) structure factor: (', FbigH.re, ', ', FbigH.im, ')');

  	Fbig0 := Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
  	writeln('  F0=FH(0,0,0) structure factor: (', Fbig0.re, ', ', Fbig0.im, ')');

  	FbigHbar := Crystal_F_H_StructureFactor (cryst, energy, -1, -1, -1, debye_temp_factor, rel_angle);
  	dw := 1E10 * 2 * (R_E / cryst^.volume) * (KEV2ANGST * KEV2ANGST/ (energy * energy)) * 
                                                  sqrt(c_abs(c_mul(FbigH, FbigHbar))) / PI / sin(2*bragg);
  	writeln('  Darwin width: ', 1e6*dw,' micro-radians');

  	{ Alpha Quartz diffraction parameters }

  	cryst := Crystal_GetCrystal('AlphaQuartz');
	writeln('');
  	writeln('Alpha Quartz 020 at 8 KeV. Incidence at the Bragg angle:');

  	bragg := Bragg_angle (cryst, energy, 0, 2, 0);
  	writeln('  Bragg angle: Rad: ', bragg, ' Deg: ', bragg*180/PI);

	q := Q_scattering_amplitude (cryst, energy, 0, 2, 0, rel_angle);
  	writeln('  Q Scattering amplitude: ', q);

  	Atomic_Factors (8, energy, q, debye_temp_factor, @f0, @fp, @fpp);
  	writeln('  Atomic factors (Z = 8) f0, fp, fpp: ' , f0, ', ', fp, ', i*', fpp);

  	FbigH := Crystal_F_H_StructureFactor (cryst, energy, 0, 2, 0, debye_temp_factor, rel_angle);
  	writeln('  FH(0,2,0) structure factor: (', FbigH.re, ', ', FbigH.im, ')');

  	Fbig0 := Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
  	writeln('  F0=FH(0,0,0) structure factor: (', Fbig0.re, ', ', Fbig0.im, ')');

  	{ Muscovite diffraction parameters }

  	cryst := Crystal_GetCrystal('Muscovite');
	writeln('');
  	writeln('Muscovite 331 at 8 KeV. Incidence at the Bragg angle:');

  	bragg := Bragg_angle (cryst, energy, 3, 3, 1);
  	writeln('  Bragg angle: Rad: ', bragg, ' Deg: ', bragg*180/PI);

  	q := Q_scattering_amplitude (cryst, energy, 3, 3, 1, rel_angle);
  	writeln('  Q Scattering amplitude: ', q);

  	Atomic_Factors (19, energy, q, debye_temp_factor, @f0, @fp, @fpp);
  	writeln('  Atomic factors (Z = 19) f0, fp, fpp: ' , f0, ', ', fp, ', i*', fpp);

  	FbigH := Crystal_F_H_StructureFactor (cryst, energy, 3, 3, 1, debye_temp_factor, rel_angle);
  	writeln('  FH(3,3,1) structure factor: (', FbigH.re, ', ', FbigH.im, ')');

  	Fbig0 := Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
  	writeln('  F0=FH(0,0,0) structure factor: (', Fbig0.re, ', ', Fbig0.im, ')');

	writeln('');

	{ compoundDataNIST tests }
	cdn := GetCompoundDataNISTByName('Uranium Monocarbide');
	writeln('Uranium Monocarbide');
	if (cdn = nil) then
	begin
		Halt(1)
	end;
	writeln('  Name: ', cdn^.name);
  	writeln('  Density: ',cdn^.density ,' g/cm3');
	for  i := 0 to cdn^.nElements-1 do
	begin
		writeln('  Element ', cdn^.Elements[i], ' : ', cdn^.massFractions[i]*100.0, ' %');
	end;

	Dispose(cdn);

	cdn := GetCompoundDataNISTByIndex(NIST_COMPOUND_BRAIN_ICRP);
	writeln('NIST_COMPOUND_BRAIN_ICRP');
	writeln('  Name: ', cdn^.name);
  	writeln('  Density: ',cdn^.density ,' g/cm3');
	for  i := 0 to cdn^.nElements-1 do
	begin
		writeln('  Element ', cdn^.Elements[i], ' : ', cdn^.massFractions[i]*100.0, ' %');
	end;

	Dispose(cdn);

	nistCompounds := GetCompoundDataNISTList();
	writeln('List of available NIST compounds:');
	for  i := 0 to Length(nistCompounds)-1 do
	begin
  		writeln('  Compound ',i,': ', nistCompounds[i]);
	end;
 
	writeln('');

  	{ radioNuclideData tests }
	rnd := GetRadioNuclideDataByName('109Cd');
	writeln('109Cd');
	writeln('  Name: ', rnd^.name);
	writeln('  Z: ', rnd^.Z);
	writeln('  A: ', rnd^.A);
	writeln('  N: ', rnd^.N);
	writeln('  Z_xray: ', rnd^.Z_xray);
	writeln('  X-rays:');
	for  i := 0 to rnd^.nXrays-1 do
	begin
		writeln('  ', LineEnergy(rnd^.Z_xray, rnd^.XrayLines[i]), ' keV -> ', rnd^.XrayIntensities[i]);
	end;
	writeln('  Gamma rays:');
	for  i := 0 to rnd^.nGammas-1 do
	begin
		writeln('  ', rnd^.GammaEnergies[i], ' keV -> ', rnd^.GammaIntensities[i]);
	end;

	Dispose(rnd);

	rnd := GetRadioNuclideDataByIndex(RADIO_NUCLIDE_125I);
	writeln('RADIO_NUCLIDE_125I');
	writeln('  Name: ', rnd^.name);
	writeln('  Z: ', rnd^.Z);
	writeln('  A: ', rnd^.A);
	writeln('  N: ', rnd^.N);
	writeln('  Z_xray: ', rnd^.Z_xray);
	writeln('  X-rays:');
	for  i := 0 to rnd^.nXrays-1 do
	begin
		writeln('  ', LineEnergy(rnd^.Z_xray, rnd^.XrayLines[i]), ' keV -> ', rnd^.XrayIntensities[i]);
	end;
	writeln('  Gamma rays:');
	for  i := 0 to rnd^.nGammas-1 do
	begin
		writeln('  ', rnd^.GammaEnergies[i], ' keV -> ', rnd^.GammaIntensities[i]);
	end;

	Dispose(rnd);

	radioNuclides := GetRadioNuclideDataList();
	writeln('List of available radionuclides:');
	for  i := 0 to Length(radioNuclides)-1 do
	begin
  		writeln('  Radionuclide ',i,': ', radioNuclides[i]);
	end;
 



	writeln('');
	writeln('--------------------------- END OF XRLEXAMPLE14 -------------------------------');
	writeln('');
end.
