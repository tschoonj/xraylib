#Copyright (c) 2012, Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

require 'xraylib'
if RUBY_VERSION < "1.9"
require 'complex'
end

printf("Example of ruby program using xraylib\n")
printf("Density of pure Al: %f g/cm3\n", Xraylib.ElementDensity(13))
printf("Ca K-alpha Fluorescence Line Energy: %f\n",
	 Xraylib.LineEnergy(20,Xraylib::KA_LINE))
printf("Fe partial photoionization cs of L3 at 6.0 keV: %f\n",Xraylib.CS_Photo_Partial(26,Xraylib::L3_SHELL,6.0))
printf("Zr L1 edge energy: %f\n",Xraylib.EdgeEnergy(40,Xraylib::L1_SHELL))
printf("Pb Lalpha XRF production cs at 20.0 keV (jump approx): %f\n",Xraylib.CS_FluorLine(82,Xraylib::LA_LINE,20.0))
printf("Pb Lalpha XRF production cs at 20.0 keV (Kissel): %f\n",Xraylib.CS_FluorLine_Kissel(82,Xraylib::LA_LINE,20.0))
printf("Bi M1N2 radiative rate: %f\n",Xraylib.RadRate(83,Xraylib::M1N2_LINE))
printf("U M3O3 Fluorescence Line Energy: %f\n",Xraylib.LineEnergy(92,Xraylib::M3O3_LINE))

cdtest = Xraylib.CompoundParser("Ca(HCO3)2")
exit(1) if not cdtest

#puts PP.pp(cdtest, "")

printf("Ca(HCO3)2 contains %g atoms, %i elements and has a molar mass of %g g/mol\n",cdtest['nAtomsAll'],cdtest['nElements'],cdtest['molarMass'])

for i in (0..cdtest['nElements']-1)
    printf("Element %i: %f %% and %g atoms\n",cdtest['Elements'][i],cdtest['massFractions'][i]*100.0, cdtest['nAtoms'][i])
end

cdtest = Xraylib.CompoundParser("SiO2")
exit(1) if not cdtest

printf("SiO2 contains %g atoms, %i elements and has a molar mass of %g g/mol\n",cdtest['nAtomsAll'],cdtest['nElements'],cdtest['molarMass'])

for i in (0..cdtest['nElements']-1)
    printf("Element %i: %f %% and %g atoms\n",cdtest['Elements'][i],cdtest['massFractions'][i]*100.0, cdtest['nAtoms'][i])
end

printf("Ca(HCO3)2 Rayleigh cs at 10.0 keV: %f\n",Xraylib.CS_Rayl_CP("Ca(HCO3)2",10.0) )
printf("CS2 Refractive Index at 10.0 keV : %g - %g i\n",Xraylib.Refractive_Index_Re("CS2",10.0,1.261),Xraylib.Refractive_Index_Im("CS2",10.0,1.261))
printf("C16H14O3 Refractive Index at 1 keV : %g - %g i\n",Xraylib.Refractive_Index_Re("C16H14O3",1.0,1.2),Xraylib.Refractive_Index_Im("C16H14O3",1.0,1.2))
printf("SiO2 Refractive Index at 5 keV : %g - %g i\n",Xraylib.Refractive_Index_Re("SiO2",5.0,2.65),Xraylib.Refractive_Index_Im("SiO2",5.0,2.65))
printf("Compton profile for Fe at pz = 1.1 : %g\n",Xraylib.ComptonProfile(26,1.1))
printf("M5 Compton profile for Fe at pz = 1.1 : %g\n",Xraylib.ComptonProfile_Partial(26,Xraylib::M5_SHELL,1.1))
printf("M1->M5 Coster-Kronig transition probability for Au : %f\n",Xraylib.CosKronTransProb(79,Xraylib::FM15_TRANS))
printf("L1->L3 Coster-Kronig transition probability for Fe : %f\n",Xraylib.CosKronTransProb(26,Xraylib::FL13_TRANS))
printf("Au Ma1 XRF production cs at 10.0 keV (Kissel): %f\n", Xraylib.CS_FluorLine_Kissel(79,Xraylib::MA1_LINE,10.0))
printf("Au Mb XRF production cs at 10.0 keV (Kissel): %f\n", Xraylib.CS_FluorLine_Kissel(79,Xraylib::MB_LINE,10.0))
printf("Au Mg XRF production cs at 10.0 keV (Kissel): %f\n", Xraylib.CS_FluorLine_Kissel(79,Xraylib::MG_LINE,10.0))

printf("K atomic level width for Fe: %g\n", Xraylib.AtomicLevelWidth(26,Xraylib::K_SHELL))
printf("Bi L2-M5M5 Auger non-radiative rate: %g\n",Xraylib.AugerRate(86,Xraylib::L2_M5M5_AUGER))
printf("Bi L3 Auger yield: %f\n", Xraylib.AugerYield(86, Xraylib::L3_SHELL))

printf("Sr anomalous scattering factor Fi at 10.0 keV: %f\n", Xraylib.Fi(38, 10.0))
printf("Sr anomalous scattering factor Fii at 10.0 keV: %f\n", Xraylib.Fii(38, 10.0))

symbol = Xraylib.AtomicNumberToSymbol(26)
printf("Symbol of element 26 is: %s\n",symbol)

printf("Number of element Fe is: %i\n", Xraylib.SymbolToAtomicNumber("Fe"))

printf("Pb Malpha XRF production cs at 20.0 keV with cascade effect: %f\n",Xraylib.CS_FluorLine_Kissel(82,Xraylib::MA1_LINE,20.0))
printf("Pb Malpha XRF production cs at 20.0 keV with radiative cascade effect: %f\n",Xraylib.CS_FluorLine_Kissel_Radiative_Cascade(82,Xraylib::MA1_LINE,20.0))
printf("Pb Malpha XRF production cs at 20.0 keV with non-radiative cascade effect: %f\n",Xraylib.CS_FluorLine_Kissel_Nonradiative_Cascade(82,Xraylib::MA1_LINE,20.0))
printf("Pb Malpha XRF production cs at 20.0 keV without cascade effect: %f\n",Xraylib.CS_FluorLine_Kissel_no_Cascade(82,Xraylib::MA1_LINE,20.0))


printf("Al mass energy-absorption cs at 20.0 keV: %f\n", Xraylib::CS_Energy(13, 20.0))
printf("Pb mass energy-absorption cs at 40.0 keV: %f\n", Xraylib::CS_Energy(82, 40.0))
printf("CdTe mass energy-absorption cs at 40.0 keV: %f\n", Xraylib::CS_Energy_CP("CdTe", 40.0))


# Si Crystal structure

cryst = Xraylib.Crystal_GetCrystal("Si")
exit(1) if not cryst

printf("Si unit cell dimensions are %f %f %f\n", cryst["a"], cryst["b"], cryst["c"])
printf("Si unit cell angles are %f %f %f\n", cryst["alpha"], cryst["beta"], cryst["gamma"])
printf("Si unit cell volume is %f\n", cryst["volume"])
printf("Si atoms at:\n")
printf("   Z  fraction    X        Y        Z\n")
for i in (0..cryst["n_atom"]-1)
    atom = cryst["atom"][i]
    printf("  %3i %f %f %f %f\n", atom["Zatom"], atom["fraction"], atom["x"], atom["y"], atom["z"])
end

# Si diffraction parameters
printf ("\nSi111 at 8 KeV. Incidence at the Bragg angle:\n")

energy = 8
debye_temp_factor = 1.0
rel_angle = 1.0

bragg = Xraylib.Bragg_angle(cryst, energy, 1, 1, 1)
printf("  Bragg angle: Rad: %f Deg: %f\n", bragg, bragg*180/Math::PI)

q = Xraylib.Q_scattering_amplitude(cryst, energy, 1, 1, 1, rel_angle)
printf("  Q Scattering amplitude: %f\n", q)

f0, fp, fpp =  Xraylib.Atomic_Factors(14, energy, q, debye_temp_factor)
printf("  Atomic factors (Z = 14) f0, fp, fpp: %f, %f, i*%f\n", f0, fp, fpp)

fh = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle)
printf("  FH(1,1,1) structure factor: (%f, %f)\n", fh.real, fh.imag)

f0 = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
printf("  F0=FH(0,0,0) structure factor: (%f, %f)\n", f0.real, f0.imag)

# Diamond diffraction parameters
cryst = Xraylib.Crystal_GetCrystal("Diamond")
exit(1) if not cryst

printf ("\nDiamond 111 at 8 KeV. Incidence at the Bragg angle:\n")

bragg = Xraylib.Bragg_angle(cryst, energy, 1, 1, 1)
printf("  Bragg angle: Rad: %f Deg: %f\n", bragg, bragg*180/Math::PI)

q = Xraylib.Q_scattering_amplitude(cryst, energy, 1, 1, 1, rel_angle)
printf("  Q Scattering amplitude: %f\n", q)

f0, fp, fpp =  Xraylib.Atomic_Factors(6, energy, q, debye_temp_factor)
printf("  Atomic factors (Z = 14) f0, fp, fpp: %f, %f, i*%f\n", f0, fp, fpp)

fh = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle)
printf("  FH(1,1,1) structure factor: (%f, %f)\n", fh.real, fh.imag)

f0 = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
printf("  F0=FH(0,0,0) structure factor: (%f, %f)\n", f0.real, f0.imag)

fhbar = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, -1, -1, -1, debye_temp_factor, rel_angle)

dw = 1e10 * 2 * (Xraylib::R_E / cryst["volume"]) * (Xraylib::KEV2ANGST * Xraylib::KEV2ANGST/ (energy * energy)) * Math.sqrt((fh * fhbar).abs) / Math::PI / Math.sin(2*bragg)
printf("  Darwin width: %f micro-radians\n", 1e6*dw)

# Alpha Quartz diffraction parameters

cryst = Xraylib.Crystal_GetCrystal("AlphaQuartz")

printf("\nAlpha Quartz 020 at 8 KeV. Incidence at the Bragg angle:\n")

bragg = Xraylib.Bragg_angle(cryst, energy, 0, 2, 0)
printf("  Bragg angle: Rad: %f Deg: %f\n", bragg, bragg*180/Math::PI)

q = Xraylib.Q_scattering_amplitude(cryst, energy, 0, 2, 0, rel_angle)
printf("  Q Scattering amplitude: %f\n", q)

f0, fp, fpp = Xraylib.Atomic_Factors(8, energy, q, debye_temp_factor)
printf("  Atomic factors (Z = 8) f0, fp, fpp: %f, %f, i*%f\n", f0, fp, fpp)

fh = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 2, 0, debye_temp_factor, rel_angle)
printf("  FH(0,2,0) structure factor: (%f, %f)\n", fh.real, fh.imag)

f0 = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
printf("  F0=FH(0,0,0) structure factor: (%f, %f)\n", f0.real, f0.imag)

# Muscovite diffraction parameters

cryst = Xraylib.Crystal_GetCrystal("Muscovite")

printf("\nMuscovite 331 at 8 KeV. Incidence at the Bragg angle:\n")

bragg = Xraylib.Bragg_angle(cryst, energy, 3, 3, 1)
printf("  Bragg angle: Rad: %f Deg: %f\n", bragg, bragg*180/Math::PI)

q = Xraylib.Q_scattering_amplitude(cryst, energy, 3, 3, 1, rel_angle)
printf("  Q Scattering amplitude: %f\n", q)

f0, fp, fpp = Xraylib.Atomic_Factors(19, energy, q, debye_temp_factor)
printf("  Atomic factors (Z = 19) f0, fp, fpp: %f, %f, i*%f\n", f0, fp, fpp)

fh = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 3, 3, 1, debye_temp_factor, rel_angle)
printf("  FH(3,3,1) structure factor: (%f, %f)\n", fh.real, fh.imag)

f0 = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
printf("  F0=FH(0,0,0) structure factor: (%f, %f)\n", f0.real, f0.imag)
crystals = Xraylib.Crystal_GetCrystalsList()
counter = 0
printf ("List of available crystals:\n")
crystals.each do |crystal|
	puts "  Crystal #{counter}: #{crystal}"
	counter = counter + 1
end
printf("\n")

# compoundDataNIST tests
cdn = Xraylib.GetCompoundDataNISTByName("Uranium Monocarbide")
printf("Uranium Monocarbide\n")
printf("  Name: %s\n", cdn['name'])
printf("  Density: %f g/cm3\n", cdn['density'])
for i in (0..cdn['nElements']-1)
    	printf("  Element %i: %f %%\n",cdn['Elements'][i],cdn['massFractions'][i]*100.0)
end

cdn = Xraylib.GetCompoundDataNISTByIndex(Xraylib::NIST_COMPOUND_BRAIN_ICRP)
printf("NIST_COMPOUND_BRAIN_ICRP\n")
printf("  Name: %s\n", cdn['name'])
printf("  Density: %f g/cm3\n", cdn['density'])
for i in (0..cdn['nElements']-1)
    	printf("  Element %i: %f %%\n",cdn['Elements'][i],cdn['massFractions'][i]*100.0)
end

nistCompounds = Xraylib.GetCompoundDataNISTList()
counter = 0
printf ("List of available NIST compounds:\n")
nistCompounds.each do |nistCompound|
	puts "  Compound #{counter}: #{nistCompound}"
	counter = counter + 1
end

# radioNuclideData tests
rnd = Xraylib.GetRadioNuclideDataByName("109Cd")
printf("109Cd\n")
printf("  Name: %s\n" , rnd['name'])
printf("  Z: %i\n" , rnd['Z'])
printf("  A: %i\n" , rnd['A'])
printf("  N: %i\n" , rnd['N'])
printf("  Z_xray: %i\n" , rnd['Z_xray'])
printf("  X-rays:\n")
for i in (0..rnd['nXrays']-1)
	printf("  %f keV -> %f\n", Xraylib.LineEnergy(rnd['Z_xray'], rnd['XrayLines'][i]), rnd['XrayIntensities'][i])
end
printf("  Gamma rays:\n")
for i in (0..rnd['nGammas']-1)
	printf("  %f keV -> %f\n" , rnd['GammaEnergies'][i], rnd['GammaIntensities'][i])
end

rnd = Xraylib.GetRadioNuclideDataByIndex(Xraylib::RADIO_NUCLIDE_125I)
printf("RADIO_NUCLIDE_125I\n")
printf("  Name: %s\n" , rnd['name'])
printf("  Z: %i\n" , rnd['Z'])
printf("  A: %i\n" , rnd['A'])
printf("  N: %i\n" , rnd['N'])
printf("  Z_xray: %i\n" , rnd['Z_xray'])
printf("  X-rays:\n")
for i in (0..rnd['nXrays']-1)
	printf("  %f keV -> %f\n", Xraylib.LineEnergy(rnd['Z_xray'], rnd['XrayLines'][i]), rnd['XrayIntensities'][i])
end
printf("  Gamma rays:\n")
for i in (0..rnd['nGammas']-1)
	printf("  %f keV -> %f\n" , rnd['GammaEnergies'][i], rnd['GammaIntensities'][i])
end

radioNuclides = Xraylib.GetRadioNuclideDataList()
counter = 0
printf ("List of available radionuclides:\n")
radioNuclides.each do |radioNuclide|
	puts "  RadioNuclide #{counter}: #{radioNuclide}"
	counter = counter + 1
end
printf("\n--------------------------- END OF XRLEXAMPLE10 -------------------------------\n")
