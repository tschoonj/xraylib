#Copyright (c) 2009, 2010, 2011 Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""Example of using various xraylib functionality in python."""

import os
if "EXTRA_DLL_SEARCH_PATHS" in os.environ and hasattr(os, "add_dll_directory"):
    for path in os.environ["EXTRA_DLL_SEARCH_PATHS"].split(os.pathsep):
        os.add_dll_directory(path)
import xraylib
import math
import numpy as np


xraylib.XRayInit()

print("Example of python program using xraylib")
print("xraylib version: {}".format(xraylib.__version__))
print("Density of pure Al : {} g/cm3".format(xraylib.ElementDensity(13)))
print("Ca K-alpha Fluorescence Line Energy: {}".format(xraylib.LineEnergy(20, xraylib.KA_LINE)))
print("Fe partial photoionization cs of L3 at 6.0 keV: {}".format(xraylib.CS_Photo_Partial(26, xraylib.L3_SHELL,6.0)))
print("Zr L1 edge energy: {}".format(xraylib.EdgeEnergy(40, xraylib.L1_SHELL)))
print("Pb Lalpha XRF production cs at 20.0 keV (jump approx): {}".format(xraylib.CS_FluorLine(82, xraylib.LA_LINE,20.0)))
print("Pb Lalpha XRF production cs at 20.0 keV (Kissel): {}".format(xraylib.CS_FluorLine_Kissel(82, xraylib.LA_LINE,20.0)))
print("Bi M1N2 radiative rate: {}".format(xraylib.RadRate(83, xraylib.M1N2_LINE)))
print("U M3O3 Fluorescence Line Energy: {}".format(xraylib.LineEnergy(92, xraylib.M3O3_LINE)))
print("Ca(HCO3)2 Rayleigh cs at 10.0 keV: {}".format(xraylib.CS_Rayl_CP("Ca(HCO3)2",10.0)))

cdtest = xraylib.CompoundParser("Ca(HCO3)2")
print("Ca(HCO3)2 contains {} atoms, {} elements and has a molar mass of {} g/mol".format(cdtest['nAtomsAll'], cdtest['nElements'], cdtest['molarMass']))
for i in range(cdtest['nElements']):
    print("Element {}: {} % and {} atoms".format(cdtest['Elements'][i], cdtest['massFractions'][i]*100.0, cdtest['nAtoms'][i]))

cdtest = xraylib.CompoundParser("SiO2")
print("SiO2 contains {} atoms, {} elements and has a molar mass of {} g/mol".format(cdtest['nAtomsAll'], cdtest['nElements'], cdtest['molarMass']))
for i in range(cdtest['nElements']):
    print("Element {}: {} % and {} atoms".format(cdtest['Elements'][i], cdtest['massFractions'][i]*100.0, cdtest['nAtoms'][i]))

print("CS2 Refractive Index at 10.0 keV : {} - {} i".format(xraylib.Refractive_Index_Re("CS2",10.0,1.261), xraylib.Refractive_Index_Im("CS2",10.0,1.261)))
print("C16H14O3 Refractive Index at 1 keV : {} - {} i".format(xraylib.Refractive_Index_Re("C16H14O3",1.0,1.2), xraylib.Refractive_Index_Im("C16H14O3",1.0,1.2)))
print("SiO2 Refractive Index at 5 keV : {} - {} i".format(xraylib.Refractive_Index_Re("SiO2",5.0,2.65), xraylib.Refractive_Index_Im("SiO2",5.0,2.65)))
print("Compton profile for Fe at pz = 1.1 : {}".format(xraylib.ComptonProfile(26,1.1)))
print("M5 Compton profile for Fe at pz = 1.1 : {}".format(xraylib.ComptonProfile_Partial(26, xraylib.M5_SHELL,1.1)))
print("M1->M5 Coster-Kronig transition probability for Au : {}".format(xraylib.CosKronTransProb(79, xraylib.FM15_TRANS)))
print("L1->L3 Coster-Kronig transition probability for Fe : {}".format(xraylib.CosKronTransProb(26, xraylib.FL13_TRANS)))
print("Au Ma1 XRF production cs at 10.0 keV (Kissel): {}".format(xraylib.CS_FluorLine_Kissel(79, xraylib.MA1_LINE,10.0)))
print("Au Mb XRF production cs at 10.0 keV (Kissel): {}".format(xraylib.CS_FluorLine_Kissel(79, xraylib.MB_LINE,10.0)))
print("Au Mg XRF production cs at 10.0 keV (Kissel): {}".format(xraylib.CS_FluorLine_Kissel(79, xraylib.MG_LINE,10.0)))
print("K atomic level width for Fe: {}".format(xraylib.AtomicLevelWidth(26, xraylib.K_SHELL)))
print("Bi L2-M5M5 Auger non-radiative rate: {}".format(xraylib.AugerRate(86, xraylib.L2_M5M5_AUGER)))
print("Bi L3 Auger yield: {}".format(xraylib.AugerYield(86, xraylib.L3_SHELL)))

print("Sr anomalous scattering factor Fi at 10.0 keV: {}".format(xraylib.Fi(38, 10.0)))
print("Sr anomalous scattering factor Fii at 10.0 keV: {}".format(xraylib.Fii(38, 10.0)))

symbol = xraylib.AtomicNumberToSymbol(26)
print("Symbol of element 26 is: {}".format(symbol))
print("Number of element Fe is: {}".format(xraylib.SymbolToAtomicNumber("Fe")))
Z = np.array([26])
symbol = xraylib.AtomicNumberToSymbol(Z[0])
print("Symbol of element 26 is: {}".format(symbol))
print("Pb Malpha XRF production cs at 20.0 keV with cascade effect: {}".format(xraylib.CS_FluorLine_Kissel(82, xraylib.MA1_LINE,20.0)))
print("Pb Malpha XRF production cs at 20.0 keV with radiative cascade effect: {}".format(xraylib.CS_FluorLine_Kissel_Radiative_Cascade(82, xraylib.MA1_LINE,20.0)))
print("Pb Malpha XRF production cs at 20.0 keV with non-radiative cascade effect: {}".format(xraylib.CS_FluorLine_Kissel_Nonradiative_Cascade(82, xraylib.MA1_LINE,20.0)))
print("Pb Malpha XRF production cs at 20.0 keV without cascade effect: {}".format(xraylib.CS_FluorLine_Kissel_no_Cascade(82, xraylib.MA1_LINE,20.0)))
print("Al mass energy-absorption cs at 20.0 keV: {}".format(xraylib.CS_Energy(13, 20.0)))
print("Pb mass energy-absorption cs at 40.0 keV: {}".format(xraylib.CS_Energy(82, 40.0)))
print("CdTe mass energy-absorption cs at 40.0 keV: {}".format(xraylib.CS_Energy_CP("CdTe", 40.0)))

cryst = xraylib.Crystal_GetCrystal("Si")
if cryst is None:
    raise KeyError('Diamond crystal not found')

print("Si unit cell dimensions are {} {} {}".format(cryst['a'],cryst['b'],cryst['c']))
print("Si unit cell angles are {} {} {}".format(cryst['alpha'],cryst['beta'],cryst['gamma']))
print("Si unit cell volume is {}".format(cryst['volume']))
print("Si atoms at:")
print("   Z  fraction    X        Y        Z")
for i in range(cryst['n_atom']):
    atom =  cryst['atom'][i]
    print("  {} {} {} {} {}".format(atom['Zatom'], atom['fraction'], atom['x'], atom['y'], atom['z']))
print("")

print("Si111 at 8 KeV. Incidence at the Bragg angle:")
energy = 8
debye_temp_factor = 1.0
rel_angle = 1.0

bragg = xraylib.Bragg_angle(cryst, energy, 1, 1, 1)
print("  Bragg angle: Rad: {} Deg: {}".format(bragg, bragg*180/math.pi))

q = xraylib.Q_scattering_amplitude(cryst, energy, 1, 1, 1, rel_angle)
print("  Q Scattering amplitude: {}".format(q))

#notice the 3 return values!!!
f0, fp, fpp = xraylib.Atomic_Factors(14, energy, q, debye_temp_factor)
print("  Atomic factors (Z = 14) f0, fp, fpp: {}, {}, i*{}".format(f0, fp, fpp))

FH = xraylib.Crystal_F_H_StructureFactor(cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle)
print("  FH(1,1,1) structure factor: ({}, {})".format(FH.real, FH.imag))

F0 = xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
print("  F0=FH(0,0,0) structure factor: ({}, {})".format(F0.real, F0.imag))

# Diamond diffraction parameters
cryst = xraylib.Crystal_GetCrystal("Diamond")
if cryst is None:
    raise KeyError('Diamond crystal not found')

print("")
print("Diamond 111 at 8 KeV. Incidence at the Bragg angle:")
bragg = xraylib.Bragg_angle(cryst, energy, 1, 1, 1)
print("  Bragg angle: Rad: {} Deg: {}".format(bragg, bragg*180/math.pi))

q = xraylib.Q_scattering_amplitude(cryst, energy, 1, 1, 1, rel_angle)
print("  Q Scattering amplitude: {}".format(q))

# notice the 3 return values!!!
f0, fp, fpp = xraylib.Atomic_Factors(6, energy, q, debye_temp_factor)
print("  Atomic factors (Z = 6) f0, fp, fpp: {}, {}, {}".format(f0, fp, fpp))

FH = xraylib.Crystal_F_H_StructureFactor(cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle)
print("  FH(1,1,1) structure factor: ({}, {})".format(FH.real, FH.imag))

F0 = xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
print("  F0=FH(0,0,0) structure factor: ({}, {})".format(F0.real, F0.imag))

FHbar = xraylib.Crystal_F_H_StructureFactor(cryst, energy, -1, -1, -1, debye_temp_factor, rel_angle)
dw = 1e10 * 2 * (xraylib.R_E / cryst['volume']) * (xraylib.KEV2ANGST * xraylib.KEV2ANGST/ (energy * energy)) * math.sqrt(abs(FH * FHbar)) / math.pi / math.sin(2*bragg)

print("  Darwin width: {} micro-radians".format(1.0E6*dw))
print("")

# Alpha Quartz diffraction parameters

cryst = xraylib.Crystal_GetCrystal("AlphaQuartz")

print("Alpha Quartz 020 at 8 KeV. Incidence at the Bragg angle:")

bragg = xraylib.Bragg_angle(cryst, energy, 0, 2, 0)
print("  Bragg angle: Rad: {} Deg: {}".format(bragg, bragg*180/math.pi))

q = xraylib.Q_scattering_amplitude(cryst, energy, 0, 2, 0, rel_angle)
print("  Q Scattering amplitude: {}".format(q))

f0, fp, fpp =xraylib.Atomic_Factors(8, energy, q, debye_temp_factor)
print("  Atomic factors (Z = 8) f0, fp, fpp: {}, {}, {}".format( f0, fp, fpp))

FH = xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 2, 0, debye_temp_factor, rel_angle)
print("  FH(0,2,0) structure factor: ({}, {})".format(FH.real, FH.imag))

F0 = xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
print("  F0=FH(0,0,0) structure factor: ({}, {})".format(F0.real, F0.imag))

#Muscovite diffraction parameters

cryst = xraylib.Crystal_GetCrystal("Muscovite")

print("\nMuscovite 331 at 8 KeV. Incidence at the Bragg angle:")

bragg = xraylib.Bragg_angle(cryst, energy, 3, 3, 1)
print("  Bragg angle: Rad: {} Deg: {}".format(bragg, bragg*180/math.pi))

q = xraylib.Q_scattering_amplitude(cryst, energy, 3, 3, 1, rel_angle)
print("  Q Scattering amplitude: {}".format(q))

f0, fp, fpp =xraylib.Atomic_Factors(19, energy, q, debye_temp_factor)
print("  Atomic factors (Z = 19) f0, fp, fpp: {}, {}, {}".format(f0, fp, fpp))

FH = xraylib.Crystal_F_H_StructureFactor(cryst, energy, 3, 3, 1, debye_temp_factor, rel_angle)
print("  FH(3,3,1) structure factor: ({}, {})".format(FH.real, FH.imag))

F0 = xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
print("  F0=FH(0,0,0) structure factor: ({}, {})".format(F0.real, F0.imag))

crystals = xraylib.Crystal_GetCrystalsList()
print("List of available crystals:")
for i in range(len(crystals)):
    print("  Crystal {}: {}".format(i, crystals[i]))
print("")

cdn = xraylib.GetCompoundDataNISTByName("Uranium Monocarbide")
print("Uranium Monocarbide")
print("  Name: {}".format(cdn['name']))
print("  Density: {}".format(cdn['density']))
for i in range(cdn['nElements']):
    print("  Element {}: {} %%".format(cdn['Elements'][i], cdn['massFractions'][i]*100.0))

cdn = xraylib.GetCompoundDataNISTByIndex(xraylib.NIST_COMPOUND_BRAIN_ICRP)
print("NIST_COMPOUND_BRAIN_ICRP")
print("  Name: {}".format(cdn['name']))
print("  Density: {}".format(cdn['density']))
for i in range(cdn['nElements']):
    print("  Element {}: {} %%".format(cdn['Elements'][i], cdn['massFractions'][i]*100.0))

nistCompounds = xraylib.GetCompoundDataNISTList()
print("List of available NIST compounds:")
for i in range(len(nistCompounds)):
    print("  Compound {}: {}".format(i,nistCompounds[i]))

print("")

# radioNuclideData tests
rnd = xraylib.GetRadioNuclideDataByName("109Cd")
print("109Cd")
print("  Name: {}".format(rnd['name']))
print("  Z: {}".format(rnd['Z']))
print("  A: {}".format(rnd['A']))
print("  N: {}".format(rnd['N']))
print("  Z_xray: {}".format(rnd['Z_xray']))
print("  X-rays:")
for i in range(rnd['nXrays']):
    print("  {} keV -> {}".format(xraylib.LineEnergy(rnd['Z_xray'], rnd['XrayLines'][i]), rnd['XrayIntensities'][i]))
print("  Gamma rays:")
for i in range(rnd['nGammas']):
    print("  {} keV -> {}".format(rnd['GammaEnergies'][i], rnd['GammaIntensities'][i]))

rnd = xraylib.GetRadioNuclideDataByIndex(xraylib.RADIO_NUCLIDE_125I)
print("RADIO_NUCLIDE_125I")
print("  Name: {}".format(rnd['name']))
print("  Z: {}".format(rnd['Z']))
print("  A: {}".format(rnd['A']))
print("  N: {}".format(rnd['N']))
print("  Z_xray: {}".format(rnd['Z_xray']))
print("  X-rays:")
for i in range(rnd['nXrays']):
    print("  {} keV -> {}".format(xraylib.LineEnergy(rnd['Z_xray'], rnd['XrayLines'][i]), rnd['XrayIntensities'][i]))
print("  Gamma rays:")
for i in range(rnd['nGammas']):
    print("  {} keV -> {}".format(rnd['GammaEnergies'][i], rnd['GammaIntensities'][i]))

radioNuclides = xraylib.GetRadioNuclideDataList()
print("List of available radionuclides:")
for i in range(len(radioNuclides)):
    print("  Radionuclide {}: {}".format(i, radioNuclides[i]))
print("")
print("------------------------ END OF XRLEXAMPLE5 -------------------------")
print("")
