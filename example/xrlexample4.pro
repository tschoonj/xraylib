
;Copyright (c) 2009, Tom Schoonjans
;All rights reserved.

;Redistribution and use in source and binary forms, with or without
;modification, are permitted provided that the following conditions are met:
;    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
;    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
;    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

;THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


;initialize the xraylib variables
;since this batch script will run in the $MAIN$, there is no need to call the xraylib common block
;however if you would want access to the xraylib variables from within a function or procedure, then you must call it

@xraylib

CATCH, Error_status

IF Error_status NE 0 THEN EXIT,STATUS=0

PRINT,'Example of IDL program using xraylib'
PRINT,'Density of pure Al: ', ElementDensity(13), ' g/cm3'
PRINT,'Ca K-alpha Fluorescence Line Energy: ',LineEnergy(20,KA_LINE)
PRINT,'Fe partial photoionization cs of L3 at 6.0 keV: ',CS_Photo_Partial(26,L3_SHELL,6.0)
PRINT,'Zr L1 edge energy: ',EdgeEnergy(40,L1_SHELL)
PRINT,'Pb Lalpha XRF production cs at 20.0 keV (jump approx): ',CS_FluorLine(82,LA_LINE,20.0)
PRINT,'Pb Lalpha XRF production cs at 20.0 keV (Kissel): ',CS_FluorLine_Kissel(82,LA_LINE,20.0)
PRINT,'Bi M1N2 radiative rate: ',RadRate(83,M1N2_LINE)
PRINT,'U M3O3 Fluorescence Line Energy: ',LineEnergy(92,M3O3_LINE)

cdtest = CompoundParser('Ca(HCO3)2')
PRINT,'Ca(HCO3)2 contains ', cdtest.nAtomsAll, ' atoms, ',cdtest.nElements,' elements and has a molar mass of ', cdtest.molarMass, ' g/mol'
FOR i=0L,cdtest.nElements-1 DO PRINT,'Element ',cdtest.Elements[i],' : ',cdtest.massFractions[i]*100.0,' % and ', cdtest.nAtoms[i], ' atoms'

cdtest = CompoundParser('SiO2')
PRINT,'SiO2 contains ', cdtest.nAtomsAll, ' atoms, ',cdtest.nElements,' elements and has a molar mass of ', cdtest.molarMass, ' g/mol'
FOR i=0L,cdtest.nElements-1 DO PRINT,'Element ',cdtest.Elements[i],' : ',cdtest.massFractions[i]*100.0,' % and ', cdtest.nAtoms[i], ' atoms'

PRINT,'Ca(HCO3)2 Rayleigh cs at 10.0 keV: ',CS_Rayl_CP("Ca(HCO3)2",10.0)

PRINT,'CS2 Refractive Index at 10.0 keV : ',Refractive_Index_Re("CS2",10.0,1.261),' - ',Refractive_Index_Im("CS2",10.0,1.261),' i'
PRINT,'C16H14O3 Refractive Index at 1 keV : ',Refractive_Index_Re("C16H14O3",1.0,1.2),' - ',Refractive_Index_Im("C16H14O3",1.0,1.2),' i'
PRINT,'SiO2 Refractive Index at 5.0 keV : ',Refractive_Index_Re("SiO2",5.0,2.65),' - ',Refractive_Index_Im("SiO2",5.0,2.65),' i'
PRINT,'Compton profile for Fe at pz = 1.1 : ',ComptonProfile(26,1.1)
PRINT,'M5 Partial Compton profile for Fe at pz = 1.1 : ',ComptonProfile_Partial(26,M5_SHELL,1.1)
PRINT,'K atomic level width for Fe: ',$
        AtomicLevelWidth(26,K_SHELL)
PRINT,'M1->M5 Coster-Kronig transition probability for Au : ',CosKronTransProb(79,FM15_TRANS)
PRINT,'L1->L3 Coster-Kronig transition probability for Fe : ',CosKronTransProb(26,FL13_TRANS)
PRINT,'Au Ma1 XRF production cs at 10.0 keV (Kissel): ',CS_FluorLine_Kissel(79,MA1_LINE,10.0)
PRINT,'Au Mb XRF production cs at 10.0 keV (Kissel): ',CS_FluorLine_Kissel(79,MB_LINE,10.0)
PRINT,'Au Mg XRF production cs at 10.0 keV (Kissel): ',CS_FluorLine_Kissel(79,MG_LINE,10.0)

PRINT, 'Sr anomalous scattering factor Fi at 10.0 keV: ', Fi(38, 10.0)
PRINT, 'Sr anomalous scattering factor Fii at 10.0 keV: ', Fii(38, 10.0)

PRINT,'Symbol of element 26 is: ',AtomicNumberToSymbol(26)
PRINT,'Number of element Fe is: ',SymbolToAtomicNumber('Fe')
PRINT,'Bi L2-M5M5 Auger non-radiative rate: ', AugerRate(86,L2_M5M5_AUGER)
PRINT,'Bi L3 Auger yield: ', AugerYield(86, L3_SHELL)

PRINT,'Pb Malpha XRF production cs at 20.0 keV with cascade effect: ',CS_FluorLine_Kissel(82,MA1_LINE,20.0)
PRINT,'Pb Malpha XRF production cs at 20.0 keV with radiative cascade effect: ',CS_FluorLine_Kissel_Radiative_Cascade(82,MA1_LINE,20.0)
PRINT,'Pb Malpha XRF production cs at 20.0 keV with non-radiative cascade effect: ',CS_FluorLine_Kissel_Nonradiative_Cascade(82,MA1_LINE,20.0)
PRINT,'Pb Malpha XRF production cs at 20.0 keV without cascade effect: ',CS_FluorLine_Kissel_no_Cascade(82,MA1_LINE,20.0)

PRINT,'Al mass energy-absorption cs at 20.0 keV: ', CS_Energy(13, 20.0)
PRINT,'Pb mass energy-absorption cs at 40.0 keV: ', CS_Energy(82, 40.0)
PRINT,'CdTe mass energy-absorption cs at 40.0 keV: ', CS_Energy_CP('CdTe', 40.0)

;Si crystal structure
cryst = Crystal_GetCrystal('Si')
PRINT,'Si unit cell dimensions are ',cryst.a,cryst.b,cryst.c
PRINT,'Si unit cell angle are ',cryst.alpha,cryst.beta,cryst.gamma
PRINT,'Si unit cell volume is ',cryst.volume
PRINT,'Si atoms at:'
PRINT,'   Z  fraction      X         Y         Z'
FOR i=0,7 DO $
	PRINT,FORMAT='(%"%4i  %f  %f  %f  %f")',$
		cryst.atom[i].Zatom,$
		cryst.atom[i].fraction,$
		cryst.atom[i].x,$
		cryst.atom[i].y,$
		cryst.atom[i].z

PRINT,''
PRINT,'Si111 at 8 KeV. Incidence at the Bragg angle:'
energy = 8
debye_temp_factor = 1.0
rel_angle = 1.0

bragg = Bragg_angle (cryst, energy, 1, 1, 1)
PRINT, FORMAT='(%"  Bragg angle: Rad: %f  Deg: %f")',bragg, bragg*180/!PI

q = Q_scattering_amplitude (cryst, energy, 1, 1, 1, rel_angle)
PRINT, FORMAT='(%"  Q Scattering amplitude: %f")',q

Atomic_Factors, 14, energy, q, debye_temp_factor, f0, fp, fpp
PRINT, FORMAT='(%"  Atomic factors (Z = 14) f0, fp, fpp: %f, %f, i*%f")', f0, fp, fpp

FH = Crystal_F_H_StructureFactor (cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle);
PRINT, FORMAT='(%"  FH(1,1,1) structure factor: (%f, %f)")', REAL_PART(FH), IMAGINARY(FH)

F0 = Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
PRINT, FORMAT='(%"  F0=FH(0,0,0) structure factor: (%f, %f)")', REAL_PART(F0), IMAGINARY(F0)

; Diamond diffraction parameters
cryst = Crystal_GetCrystal('Diamond')

PRINT, ''
PRINT, 'Diamond 111 at 8 KeV. Incidence at the Bragg angle:'
bragg = Bragg_angle (cryst, energy, 1, 1, 1);
PRINT, FORMAT='(%"  Bragg angle: Rad: %f  Deg: %f")',bragg, bragg*180/!PI

q = Q_scattering_amplitude (cryst, energy, 1, 1, 1, rel_angle)
PRINT, FORMAT='(%"  Q Scattering amplitude: %f")',q

Atomic_Factors, 6, energy, q, debye_temp_factor, f0, fp, fpp
PRINT, FORMAT='(%"  Atomic factors (Z = 6) f0, fp, fpp: %f, %f, i*%f")', f0, fp, fpp

FH = Crystal_F_H_StructureFactor (cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle);
PRINT, FORMAT='(%"  FH(1,1,1) structure factor: (%f, %f)")', REAL_PART(FH), IMAGINARY(FH)

F0 = Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
PRINT, FORMAT='(%"  F0=FH(0,0,0) structure factor: (%f, %f)")', REAL_PART(F0), IMAGINARY(F0)

FHBar = Crystal_F_H_StructureFactor (cryst, energy, -1, -1, -1, debye_temp_factor, rel_angle);

dw = 1E10 * 2 * (R_E / cryst.volume) * (KEV2ANGST * KEV2ANGST/ (energy * energy)) * SQRT(ABS(FH * FHbar)) / !PI / SIN(2*bragg);
PRINT, FORMAT='(%"  Darwin width: %f micro-radians")', 1e6*dw

; Alpha Quartz diffraction parameters
cryst = Crystal_GetCrystal('AlphaQuartz')

PRINT, ''
PRINT,'Alpha Quartz 020 at 8 KeV. Incidence at the Bragg angle:'

bragg = Bragg_angle (cryst, energy, 0, 2, 0);
PRINT, FORMAT='(%"  Bragg angle: Rad: %f  Deg: %f")',bragg, bragg*180/!PI

q = Q_scattering_amplitude (cryst, energy, 0, 2, 0, rel_angle)
PRINT, FORMAT='(%"  Q Scattering amplitude: %f")',q

Atomic_Factors, 8, energy, q, debye_temp_factor, f0, fp, fpp
PRINT, FORMAT='(%"  Atomic factors (Z = 8) f0, fp, fpp: %f, %f, i*%f")', f0, fp, fpp

FH = Crystal_F_H_StructureFactor (cryst, energy, 0, 2, 0, debye_temp_factor, rel_angle);
PRINT, FORMAT='(%"  FH(0,2,0) structure factor: (%f, %f)")', REAL_PART(FH), IMAGINARY(FH)

F0 = Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
PRINT, FORMAT='(%"  F0=FH(0,0,0) structure factor: (%f, %f)")', REAL_PART(F0), IMAGINARY(F0)

; Muscovite diffraction parameters
cryst = Crystal_GetCrystal('Muscovite')

PRINT, ''
PRINT, 'Muscovite 331 at 8 KeV. Incidence at the Bragg angle:'

bragg = Bragg_angle (cryst, energy, 3, 3, 1);
PRINT, FORMAT='(%"  Bragg angle: Rad: %f  Deg: %f")',bragg, bragg*180/!PI

q = Q_scattering_amplitude (cryst, energy, 3, 3, 1, rel_angle)
PRINT, FORMAT='(%"  Q Scattering amplitude: %f")',q

Atomic_Factors, 19, energy, q, debye_temp_factor, f0, fp, fpp
PRINT, FORMAT='(%"  Atomic factors (Z = 19) f0, fp, fpp: %f, %f, i*%f")', f0, fp, fpp

FH = Crystal_F_H_StructureFactor (cryst, energy, 3, 3, 1, debye_temp_factor, rel_angle);
PRINT, FORMAT='(%"  FH(3,3,1) structure factor: (%f, %f)")', REAL_PART(FH), IMAGINARY(FH)

F0 = Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
PRINT, FORMAT='(%"  F0=FH(0,0,0) structure factor: (%f, %f)")', REAL_PART(F0), IMAGINARY(F0)

crystals = Crystal_GetCrystalsList()
PRINT, 'List of available crystals'
FOR i=0,N_ELEMENTS(crystals)-1 DO $
	PRINT, FORMAT='(%"  Crystal %i: %s")', i, crystals[i]

PRINT, ''

; compoundDataNIST tests
cdn = GetCompoundDataNISTByName('Uranium Monocarbide')
PRINT, 'Uranium Monocarbide'
PRINT, '  Name: ',cdn.name
PRINT, '  Density: ',cdn.density, ' g/cm3'
FOR i=0,cdn.nElements-1 DO $
	PRINT, FORMAT='(%"  Element %i: %f %%")', cdn.Elements[i], $
	cdn.massFractions[i]*100.0

cdn = GetCompoundDataNISTByIndex(NIST_COMPOUND_BRAIN_ICRP)
PRINT, 'NIST_COMPOUND_BRAIN_ICRP'
PRINT, '  Name: ',cdn.name
PRINT, '  Density: ',cdn.density, ' g/cm3'
FOR i=0,cdn.nElements-1 DO $
	PRINT, FORMAT='(%"  Element %i: %f %%")', cdn.Elements[i], $
	cdn.massFractions[i]*100.0

nistCompounds = GetCompoundDataNISTList()
PRINT, 'List of available NIST compounds'
FOR i=0,N_ELEMENTS(nistCompounds)-1 DO $
	PRINT, FORMAT='(%"  Compound %i: %s")', i, nistCompounds[i]

PRINT,''

; radioNuclides
rnd = GetRadioNuclideDataByName('109Cd')
PRINT, '109Cd'
PRINT, '  Name: ', rnd.name
PRINT,'  Z: ',rnd.Z
PRINT, '  A: ',rnd.A
PRINT, '  N: ',rnd.N
PRINT, '  Z_xray: ',rnd.Z_xray
;PRINT, '  nXrays: ',rnd.nXrays
;PRINT, '  nXrays: ',N_ELEMENTS(rnd.XrayLines)
;PRINT, '  nXrays: ',N_ELEMENTS(rnd.XrayIntensities)
;PRINT, '  nGammas: ',rnd.nGammas
PRINT, '  X-rays: '
FOR i=0,rnd.nXrays-1 DO $
        PRINT, FORMAT='(%"  %f keV -> %f")',$
        LineEnergy(rnd.Z_xray, rnd.XrayLines[i]),$
        rnd.XrayIntensities[i]
PRINT, '  Gamma rays: '
FOR i=0,rnd.nGammas-1 DO $
        PRINT, FORMAT='(%"  %f keV -> %f")',$
        rnd.GammaEnergies[i],$
        rnd.GammaIntensities[i]

rnd = GetRadioNuclideDataByIndex(RADIO_NUCLIDE_125I)
PRINT, 'RADIO_NUCLIDE_125I'
PRINT, '  Name: ', rnd.name
PRINT, '  Z: ',rnd.Z
PRINT, '  A: ',rnd.A
PRINT, '  N: ',rnd.N
PRINT, '  Z_xray: ',rnd.Z_xray
PRINT, '  X-rays: '
FOR i=0,rnd.nXrays-1 DO $
        PRINT, FORMAT='(%"  %f keV -> %f")',$
        LineEnergy(rnd.Z_xray, rnd.XrayLines[i]),$
        rnd.XrayIntensities[i]

PRINT, '  Gamma rays: '
FOR i=0,rnd.nGammas-1 DO $
        PRINT, FORMAT='(%"  %f keV -> %f")',$
        rnd.GammaEnergies[i],$
        rnd.GammaIntensities[i]

radioNuclides = GetRadioNuclideDataList()
PRINT, 'List of available radionuclides'
FOR i=0,N_ELEMENTS(radioNuclides)-1 DO $
	PRINT, FORMAT='(%"  Radionuclide %i: %s")', i, radioNuclides[i]



PRINT,''
PRINT,'--------------------------- END OF XRLEXAMPLE4 -------------------------------'
PRINT,''

;the value of !ERROR_STATE will determine the exit status of IDL and therefore the outcome of make check
IF !ERROR_STATE.CODE eq 0 THEN EXIT,STATUS=0 ELSE EXIT,STATUS=1
