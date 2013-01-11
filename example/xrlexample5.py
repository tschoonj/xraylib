#Copyright (c) 2009, 2010, 2011 Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#from xraylib import *
import sys, string
import xraylib
import math


if __name__ == '__main__' :
	xraylib.XRayInit()
	print "Example of python program using xraylib"
	print "Ca K-alpha Fluorescence Line Energy: %f" % xraylib.LineEnergy(20,xraylib.KA_LINE)
	print "Fe partial photoionization cs of L3 at 6.0 keV: %f" % xraylib.CS_Photo_Partial(26,xraylib.L3_SHELL,6.0)
	print "Zr L1 edge energy: %f" % xraylib.EdgeEnergy(40,xraylib.L1_SHELL)
	print "Pb Lalpha XRF production cs at 20.0 keV (jump approx): %f" % xraylib.CS_FluorLine(82,xraylib.LA_LINE,20.0)
	print "Pb Lalpha XRF production cs at 20.0 keV (Kissel): %f" % xraylib.CS_FluorLine_Kissel(82,xraylib.LA_LINE,20.0)
  	print "Bi M1N2 radiative rate: %f" % xraylib.RadRate(83,xraylib.M1N2_LINE)
	print "U M3O3 Fluorescence Line Energy: %f" % xraylib.LineEnergy(92,xraylib.M3O3_LINE)
	print "Ca(HCO3)2 Rayleigh cs at 10.0 keV: %f" % xraylib.CS_Rayl_CP("Ca(HCO3)2",10.0)

	cdtest = xraylib.CompoundParser("Ca(HCO3)2")
	print "Ca(HCO3)2 contains %i atoms and %i elements"% (cdtest['nAtomsAll'], cdtest['nElements'])
	for i in range(cdtest['nElements']):
        	print "Element %i: %lf %%" % (cdtest['Elements'][i],cdtest['massFractions'][i]*100.0)
		
	cdtest = xraylib.CompoundParser("SiO2")
	print "SiO2 contains %i atoms and %i elements"% (cdtest['nAtomsAll'], cdtest['nElements'])
	for i in range(cdtest['nElements']):
        	print "Element %i: %lf %%" % (cdtest['Elements'][i],cdtest['massFractions'][i]*100.0)
		

	print "CS2 Refractive Index at 10.0 keV : %g - %g i" % (xraylib.Refractive_Index_Re("CS2",10.0,1.261),xraylib.Refractive_Index_Im("CS2",10.0,1.261))
	print "C16H14O3 Refractive Index at 1 keV : %g - %g i" % (xraylib.Refractive_Index_Re("C16H14O3",1.0,1.2),xraylib.Refractive_Index_Im("C16H14O3",1.0,1.2))
	print "SiO2 Refractive Index at 5 keV : %g - %g i" % (xraylib.Refractive_Index_Re("SiO2",5.0,2.65),xraylib.Refractive_Index_Im("SiO2",5.0,2.65))
	print "Compton profile for Fe at pz = 1.1 : %g" % xraylib.ComptonProfile(26,1.1)
	print "M5 Compton profile for Fe at pz = 1.1 : %g" % xraylib.ComptonProfile_Partial(26,xraylib.M5_SHELL,1.1)
	print "M1->M5 Coster-Kronig transition probability for Au : %f" % xraylib.CosKronTransProb(79,xraylib.FM15_TRANS)
	print "L1->L3 Coster-Kronig transition probability for Fe : %f" % xraylib.CosKronTransProb(26,xraylib.FL13_TRANS)
	print "Au Ma1 XRF production cs at 10.0 keV (Kissel): %f" % xraylib.CS_FluorLine_Kissel(79,xraylib.MA1_LINE,10.0)
	print "Au Mb XRF production cs at 10.0 keV (Kissel): %f" % xraylib.CS_FluorLine_Kissel(79,xraylib.MB_LINE,10.0)
	print "Au Mg XRF production cs at 10.0 keV (Kissel): %f" % xraylib.CS_FluorLine_Kissel(79,xraylib.MG_LINE,10.0)
	print "K atomic level width for Fe: %g" % xraylib.AtomicLevelWidth(26,xraylib.K_SHELL)
	print "Bi L2-M5M5 Auger non-radiative rate: %g" % xraylib.AugerRate(86,xraylib.L2_M5M5_AUGER)
	symbol = xraylib.AtomicNumberToSymbol(26)
	print "Symbol of element 26 is: %s" % symbol
	print "Number of element Fe is: %i" % xraylib.SymbolToAtomicNumber("Fe")
	print "Pb Malpha XRF production cs at 20.0 keV with cascade effect: %g" % xraylib.CS_FluorLine_Kissel(82,xraylib.MA1_LINE,20.0)
	print "Pb Malpha XRF production cs at 20.0 keV with radiative cascade effect: %g" % xraylib.CS_FluorLine_Kissel_Radiative_Cascade(82,xraylib.MA1_LINE,20.0)
	print "Pb Malpha XRF production cs at 20.0 keV with non-radiative cascade effect: %g" % xraylib.CS_FluorLine_Kissel_Nonradiative_Cascade(82,xraylib.MA1_LINE,20.0)
	print "Pb Malpha XRF production cs at 20.0 keV without cascade effect: %g" % xraylib.CS_FluorLine_Kissel_no_Cascade(82,xraylib.MA1_LINE,20.0)
	print "Al mass energy-absorption cs at 20.0 keV: %f" % xraylib.CS_Energy(13, 20.0)
	print "Pb mass energy-absorption cs at 40.0 keV: %f" % xraylib.CS_Energy(82, 40.0)
	print "CdTe mass energy-absorption cs at 40.0 keV: %f" % xraylib.CS_Energy_CP("CdTe", 40.0)


	cryst = xraylib.Crystal_GetCrystal("Si")
	if (cryst == None):
		sys.exit(1)
	print "Si unit cell dimensions are %f %f %f" % (cryst['a'],cryst['b'],cryst['c'])
	print "Si unit cell angles are %f %f %f" % (cryst['alpha'],cryst['beta'],cryst['gamma'])
	print "Si unit cell volume is %f" % cryst['volume'] 
	print "Si atoms at:"
	print "   Z  fraction    X        Y        Z"
	for i in range(cryst['n_atom']):
		atom =  cryst['atom'][i]
		print "  %3i %f %f %f %f" % (atom['Zatom'], atom['fraction'], atom['x'], atom['y'], atom['z'])
	print ""

	
  	print "Si111 at 8 KeV. Incidence at the Bragg angle:"
  	energy = 8
	debye_temp_factor = 1.0
	rel_angle = 1.0

  	bragg = xraylib.Bragg_angle (cryst, energy, 1, 1, 1)
  	print "  Bragg angle: Rad: %f Deg: %f" % (bragg, bragg*180/math.pi)

	q = xraylib.Q_scattering_amplitude (cryst, energy, 1, 1, 1, rel_angle);
  	print "  Q Scattering amplitude: %f" % q
  
  	#notice the 3 return values!!!
	f0, fp, fpp = xraylib.Atomic_Factors (14, energy, q, debye_temp_factor)
	print "  Atomic factors (Z = 14) f0, fp, fpp: %f, %f, i*%f" % (f0, fp, fpp)

  	FH = xraylib.Crystal_F_H_StructureFactor (cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle)
	print "  FH(1,1,1) structure factor: (%f, %f)" % (FH.real, FH.imag)

	F0 = xraylib.Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
	print "  F0=FH(0,0,0) structure factor: (%f, %f)" % (F0.real, F0.imag)

	# Diamond diffraction parameters
	cryst = xraylib.Crystal_GetCrystal("Diamond")
	if (cryst == None):
		sys.exit(1)

	print ""
  	print "Diamond 111 at 8 KeV. Incidence at the Bragg angle:"
  	bragg = xraylib.Bragg_angle (cryst, energy, 1, 1, 1)
  	print "  Bragg angle: Rad: %f Deg: %f" % (bragg, bragg*180/math.pi)

	q = xraylib.Q_scattering_amplitude (cryst, energy, 1, 1, 1, rel_angle);
  	print "  Q Scattering amplitude: %f" % q
  
  	#notice the 3 return values!!!
	f0, fp, fpp = xraylib.Atomic_Factors (6, energy, q, debye_temp_factor)
	print "  Atomic factors (Z = 14) f0, fp, fpp: %f, %f, i*%f" % (f0, fp, fpp)

  	FH = xraylib.Crystal_F_H_StructureFactor (cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle)
	print "  FH(1,1,1) structure factor: (%f, %f)" % (FH.real, FH.imag)

	F0 = xraylib.Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
	print "  F0=FH(0,0,0) structure factor: (%f, %f)" % (F0.real, F0.imag)

	FHbar = xraylib.Crystal_F_H_StructureFactor (cryst, energy, -1, -1, -1, debye_temp_factor, rel_angle)
	dw = 1e10 * 2 * (xraylib.R_E / cryst['volume']) * (xraylib.KEV2ANGST * xraylib.KEV2ANGST/ (energy * energy)) * math.sqrt(abs(FH * FHbar)) / math.pi / math.sin(2*bragg)

  	print "  Darwin width: %f micro-radians" % (1.0E6*dw)
	print ""

  	# Alpha Quartz diffraction parameters 

  	cryst = xraylib.Crystal_GetCrystal("AlphaQuartz")

  	print "Alpha Quartz 020 at 8 KeV. Incidence at the Bragg angle:"

  	bragg = xraylib.Bragg_angle (cryst, energy, 0, 2, 0);
  	print "  Bragg angle: Rad: %f Deg: %f" % (bragg, bragg*180/math.pi)

  	q = xraylib.Q_scattering_amplitude (cryst, energy, 0, 2, 0, rel_angle)
  	print "  Q Scattering amplitude: %f" % q

  	f0, fp, fpp =xraylib.Atomic_Factors (8, energy, q, debye_temp_factor)
  	print "  Atomic factors (Z = 8) f0, fp, fpp: %f, %f, i*%f" % ( f0, fp, fpp)

  	FH = xraylib.Crystal_F_H_StructureFactor (cryst, energy, 0, 2, 0, debye_temp_factor, rel_angle)
  	print "  FH(0,2,0) structure factor: (%f, %f)" % (FH.real, FH.imag)

  	F0 = xraylib.Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
  	print "  F0=FH(0,0,0) structure factor: (%f, %f)"% (F0.real, F0.imag)

  	#Muscovite diffraction parameters

  	cryst = xraylib.Crystal_GetCrystal("Muscovite")

  	print "\nMuscovite 331 at 8 KeV. Incidence at the Bragg angle:"

  	bragg = xraylib.Bragg_angle (cryst, energy, 3, 3, 1)
  	print "  Bragg angle: Rad: %f Deg: %f" % (bragg, bragg*180/math.pi)

  	q = xraylib.Q_scattering_amplitude (cryst, energy, 3, 3, 1, rel_angle)
  	print "  Q Scattering amplitude: %f"% q

  	f0, fp, fpp =xraylib.Atomic_Factors (19, energy, q, debye_temp_factor);
  	print "  Atomic factors (Z = 19) f0, fp, fpp: %f, %f, i*%f" % (f0, fp, fpp)

  	FH = xraylib.Crystal_F_H_StructureFactor (cryst, energy, 3, 3, 1, debye_temp_factor, rel_angle)
  	print "  FH(3,3,1) structure factor: (%f, %f)" % (FH.real, FH.imag)

  	F0 = xraylib.Crystal_F_H_StructureFactor (cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle)
  	print"  F0=FH(0,0,0) structure factor: (%f, %f)" % (F0.real, F0.imag)

	print ""
	print "--------------------------- END OF XRLEXAMPLE5 -------------------------------"
	print ""
	sys.exit(0)
