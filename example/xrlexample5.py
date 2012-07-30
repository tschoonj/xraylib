#Copyright (c) 2009, 2010, 2011 Tom Schoonjans
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#from _xraylib import *
import sys, string
import _xraylib


if __name__ == '__main__' :
	_xraylib.XRayInit()
	print "Example of python program using xraylib"
	print "Ca K-alpha Fluorescence Line Energy: %f" % _xraylib.LineEnergy(20,_xraylib.KA_LINE)
	print "Fe partial photoionization cs of L3 at 6.0 keV: %f" % _xraylib.CS_Photo_Partial(26,_xraylib.L3_SHELL,6.0)
	print "Zr L1 edge energy: %f" % _xraylib.EdgeEnergy(40,_xraylib.L1_SHELL)
	print "Pb Lalpha XRF production cs at 20.0 keV (jump approx): %f" % _xraylib.CS_FluorLine(82,_xraylib.LA_LINE,20.0)
	print "Pb Lalpha XRF production cs at 20.0 keV (Kissel): %f" % _xraylib.CS_FluorLine_Kissel(82,_xraylib.LA_LINE,20.0)
  	print "Bi M1N2 radiative rate: %f" % _xraylib.RadRate(83,_xraylib.M1N2_LINE)
	print "U M3O3 Fluorescence Line Energy: %f" % _xraylib.LineEnergy(92,_xraylib.M3O3_LINE)
	print "Ca(HCO3)2 Rayleigh cs at 10.0 keV: %f" % _xraylib.CS_Rayl_CP("Ca(HCO3)2",10.0)
	print "CS2 Refractive Index at 10.0 keV : %g - %g i" % (_xraylib.Refractive_Index_Re("CS2",10.0,1.261),_xraylib.Refractive_Index_Im("CS2",10.0,1.261))
	print "C16H14O3 Refractive Index at 1 keV : %g - %g i" % (_xraylib.Refractive_Index_Re("C16H14O3",1.0,1.2),_xraylib.Refractive_Index_Im("C16H14O3",1.0,1.2))
	print "SiO2 Refractive Index at 5 keV : %g - %g i" % (_xraylib.Refractive_Index_Re("SiO2",5.0,2.65),_xraylib.Refractive_Index_Im("SiO2",5.0,2.65))
	print "Compton profile for Fe at pz = 1.1 : %g" % _xraylib.ComptonProfile(26,1.1)
	print "M5 Compton profile for Fe at pz = 1.1 : %g" % _xraylib.ComptonProfile_Partial(26,_xraylib.M5_SHELL,1.1)
	print "Bi L2-M5M5 Auger non-radiative rate: %g" % _xraylib.AugerRate(86,_xraylib.L2_M5M5_AUGER)
	print "K atomic level width for Fe: %g" % _xraylib.AtomicLevelWidth(26,_xraylib.K_SHELL)
	print "M1->M5 Coster-Kronig transition probability for Au : %f" % _xraylib.CosKronTransProb(79,_xraylib.FM15_TRANS)
	print "L1->L3 Coster-Kronig transition probability for Fe : %f" % _xraylib.CosKronTransProb(26,_xraylib.FL13_TRANS)
	print "Au Ma1 XRF production cs at 10.0 keV (Kissel): %f" % _xraylib.CS_FluorLine_Kissel(79,_xraylib.MA1_LINE,10.0)
	print "Au Mb XRF production cs at 10.0 keV (Kissel): %f" % _xraylib.CS_FluorLine_Kissel(79,_xraylib.MB_LINE,10.0)
	print "Au Mg XRF production cs at 10.0 keV (Kissel): %f" % _xraylib.CS_FluorLine_Kissel(79,_xraylib.MG_LINE,10.0)
	print "Pb Malpha XRF production cs at 20.0 keV with cascade effect: %g" % _xraylib.CS_FluorLine_Kissel(82,_xraylib.MA1_LINE,20.0)
	print "Pb Malpha XRF production cs at 20.0 keV with radiative cascade effect: %g" % _xraylib.CS_FluorLine_Kissel_Radiative_Cascade(82,_xraylib.MA1_LINE,20.0)
	print "Pb Malpha XRF production cs at 20.0 keV with non-radiative cascade effect: %g" % _xraylib.CS_FluorLine_Kissel_Nonradiative_Cascade(82,_xraylib.MA1_LINE,20.0)
	print "Pb Malpha XRF production cs at 20.0 keV without cascade effect: %g" % _xraylib.CS_FluorLine_Kissel_no_Cascade(82,_xraylib.MA1_LINE,20.0)
	print ""
	print "--------------------------- END OF XRLEXAMPLE5 -------------------------------"
	print ""
	sys.exit(0)
