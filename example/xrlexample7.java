/*Copyright (c) 2010, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


public class xrlexample7 {
	static {
		System.loadLibrary("xraylib");
	}


	public static void main(String argv[]) {
		xraylib.XRayInit();
		xraylib.SetHardExit(1);
		System.out.println("Example of java program using xraylib");
		System.out.println("Ca K-alpha Fluorescence Line Energy: "+xraylib.LineEnergy(20,xraylib.KA_LINE));
  		System.out.println("Fe partial photoionization cs of L3 at 6.0 keV: "+xraylib.CS_Photo_Partial(26,xraylib.L3_SHELL,(float) 6.0));
		System.out.println("Zr L1 edge energy: "+xraylib.EdgeEnergy(40,xraylib.L1_SHELL));
		System.out.println("Pb Lalpha XRF production cs at 20.0 keV (jump approx): "+xraylib.CS_FluorLine(82,xraylib.LA_LINE,(float) 20.0));
		System.out.println("Pb Lalpha XRF production cs at 20.0 keV (Kissel): "+xraylib.CS_FluorLine_Kissel(82,xraylib.LA_LINE,(float) 20.0));
		System.out.println("Bi M1N2 radiative rate: "+xraylib.RadRate(83,xraylib.M1N2_LINE));
		System.out.println("U M3O3 Fluorescence Line Energy: "+xraylib.LineEnergy(92,xraylib.M3O3_LINE));
		System.out.println("Ca(HCO3)2 Rayleigh cs at 10.0 keV: "+xraylib.CS_Rayl_CP("Ca(HCO3)2",(float) 10.0) );
		System.out.println("CS2 Refractive Index at 10.0 keV : "+xraylib.Refractive_Index_Re("CS2",(float) 10.0,(float) 1.261)+" - "+xraylib.Refractive_Index_Im("CS2",(float) 10.0,(float) 1.261)+" i");  
		System.out.println("C16H14O3 Refractive Index at 1 keV : "+xraylib.Refractive_Index_Re("C16H14O3",(float) 1.0,(float) 1.2)+" - "+xraylib.Refractive_Index_Im("C16H14O3",(float) 1.0,(float) 1.2)+" i");
		System.out.println("SiO2 Refractive Index at 5.0 keV : "+xraylib.Refractive_Index_Re("SiO2",(float) 5.0,(float) 2.65)+" - "+xraylib.Refractive_Index_Im("SiO2",(float) 5.0,(float) 2.65)+" i");  
		System.out.println("Compton profile for Fe at pz = 1.1: "+xraylib.ComptonProfile(26,(float) 1.1));
		System.exit(0);
	}
}


