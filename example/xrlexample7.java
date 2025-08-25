/*Copyright (c) 2010, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

import com.github.tschoonj.xraylib.*;

import org.apache.commons.numbers.complex.Complex;


public class xrlexample7 {
	public static void main(String argv[]) {
		System.out.println("Example of java program using Xraylib");
		System.out.println("Atomic weight of Fe: "+Xraylib.AtomicWeight(26)+" g/mol");
		System.out.println("Density of pure Al: "+Xraylib.ElementDensity(13)+" g/cm3");
		System.out.println("Photoionization cs of Fe at 10.0 keV: "+Xraylib.CS_Photo(26, 10.0)+" cm2/g");
		System.out.println("Rayleigh scattering cs of Fe at 10.0 keV: "+Xraylib.CS_Rayl(26, 10.0)+" cm2/g");
		System.out.println("Compton scattering cs of Fe at 10.0 keV: "+Xraylib.CS_Compt(26, 10.0)+" cm2/g");
		System.out.println("Total cs of Fe at 10.0 keV: "+Xraylib.CS_Total(26, 10.0)+" cm2/g");
		System.out.println("Total cs (Kissel) of Fe at 10.0 keV: "+Xraylib.CS_Total_Kissel(26, 10.0)+" cm2/g");
		System.out.println("Al mass energy-absorption cs at 20.0 keV: "+ Xraylib.CS_Energy(13, 20.0));
		System.out.println("K atomic level width for Fe: "+Xraylib.AtomicLevelWidth(26,Xraylib.K_SHELL) + " keV");
		System.out.println("K fluorescence yield for Fe: "+Xraylib.FluorYield(26,Xraylib.K_SHELL));
		System.out.println("K jumpfactor for Fe: "+Xraylib.JumpFactor(26,Xraylib.K_SHELL));
		System.out.println("M1->M5 Coster-Kronig transition probability for Au : "+Xraylib.CosKronTransProb(79,Xraylib.FM15_TRANS));
		System.out.println("L1->L3 Coster-Kronig transition probability for Fe : "+Xraylib.CosKronTransProb(26,Xraylib.FL13_TRANS));
		System.out.println("Bi M1N2 radiative rate: "+Xraylib.RadRate(83,Xraylib.M1N2_LINE));
		System.out.println("Zr L1 edge energy: " + Xraylib.EdgeEnergy(40, Xraylib.L1_SHELL) + " keV");
		System.out.println("Fe atomic form factor: " + Xraylib.FF_Rayl(26, 1.0));
		System.out.println("Ni scattering form factor: " + Xraylib.SF_Compt(28, 1.0));
		System.out.println("Differential Thomson cross section at 45 deg: " + Xraylib.DCS_Thoms(45.0*Math.PI/180.0) + " cm2/g");
		System.out.println("Differential Klein-Nishina cross section at 10 keV and 45 deg: " + Xraylib.DCS_KN(10.0, 45.0*Math.PI/180.0) + " cm2/g");
		System.out.println("Differential Rayleigh cross section for Zn at 10 keV and 45 deg: " + Xraylib.DCS_Rayl(30, 10.0, 45.0*Math.PI/180.0) + " cm2/g");
		System.out.println("Differential Compton cross section for Zn at 10 keV and 45 deg: " + Xraylib.DCS_Compt(30, 10.0, 45.0*Math.PI/180.0) + " cm2/g");
		System.out.println("Moment transfer function at 10 keV and 45 deg: " + Xraylib.MomentTransf(10.0, 45.0*Math.PI/180.0));
		System.out.println("Klein-Nishina cross section at 10 keV: " + Xraylib.CS_KN(10.0) + " cm2/g");
		System.out.println("Photon energy after Compton scattering at 10 keV and 45 deg angle: " + Xraylib.ComptonEnergy(10.0, 45.0*Math.PI/180.0));
		System.out.println("Photoionization cs of Fe (Kissel) at 10.0 keV: "+Xraylib.CS_Photo_Total(26, 10.0)+" cm2/g");
  		System.out.println("Fe partial photoionization cs of L3 at 6.0 keV: "+Xraylib.CS_Photo_Partial(26,Xraylib.L3_SHELL, 6.0));
		System.out.println("ElectronConfig (Kissel) of Fe L3-shell: " + Xraylib.ElectronConfig(26, Xraylib.L3_SHELL));
		System.out.println("ElectronConfig (Biggs) of Fe L3-shell: " + Xraylib.ElectronConfig_Biggs(26, Xraylib.L3_SHELL));
		System.out.println("Compton profile for Fe at pz = 1.1: "+Xraylib.ComptonProfile(26,(float) 1.1));
		System.out.println("M5 Partial Compton profile for Fe at pz = 1.1: "+Xraylib.ComptonProfile_Partial(26,Xraylib.M5_SHELL, 1.1));
		System.out.println("Bi L2-M5M5 Auger non-radiative rate: "+Xraylib.AugerRate(86,Xraylib.L2_M5M5_AUGER));
		System.out.println("Bi L3 Auger yield: "+Xraylib.AugerYield(86, Xraylib.L3_SHELL));
		System.out.println("Ca K-alpha Fluorescence Line Energy: "+Xraylib.LineEnergy(20,Xraylib.KA_LINE));
		System.out.println("U M3O3 Fluorescence Line Energy: "+Xraylib.LineEnergy(92,Xraylib.M3O3_LINE));
                System.out.println("Pb Lalpha XRF production cs at 20.0 keV (jump approx): "+Xraylib.CS_FluorLine(82, Xraylib.LA_LINE, 20.0));
		System.out.println("Pb Lalpha XRF production cs at 20.0 keV (Kissel): "+Xraylib.CS_FluorLine_Kissel(82, Xraylib.LA_LINE, 20.0));
		System.out.println("Au Ma1 XRF production cs at 10.0 keV (Kissel): "+Xraylib.CS_FluorLine_Kissel(79,Xraylib.MA1_LINE,(float) 10.0));
		System.out.println("Au Mb XRF production cs at 10.0 keV (Kissel): "+Xraylib.CS_FluorLine_Kissel(79,Xraylib.MB_LINE,(float) 10.0));
		System.out.println("Au Mg XRF production cs at 10.0 keV (Kissel): "+Xraylib.CS_FluorLine_Kissel(79,Xraylib.MG_LINE,(float) 10.0));
		System.out.println("Pb Malpha XRF production cs at 20.0 keV with cascade effect: "+Xraylib.CS_FluorLine_Kissel(82,Xraylib.MA1_LINE,(float) 20.0));
		System.out.println("Pb Malpha XRF production cs at 20.0 keV with radiative cascade effect: "+Xraylib.CS_FluorLine_Kissel_Radiative_Cascade(82,Xraylib.MA1_LINE,(float) 20.0));
	System.out.println("Pb Malpha XRF production cs at 20.0 keV with non-radiative cascade effect: "+Xraylib.CS_FluorLine_Kissel_Nonradiative_Cascade(82,Xraylib.MA1_LINE,(float) 20.0));
		System.out.println("Pb Malpha XRF production cs at 20.0 keV without cascade effect: "+Xraylib.CS_FluorLine_Kissel_no_Cascade(82,Xraylib.MA1_LINE,(float) 20.0));
		System.out.println("Sr anomalous scattering factor Fi at 10.0 keV: " + Xraylib.Fi(38, 10.0));
		System.out.println("Sr anomalous scattering factor Fii at 10.0 keV: " + Xraylib.Fii(38, 10.0));
		System.out.println("Symbol of element 26 is: " + Xraylib.AtomicNumberToSymbol(26));
		System.out.println("Number of element Fe is: " + Xraylib.SymbolToAtomicNumber("Fe"));
		System.out.println(Xraylib.CompoundParser("Ca(HCO3)2"));
		System.out.println(Xraylib.CompoundParser("SiO2"));
		System.out.println(Xraylib.CompoundParser("Ca5(PO4)OH"));
		System.out.println(Xraylib.CompoundParser("Fe0.6Mn0.4SiO3"));
		try {
			// the following line should throw an exception
			Xraylib.CompoundParser("CuI2ww");
			System.exit(1);
		}
		catch (IllegalArgumentException e) {
		}
		System.out.println("Total cs of SiO2 at 10.0 keV: "+Xraylib.CS_Total_CP("SiO2", 10.0)+" cm2/g");
		System.out.println("Total cs of SiO2 at 10.0 keV: "+Xraylib.CSb_Total_CP("SiO2", 10.0)+" barns/atom");
		System.out.println("Rayleigh cs of SiO2 at 10.0 keV: "+Xraylib.CS_Rayl_CP("SiO2", 10.0)+" cm2/g");
		System.out.println("Rayleigh cs of SiO2 at 10.0 keV: "+Xraylib.CSb_Rayl_CP("SiO2", 10.0)+" barns/atom");
		System.out.println("Compton cs of SiO2 at 10.0 keV: "+Xraylib.CS_Compt_CP("SiO2", 10.0)+" cm2/g");
		System.out.println("Compton cs of SiO2 at 10.0 keV: "+Xraylib.CSb_Compt_CP("SiO2", 10.0)+" barns/atom");
		System.out.println("Photoionization cs of SiO2 at 10.0 keV: "+Xraylib.CS_Photo_CP("SiO2", 10.0)+" cm2/g");
		System.out.println("Photoionization cs of SiO2 at 10.0 keV: "+Xraylib.CSb_Photo_CP("SiO2", 10.0)+" barns/atom");
		System.out.println("Differential Rayleigh cs of SiO2 at 10.0 keV and 45 deg theta: "+Xraylib.DCS_Rayl_CP("SiO2", 10.0, Math.PI/4.0)+" cm2/g/sterad");
		System.out.println("Differential Rayleigh cs of SiO2 at 10.0 keV and 45 deg theta: "+Xraylib.DCSb_Rayl_CP("SiO2", 10.0, Math.PI/4.0)+" barns/atom/sterad");
		System.out.println("Differential Compton cs of SiO2 at 10.0 keV and 45 deg theta: "+Xraylib.DCS_Compt_CP("SiO2", 10.0, Math.PI/4.0)+" cm2/g/sterad");
		System.out.println("Differential Compton cs of SiO2 at 10.0 keV and 45 deg theta: "+Xraylib.DCSb_Compt_CP("SiO2", 10.0, Math.PI/4.0)+" barns/atom/sterad");
		System.out.println("Polarized differential Rayleigh cs of SiO2 at 10.0 keV and 45 deg theta and 90 deg phi: "+Xraylib.DCSP_Rayl_CP("SiO2", 10.0, Math.PI/4.0, Math.PI/2.0)+" cm2/g/sterad");
		System.out.println("Polarized differential Rayleigh cs of SiO2 at 10.0 keV and 45 deg theta and 90 deg phi: "+Xraylib.DCSPb_Rayl_CP("SiO2", 10.0, Math.PI/4.0, Math.PI/2.0)+" barns/atom/sterad");
		System.out.println("Polarized differential Compton cs of SiO2 at 10.0 keV and 45 deg theta and 90 deg phi: "+Xraylib.DCSP_Compt_CP("SiO2", 10.0, Math.PI/4.0, Math.PI/2.0)+" cm2/g/sterad");
		System.out.println("Polarized differential Compton cs of SiO2 at 10.0 keV and 45 deg theta and 90 deg phi: "+Xraylib.DCSPb_Compt_CP("SiO2", 10.0, Math.PI/4.0, Math.PI/2.0)+" barns/atom/sterad");
		System.out.println("Total cs of Polymethyl Methacralate (Lucite, Perspex) at 10.0 keV: "+Xraylib.CS_Total_CP("Polymethyl Methacralate (Lucite, Perspex)", 10.0)+" cm2/g");

		try {
			// the following line should throw an exception
			double testvalue = Xraylib.DCSb_Compt_CP("SiO2)", 10.0, Math.PI/4.0);
			System.exit(1);
		}
		catch (IllegalArgumentException e) {
		}

  		double energy = 8;
  		double debye_temp_factor = 1.0;
  		double rel_angle = 1.0;
		int i;

  		double bragg, q, dw;
  		double f0 = 0.0, fp = 0.0, fpp = 0.0;
		double[] factors;
  		Complex FH, F0;
  		Complex FHbar;
  		String[] crystalNames;

		/* Si Crystal structure */
		Crystal_Struct cryst = Xraylib.Crystal_GetCrystal("Si");
		System.out.format("Si unit cell dimensions are %f %f %f%n", cryst.a, cryst.b, cryst.c);
		System.out.format("Si unit cell angles are %f %f %f%n", cryst.alpha, cryst.beta, cryst.gamma);
		System.out.format("Si unit cell volume is %f%n", cryst.volume);
		System.out.format("Si atoms at:%n");
		System.out.format("   Z  fraction    X        Y        Z%n");
		for (i = 0; i < cryst.n_atom; i++) {
			Crystal_Atom atom = cryst.atom[i];
    			System.out.format("  %3d %f %f %f %f%n", atom.Zatom, atom.fraction, atom.x, atom.y, atom.z);
  		}

  		/* Si diffraction parameters */

  		System.out.format("%nSi111 at 8 KeV. Incidence at the Bragg angle:%n");

		bragg = Xraylib.Bragg_angle(cryst, energy, 1, 1, 1);
  		System.out.format("  Bragg angle: Rad: %f Deg: %f%n", bragg, bragg*180/Math.PI);

		q = Xraylib.Q_scattering_amplitude (cryst, energy, 1, 1, 1, rel_angle);
		System.out.format("  Q Scattering amplitude: %f%n", q);

  		factors = Xraylib.Atomic_Factors(14, energy, q, debye_temp_factor);
		f0 = factors[0];
		fp = factors[1];
		fpp = factors[2];
		System.out.format("  Atomic factors (Z = 14) f0, fp, fpp: %f, %f, i*%f%n", f0, fp, fpp);

		FH = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle);
  		System.out.format("  FH(1,1,1) structure factor: (%f, %f)%n", FH.getReal(), FH.getImaginary());

		F0 = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
  		System.out.format("  F0=FH(0,0,0) structure factor: (%f, %f)%n", F0.getReal(), F0.getImaginary());


		/* Diamond diffraction parameters */

		cryst = Xraylib.Crystal_GetCrystal("Diamond");

		System.out.format("%nDiamond 111 at 8 KeV. Incidence at the Bragg angle:%n");

		bragg = Xraylib.Bragg_angle(cryst, energy, 1, 1, 1);
		System.out.format("  Bragg angle: Rad: %f Deg: %f%n", bragg, bragg*180/Math.PI);

		q = Xraylib.Q_scattering_amplitude (cryst, energy, 1, 1, 1, rel_angle);
		System.out.format("  Q Scattering amplitude: %f%n", q);

		factors = Xraylib.Atomic_Factors (6, energy, q, debye_temp_factor);
		f0 = factors[0];
		fp = factors[1];
		fpp = factors[2];
		System.out.format("  Atomic factors (Z = 6) f0, fp, fpp: %f, %f, i*%f%n", f0, fp, fpp);

		FH = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle);
		System.out.format("  FH(1,1,1) structure factor: (%f, %f)%n", FH.getReal(), FH.getImaginary());

		F0 = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
		System.out.format("  F0=FH(0,0,0) structure factor: (%f, %f)%n", F0.getReal(), F0.getImaginary());

		FHbar = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, -1, -1, -1, debye_temp_factor, rel_angle);
		dw = 1e10 * 2 * (Xraylib.R_E / cryst.volume) * (Xraylib.KEV2ANGST * Xraylib.KEV2ANGST/ (energy * energy)) * Math.sqrt(FH.multiply(FHbar).abs()) / Math.PI / Math.sin(2.0*bragg);
		System.out.format("  Darwin width: %f micro-radians%n", 1e6*dw);

		/* Alpha Quartz diffraction parameters */
		// Object methods here

		cryst = Xraylib.Crystal_GetCrystal("AlphaQuartz");

		System.out.format("%nAlpha Quartz 020 at 8 KeV. Incidence at the Bragg angle:%n");

		bragg = cryst.Bragg_angle(energy, 0, 2, 0);
		System.out.format("  Bragg angle: Rad: %f Deg: %f%n", bragg, bragg*180/Math.PI);

		q = cryst.Q_scattering_amplitude (energy, 0, 2, 0, rel_angle);
		System.out.format("  Q Scattering amplitude: %f%n", q);

		factors = Xraylib.Atomic_Factors(8, energy, q, debye_temp_factor);
		f0 = factors[0];
		fp = factors[1];
		fpp = factors[2];
		System.out.format("  Atomic factors (Z = 8) f0, fp, fpp: %f, %f, i*%f%n", f0, fp, fpp);

		FH = cryst.Crystal_F_H_StructureFactor(energy, 0, 2, 0, debye_temp_factor, rel_angle);
		System.out.format("  FH(0,2,0) structure factor: (%f, %f)%n", FH.getReal(), FH.getImaginary());

		F0 = cryst.Crystal_F_H_StructureFactor(energy, 0, 0, 0, debye_temp_factor, rel_angle);
		System.out.format("  F0=FH(0,0,0) structure factor: (%f, %f)%n", F0.getReal(), F0.getImaginary());

		/* Muscovite diffraction parameters */

		cryst = Xraylib.Crystal_GetCrystal("Muscovite");

		System.out.format("%nMuscovite 331 at 8 KeV. Incidence at the Bragg angle:%n");

		bragg = Xraylib.Bragg_angle(cryst, energy, 3, 3, 1);
		System.out.format("  Bragg angle: Rad: %f Deg: %f%n", bragg, bragg*180/Math.PI);

		q = Xraylib.Q_scattering_amplitude(cryst, energy, 3, 3, 1, rel_angle);
		System.out.format("  Q Scattering amplitude: %f%n", q);

		factors = Xraylib.Atomic_Factors(19, energy, q, debye_temp_factor);
		f0 = factors[0];
		fp = factors[1];
		fpp = factors[2];
		System.out.format("  Atomic factors (Z = 19) f0, fp, fpp: %f, %f, i*%f%n", f0, fp, fpp);

		FH = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 3, 3, 1, debye_temp_factor, rel_angle);
  		System.out.format("  FH(3,3,1) structure factor: (%f, %f)\n", FH.getReal(), FH.getImaginary());

		F0 = Xraylib.Crystal_F_H_StructureFactor(cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle);
		System.out.format("  F0=FH(0,0,0) structure factor: (%f, %f)%n", F0.getReal(), F0.getImaginary());

		crystalNames = Xraylib.Crystal_GetCrystalsList();
		System.out.format("List of available crystals:%n");
		for (i = 0 ; i < crystalNames.length  ; i++) {
			System.out.format("  Crystal %d: %s%n", i, crystalNames[i]);
  		}

		System.out.format("%n");

		System.out.println("%n" + Xraylib.GetCompoundDataNISTByName("Uranium Monocarbide"));
		System.out.println(Xraylib.GetCompoundDataNISTByIndex(Xraylib.NIST_COMPOUND_BRAIN_ICRP));
    String[] nistCompounds = Xraylib.GetCompoundDataNISTList();

		System.out.println("List of available NIST compounds:");
		for (i = 0 ; i < nistCompounds.length ; i++) {
			System.out.format("  Compound %d: %s%n", i, nistCompounds[i]);
		}

		System.out.println("%n" + Xraylib.GetRadioNuclideDataByName("109Cd"));
		System.out.println(Xraylib.GetRadioNuclideDataByIndex(Xraylib.RADIO_NUCLIDE_125I));
    String[] radioNuclides = Xraylib.GetRadioNuclideDataList();

		System.out.println("List of available radionuclides:");
		for (i = 0 ; i < radioNuclides.length ; i++) {
			System.out.format("  Radionuclide %d: %s%n", i, radioNuclides[i]);
		}

		System.out.println("CS2 Refractive Index at 10.0 keV : "+Xraylib.Refractive_Index_Re("CS2", 10.0, 1.261)+" - "+Xraylib.Refractive_Index_Im("CS2", 10.0, 1.261)+" i");
		System.out.println("C16H14O3 Refractive Index at 1 keV : "+Xraylib.Refractive_Index_Re("C16H14O3", 1.0, 1.2)+" - "+Xraylib.Refractive_Index_Im("C16H14O3", 1.0, 1.2)+" i");
		System.out.println("SiO2 Refractive Index at 5.0 keV : "+Xraylib.Refractive_Index_Re("SiO2", 5.0, 2.65)+" - "+Xraylib.Refractive_Index_Im("SiO2", 5.0, 2.65)+" i");
		Complex refr = Xraylib.Refractive_Index("CS2", 10.0, 1.261);
		System.out.println("CS2 Refractive Index at 10.0 keV : "+ refr.getReal()+" - "+ refr.getImaginary()+" i");

		/*
		System.out.println("Ca(HCO3)2 Rayleigh cs at 10.0 keV: "+Xraylib.CS_Rayl_CP("Ca(HCO3)2",(float) 10.0) );
		System.out.println("Al mass energy-absorption cs at 20.0 keV: "+ Xraylib.CS_Energy(13, (float) 20.0));
		System.out.println("Pb mass energy-absorption cs at 40.0 keV: "+ Xraylib.CS_Energy(82, (float) 40.0));
		System.out.println("CdTe mass energy-absorption cs at 40.0 keV: "+ Xraylib.CS_Energy_CP("CdTe", (float) 40.0));
		*/
		System.out.println("");
		System.out.println("--------------------------- END OF XRLEXAMPLE7 -------------------------------");
		System.out.println("");
		System.exit(0);
	}
}
