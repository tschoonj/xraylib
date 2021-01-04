/*
Copyright (c) 2010, 2011 Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "xraylib++.h"

using namespace std;

int main()
{
  double energy = 8;
  double debye_temp_factor = 1.0;
  double rel_angle = 1.0;

  double bragg, q, dw;
  double f0, fp, fpp;
  std::complex<double> FH, F0, FHbar;
  std::vector<std::string> crystalNames, nistCompounds, radioNuclides;

  XRayInit();

  printf("Example of C++ program using xraylib\n");
  printf("Density of pure Al: %f g/cm3\n", xrlpp::ElementDensity(13));
  printf("Ca K-alpha Fluorescence Line Energy: %f\n", xrlpp::LineEnergy(20, KA_LINE));
  printf("Fe partial photoionization cs of L3 at 6.0 keV: %f\n", xrlpp::CS_Photo_Partial(26, L3_SHELL, 6.0));
  printf("Zr L1 edge energy: %f\n", xrlpp::EdgeEnergy(40, L1_SHELL));
  printf("Pb Lalpha XRF production cs at 20.0 keV (jump approx): %f\n", xrlpp::CS_FluorLine(82, LA_LINE, 20.0));
  printf("Pb Lalpha XRF production cs at 20.0 keV (Kissel): %f\n", xrlpp::CS_FluorLine_Kissel(82, LA_LINE, 20.0));
  printf("Bi M1N2 radiative rate: %f\n", xrlpp::RadRate(83, M1N2_LINE));
  printf("U M3O3 Fluorescence Line Energy: %f\n", xrlpp::LineEnergy(92, M3O3_LINE));
  /*parser test for Ca(HCO3)2 (calcium bicarbonate)*/
  {
    auto cdtest = xrlpp::CompoundParser("Ca(HCO3)2");
    printf("Ca(HCO3)2 contains %g atoms, %i elements and has a molar mass of %g g/mol\n", cdtest.nAtomsAll, cdtest.nElements, cdtest.molarMass);
    for (int i = 0 ; i < cdtest.nElements ; i++)
      printf("Element %i: %f %% and %g atoms\n", cdtest.Elements[i], cdtest.massFractions[i]*100.0, cdtest.nAtoms[i]);
  }

  /*parser test for SiO2 (quartz)*/
  {
    auto cdtest = xrlpp::CompoundParser("SiO2");
    printf("SiO2 contains %g atoms, %i elements and has a molar mass of %g g/mol\n", cdtest.nAtomsAll, cdtest.nElements, cdtest.molarMass);
    for (int i = 0 ; i < cdtest.nElements ; i++)
      printf("Element %i: %f %% and %g atoms\n", cdtest.Elements[i], cdtest.massFractions[i]*100.0, cdtest.nAtoms[i]);
  }

  printf("Ca(HCO3)2 Rayleigh cs at 10.0 keV: %f\n", xrlpp::CS_Rayl_CP("Ca(HCO3)2", 10.0f));

  printf("CS2 Refractive Index at 10.0 keV : %f - %f i\n", xrlpp::Refractive_Index_Re("CS2", 10.0f, 1.261f), xrlpp::Refractive_Index_Im("CS2", 10.0f, 1.261f));
  printf("C16H14O3 Refractive Index at 1 keV : %f - %f i\n", xrlpp::Refractive_Index_Re("C16H14O3", 1.0f, 1.2f), xrlpp::Refractive_Index_Im("C16H14O3", 1.0f, 1.2f));
  printf("SiO2 Refractive Index at 5 keV : %f - %f i\n", xrlpp::Refractive_Index_Re("SiO2", 5.0f, 2.65f), xrlpp::Refractive_Index_Im("SiO2",5.0f, 2.65f));

  printf("Compton profile for Fe at pz = 1.1 : %f\n", xrlpp::ComptonProfile(26, 1.1f));
  printf("M5 Compton profile for Fe at pz = 1.1 : %f\n", xrlpp::ComptonProfile_Partial(26, M5_SHELL, 1.1f));
  printf("M1->M5 Coster-Kronig transition probability for Au : %f\n", xrlpp::CosKronTransProb(79, FM15_TRANS));
  printf("L1->L3 Coster-Kronig transition probability for Fe : %f\n", xrlpp::CosKronTransProb(26, FL13_TRANS));
  printf("Au Ma1 XRF production cs at 10.0 keV (Kissel): %f\n", xrlpp::CS_FluorLine_Kissel(79, MA1_LINE, 10.0f));
  printf("Au Mb XRF production cs at 10.0 keV (Kissel): %f\n", xrlpp::CS_FluorLine_Kissel(79, MB_LINE, 10.0f));
  printf("Au Mg XRF production cs at 10.0 keV (Kissel): %f\n", xrlpp::CS_FluorLine_Kissel(79, MG_LINE, 10.0f));

  printf("K atomic level width for Fe: %f\n", xrlpp::AtomicLevelWidth(26, K_SHELL));
  printf("Bi L2-M5M5 Auger non-radiative rate: %f\n", xrlpp::AugerRate(86, L2_M5M5_AUGER));
  printf("Bi L3 Auger yield: %f\n", xrlpp::AugerYield(86, L3_SHELL));

  printf("Sr anomalous scattering factor Fi at 10.0 keV: %f\n", xrlpp::Fi(38, 10.0));
  printf("Sr anomalous scattering factor Fii at 10.0 keV: %f\n", xrlpp::Fii(38, 10.0));

  auto symbol = xrlpp::AtomicNumberToSymbol(26);
  printf("Symbol of element 26 is: %s\n", symbol.c_str());

  printf("Number of element Fe is: %i\n", xrlpp::SymbolToAtomicNumber("Fe"));

  printf("Pb Malpha XRF production cs at 20.0 keV with cascade effect: %f\n", xrlpp::CS_FluorLine_Kissel(82, MA1_LINE, 20.0));
  printf("Pb Malpha XRF production cs at 20.0 keV with radiative cascade effect: %f\n", xrlpp::CS_FluorLine_Kissel_Radiative_Cascade(82, MA1_LINE, 20.0));
  printf("Pb Malpha XRF production cs at 20.0 keV with non-radiative cascade effect: %f\n", xrlpp::CS_FluorLine_Kissel_Nonradiative_Cascade(82, MA1_LINE, 20.0));
  printf("Pb Malpha XRF production cs at 20.0 keV without cascade effect: %f\n", xrlpp::CS_FluorLine_Kissel_no_Cascade(82, MA1_LINE, 20.0));


  printf("Al mass energy-absorption cs at 20.0 keV: %f\n", xrlpp::CS_Energy(13, 20.0));
  printf("Pb mass energy-absorption cs at 40.0 keV: %f\n", xrlpp::CS_Energy(82, 40.0));
  printf("CdTe mass energy-absorption cs at 40.0 keV: %f\n", xrlpp::CS_Energy_CP("CdTe", 40.0));

  /* Si Crystal structure */

  auto cryst = xrlpp::Crystal::GetCrystal("Si");
  printf ("Si unit cell dimensions are %f %f %f\n", cryst.a, cryst.b, cryst.c);
  printf ("Si unit cell angles are %f %f %f\n", cryst.alpha, cryst.beta, cryst.gamma);
  printf ("Si unit cell volume is %f\n", cryst.volume);
  printf ("Si atoms at:\n");
  printf ("   Z  fraction    X        Y        Z\n");
  for (int i = 0; i < cryst.n_atom; i++) {
    auto atom = cryst.atom[i];
    printf ("  %3i %f %f %f %f\n", atom.Zatom, atom.fraction, atom.x, atom.y, atom.z);
  }

  /* Si diffraction parameters */

  printf ("\nSi111 at 8 KeV. Incidence at the Bragg angle:\n");

  bragg = cryst.Bragg_angle(energy, 1, 1, 1);
  printf ("  Bragg angle: Rad: %f Deg: %f\n", bragg, bragg*180/PI);

  q = cryst.Q_scattering_amplitude(energy, 1, 1, 1, rel_angle);
  printf ("  Q Scattering amplitude: %f\n", q);

  xrlpp::Crystal::Atomic_Factors(14, energy, q, debye_temp_factor, &f0, &fp, &fpp);
  printf ("  Atomic factors (Z = 14) f0, fp, fpp: %f, %f, i*%f\n", f0, fp, fpp);

  FH = cryst.F_H_StructureFactor(energy, 1, 1, 1, debye_temp_factor, rel_angle);
  printf ("  FH(1,1,1) structure factor: (%f, %f)\n", FH.real(), FH.imag());

  F0 = cryst.F_H_StructureFactor(energy, 0, 0, 0, debye_temp_factor, rel_angle);
  printf ("  F0=FH(0,0,0) structure factor: (%f, %f)\n", F0.real(), F0.imag());

  /* Diamond diffraction parameters */

  auto cryst2 = xrlpp::Crystal::GetCrystal("Diamond");

  printf ("\nDiamond 111 at 8 KeV. Incidence at the Bragg angle:\n");

  bragg = xrlpp::Crystal::Bragg_angle(cryst2, energy, 1, 1, 1);
  printf ("  Bragg angle: Rad: %f Deg: %f\n", bragg, bragg*180/PI);

  q = xrlpp::Crystal::Q_scattering_amplitude(cryst2, energy, 1, 1, 1, rel_angle);
  printf ("  Q Scattering amplitude: %f\n", q);

  xrlpp::Crystal::Atomic_Factors(6, energy, q, debye_temp_factor, &f0, &fp, &fpp);
  printf ("  Atomic factors (Z = 6) f0, fp, fpp: %f, %f, i*%f\n", f0, fp, fpp);

  FH = xrlpp::Crystal::F_H_StructureFactor(cryst2, energy, 1, 1, 1, debye_temp_factor, rel_angle);
  printf ("  FH(1,1,1) structure factor: (%f, %f)\n", FH.real(), FH.imag());

  F0 = xrlpp::Crystal::F_H_StructureFactor(cryst2, energy, 0, 0, 0, debye_temp_factor, rel_angle);
  printf ("  F0=FH(0,0,0) structure factor: (%f, %f)\n", F0.real(), F0.imag());

  FHbar = xrlpp::Crystal::F_H_StructureFactor(cryst2, energy, -1, -1, -1, debye_temp_factor, rel_angle);
  dw = 1e10 * 2 * (R_E / cryst.volume) * (KEV2ANGST * KEV2ANGST/ (energy * energy)) *
                                                  sqrt(std::abs(FH * FHbar)) / PI / sin(2*bragg);
  printf ("  Darwin width: %f micro-radians\n", 1e6*dw);

  /* Alpha Quartz diffraction parameters */

  auto cryst3 = xrlpp::Crystal::GetCrystal("AlphaQuartz");

  printf ("\nAlpha Quartz 020 at 8 KeV. Incidence at the Bragg angle:\n");

  bragg = cryst3.Bragg_angle(energy, 0, 2, 0);
  printf ("  Bragg angle: Rad: %f Deg: %f\n", bragg, bragg*180/PI);

  q = cryst3.Q_scattering_amplitude(energy, 0, 2, 0, rel_angle);
  printf ("  Q Scattering amplitude: %f\n", q);

  xrlpp::Crystal::Atomic_Factors(8, energy, q, debye_temp_factor, &f0, &fp, &fpp);
  printf ("  Atomic factors (Z = 8) f0, fp, fpp: %f, %f, i*%f\n", f0, fp, fpp);

  FH = cryst3.F_H_StructureFactor(energy, 0, 2, 0, debye_temp_factor, rel_angle);
  printf ("  FH(0,2,0) structure factor: (%f, %f)\n", FH.real(), FH.imag());

  F0 = cryst3.F_H_StructureFactor(energy, 0, 0, 0, debye_temp_factor, rel_angle);
  printf ("  F0=FH(0,0,0) structure factor: (%f, %f)\n", F0.real(), F0.imag());

  /* Muscovite diffraction parameters */

  auto cryst4 = xrlpp::Crystal::GetCrystal("Muscovite");

  printf ("\nMuscovite 331 at 8 KeV. Incidence at the Bragg angle:\n");

  bragg = cryst4.Bragg_angle(energy, 3, 3, 1);
  printf ("  Bragg angle: Rad: %f Deg: %f\n", bragg, bragg*180/PI);

  q = cryst4.Q_scattering_amplitude(energy, 3, 3, 1, rel_angle);
  printf ("  Q Scattering amplitude: %f\n", q);

  xrlpp::Crystal::Atomic_Factors(19, energy, q, debye_temp_factor, &f0, &fp, &fpp);
  printf ("  Atomic factors (Z = 19) f0, fp, fpp: %f, %f, i*%f\n", f0, fp, fpp);

  FH = cryst4.F_H_StructureFactor(energy, 3, 3, 1, debye_temp_factor, rel_angle);
  printf ("  FH(3,3,1) structure factor: (%f, %f)\n", FH.real(), FH.imag());

  F0 = cryst4.F_H_StructureFactor(energy, 0, 0, 0, debye_temp_factor, rel_angle);
  printf ("  F0=FH(0,0,0) structure factor: (%f, %f)\n", F0.real(), F0.imag());

  crystalNames = xrlpp::Crystal::GetCrystalsList();
  printf ("List of available crystals:\n");
  for (int i = 0 ; i < crystalNames.size() ; i++) {
  	printf ("  Crystal %i: %s\n", i, crystalNames[i].c_str());
  }

  printf ("\n");

  /* compoundDataNIST tests */
  {
    auto cdn = xrlpp::GetCompoundDataNISTByName("Uranium Monocarbide");
    printf ("Uranium Monocarbide\n");
    printf ("  Name: %s\n", cdn.name.c_str());
    printf ("  Density: %f g/cm3\n", cdn.density);
    for (int i = 0 ; i < cdn.nElements ; i++) {
     	printf("  Element %i: %f %%\n", cdn.Elements[i], cdn.massFractions[i]*100.0);
    }
  }

  {
    auto cdn = xrlpp::GetCompoundDataNISTByIndex(NIST_COMPOUND_BRAIN_ICRP);
    printf ("NIST_COMPOUND_BRAIN_ICRP\n");
    printf("  Name: %s\n", cdn.name.c_str());
    printf ("  Density: %f g/cm3\n", cdn.density);
    for (int i = 0 ; i < cdn.nElements ; i++) {
    	printf("  Element %i: %f %%\n", cdn.Elements[i], cdn.massFractions[i]*100.0);
    }
  }

  nistCompounds = xrlpp::GetCompoundDataNISTList();
  printf ("List of available NIST compounds:\n");
  for (int i = 0 ; i < nistCompounds.size() ; i++) {
  	printf ("  Compound %i: %s\n", i, nistCompounds[i].c_str());
  }

  printf ("\n");

  /* radioNuclideData tests */
  {
    auto rnd = xrlpp::GetRadioNuclideDataByName("109Cd");
    printf ("109Cd\n");
    printf ("  Name: %s\n", rnd.name.c_str());
    printf ("  Z: %i\n", rnd.Z);
    printf ("  A: %i\n", rnd.A);
    printf ("  N: %i\n", rnd.N);
    printf ("  Z_xray: %i\n", rnd.Z_xray);
    printf ("  X-rays:\n");
    for (int i = 0 ; i < rnd.nXrays ; i++)
    	printf ("  %f keV -> %f\n", xrlpp::LineEnergy(rnd.Z_xray, rnd.XrayLines[i]), rnd.XrayIntensities[i]);
    printf ("  Gamma rays:\n");
    for (int i = 0 ; i < rnd.nGammas ; i++)
    	printf ("  %f keV -> %f\n", rnd.GammaEnergies[i], rnd.GammaIntensities[i]);
  }
  
  {
    auto rnd = xrlpp::GetRadioNuclideDataByIndex(RADIO_NUCLIDE_125I);
    printf ("RADIO_NUCLIDE_125I\n");
    printf ("  Name: %s\n", rnd.name.c_str());
    printf ("  Z: %i\n", rnd.Z);
    printf ("  A: %i\n", rnd.A);
    printf ("  N: %i\n", rnd.N);
    printf ("  Z_xray: %i\n", rnd.Z_xray);
    printf ("  X-rays:\n");
    for (int i = 0 ; i < rnd.nXrays ; i++)
  	  printf ("  %f keV -> %f\n", xrlpp::LineEnergy(rnd.Z_xray, rnd.XrayLines[i]), rnd.XrayIntensities[i]);
    printf ("  Gamma rays:\n");
    for (int i = 0 ; i < rnd.nGammas ; i++)
  	  printf ("  %f keV -> %f\n", rnd.GammaEnergies[i], rnd.GammaIntensities[i]);
  }

  radioNuclides = xrlpp::GetRadioNuclideDataList();
  printf ("List of available radionuclides:\n");
  for (int i = 0 ; i < radioNuclides.size() ; i++) {
  	printf ("  Radionuclide %i: %s\n", i, radioNuclides[i].c_str());
  }
  printf ("\n--------------------------- END OF XRLEXAMPLE6 -------------------------------\n");
  return 0;
}
