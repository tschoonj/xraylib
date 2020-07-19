/*
Copyright (c) 2010, 2011 Tom Schoonjans
Updated to c++11 by Marius Schollmeier
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cmath>
#include <memory>
#include <iostream>
#include <sstream>
#include <vector>

#include <xraylib.h>

using namespace std;

/* A (very) rudimentary error string handler */
string errorstring(int line, xrl_error &err)
{
    stringstream ss;
    ss << "Error at line " << to_string(line) << ": " + string(err.message) << endl;
    return ss.str();
}


int main()
{
    cout << endl <<"------------- Example6: C++ program using xraylib -------------" << endl;

    XRayInit();

    /* Note: For clarity, error handling is shown for only a few examples! */
    xrl_error *err = nullptr;
    try {
        cout << "Density of pure Al: " << ElementDensity(13, &err) << " g/cm3" << endl;
        if (err) throw errorstring(__LINE__, *err);
        cout << "Ca K-alpha Fluorescence Line Energy: " << LineEnergy(20, KA_LINE, &err) << endl;
        cout << "Fe partial photoionization cs of L3 at 6.0 keV: " << CS_Photo_Partial(26, L3_SHELL, 6.0, &err) << endl;
        cout << "Zr L1 edge energy: " << EdgeEnergy(40, L1_SHELL, &err) << endl;
        cout << "Pb Lalpha XRF production cs at 20.0 keV (jump approx): " << CS_FluorLine(82, LA_LINE, 20.0, &err) << endl;
        cout <<"Pb Lalpha XRF production cs at 20.0 keV (Kissel): " << CS_FluorLine_Kissel(82, LA_LINE, 20.0, &err) << endl;
        cout << "Bi M1N2 radiative rate: " << RadRate(83, M1N2_LINE, &err) << endl;
        cout << "U M3O3 Fluorescence Line Energy: " << LineEnergy(92, M3O3_LINE, &err) << endl;

        /*parser test for Ca(HCO3)2 (calcium bicarbonate)*/
        auto cdtest = unique_ptr<compoundData>(CompoundParser("Ca(HCO3)2", &err));
        if (cdtest) {
            cout << "Ca(HCO3)2 contains " << cdtest->nAtomsAll << " atoms, " << cdtest->nElements
                 << " elements and has a molar mass of " << cdtest->molarMass << " g/mol" << endl;
            for (int i = 0 ; i < cdtest->nElements ; i++)
                cout << "  Element "
                     << cdtest->Elements[i] << ": "
                     << cdtest->massFractions[i] * 100.0 << "% and "
                     << cdtest->nAtoms[i] << " atoms" << endl;
        }
        else
            throw errorstring(__LINE__, *err);


        /*parser test for SiO2 (quartz)*/
        cdtest.reset(CompoundParser("SiO2", &err));
        if (cdtest) {
            cout << "SiO2 contains " << cdtest->nAtomsAll << " atoms, " << cdtest->nElements
                 << " elements and has a molar mass of " << cdtest->molarMass << " g/mol" << endl;
            for (int i = 0 ; i < cdtest->nElements ; i++)
                cout << "  Element "
                     << cdtest->Elements[i] << ": "
                     << cdtest->massFractions[i] * 100.0 << "% and "
                     << cdtest->nAtoms[i] << " atoms" << endl;
        }
        else
            throw errorstring(__LINE__, *err);


        cout << "Ca(HCO3)2 Rayleigh cs at 10.0 keV: " << CS_Rayl_CP("Ca(HCO3)2", 10.0, &err) << endl;
        cout << "CS2 Refractive Index at 10.0 keV: " << Refractive_Index_Re("CS2", 10.0, 1.261, &err) << " - " << Refractive_Index_Im("CS2", 10.0, 1.261, &err) << "i" << endl;
        cout << "C16H14O3 Refractive Index at 1 keV:" << Refractive_Index_Re("C16H14O3", 1.0, 1.2, &err)<< " - " << Refractive_Index_Im("C16H14O3", 1.0, 1.2, &err) << "i" << endl;
        cout << "SiO2 Refractive Index at 5 keV: " << Refractive_Index_Re("SiO2", 5.0, 2.65, &err) << " - " << Refractive_Index_Im("SiO2",5.0, 2.65, &err) << "i" << endl;
        cout << "Compton profile for Fe at pz = 1.1: " << ComptonProfile(26, 1.1, &err) << endl;
        cout << "M5 Compton profile for Fe at pz = 1.1:" << ComptonProfile_Partial(26, M5_SHELL, 1.1, &err) << endl;
        cout << "M1->M5 Coster-Kronig transition probability for Au: " << CosKronTransProb(79, FM15_TRANS, &err) << endl;
        cout << "L1->L3 Coster-Kronig transition probability for Fe: " << CosKronTransProb(26, FL13_TRANS, &err) << endl;
        cout << "Au Ma1 XRF production cs at 10.0 keV (Kissel): " << CS_FluorLine_Kissel(79, MA1_LINE, 10.0, &err) << endl;
        cout << "Au Mb XRF production cs at 10.0 keV (Kissel): " << CS_FluorLine_Kissel(79, MB_LINE, 10.0, &err) << endl;
        cout << "Au Mg XRF production cs at 10.0 keV (Kissel): " << CS_FluorLine_Kissel(79, MG_LINE, 10.0, &err) << endl;
        cout << "K atomic level width for Fe: " << AtomicLevelWidth(26, K_SHELL, &err) << endl;
        cout << "Bi L2-M5M5 Auger non-radiative rate: " << AugerRate(86, L2_M5M5_AUGER, &err) << endl;
        cout << "Bi L3 Auger yield: " << AugerYield(86, L3_SHELL, &err) << endl;

        auto cdtest1 = unique_ptr<compoundData>(CompoundParser("SiO2", &err));
        auto cdtest2 = unique_ptr<compoundData>(CompoundParser("Ca(HCO3)2", &err));
        if (cdtest1 && cdtest2) {
            auto cdtest3 = add_compound_data(*cdtest1, 0.4, *cdtest2, 0.6);
            cout << "40% SiO2 and 60% Ca(HCO3)2 contains "
                 << cdtest->nAtomsAll << " atoms, "
                 << cdtest->nElements << " elements and has a molar mass of "
                 << cdtest->molarMass << " g/mol" << endl;

            for (int i = 0 ; i < cdtest3->nElements ; i++)
                cout << "  Element " << cdtest3->Elements[i] << ": "
                     << cdtest3->massFractions[i] * 100.0 << "%" << endl;
        } else
            throw errorstring(__LINE__, *err);


        cout << "Sr anomalous scattering factor Fi at 10.0 keV: " << Fi(38, 10.0, &err) << endl;
        cout << "Sr anomalous scattering factor Fii at 10.0 keV: " << Fii(38, 10.0, &err) << endl;
        cout << "Symbol of element 26 is: " << AtomicNumberToSymbol(26, &err) << endl;
        cout << "Number of element Fe is: " << SymbolToAtomicNumber("Fe", &err) << endl;

        cout << "Pb Malpha XRF production cs at 20.0 keV with cascade effect: " << CS_FluorLine_Kissel(82, MA1_LINE, 20.0, &err) << endl;
        cout << "Pb Malpha XRF production cs at 20.0 keV with radiative cascade effect: " << CS_FluorLine_Kissel_Radiative_Cascade(82, MA1_LINE, 20.0, &err) << endl;
        cout << "Pb Malpha XRF production cs at 20.0 keV with non-radiative cascade effect: " << CS_FluorLine_Kissel_Nonradiative_Cascade(82, MA1_LINE, 20.0, &err) << endl;
        cout << "Pb Malpha XRF production cs at 20.0 keV without cascade effect: " << CS_FluorLine_Kissel_no_Cascade(82, MA1_LINE, 20.0, &err) << endl;

        cout << "Al mass energy-absorption cs at 20.0 keV: " << CS_Energy(13, 20.0, &err) << endl;
        cout << "Pb mass energy-absorption cs at 40.0 keV: " << CS_Energy(82, 40.0, &err) << endl;
        cout << "CdTe mass energy-absorption cs at 40.0 keV: " << CS_Energy_CP("CdTe", 40.0, &err) << endl;

        double energy {8.};
        double rel_angle {1.0};
        double debye_temp_factor {1.0};
        double f0 {0.};
        double fp {0.};
        double fpp {0.};

        /* Si Crystal structure */
        auto cryst = unique_ptr<Crystal_Struct>(Crystal_GetCrystal("Si", nullptr, &err));
        if (cryst) {
            cout << "Si unit cell dimensions are "
                 << cryst->a << " "
                 << cryst->b << " "
                 << cryst->c << endl;
            cout << "Si unit cell angles are "
                 << cryst->alpha << " "
                 << cryst->beta  << " "
                 << cryst->gamma << endl;
            cout << "Si unit cell volume is " << cryst->volume << endl;
            cout << "Si atoms at:" << endl;
            cout << "  Z   fraction\t\tX\t\tY\t\tZ" << endl;
            for (int i = 0; i < cryst->n_atom; i++) {
                auto atom = cryst->atom[i];
                cout << "  " << atom.Zatom
                     << "  " << atom.fraction
                     << "\t\t" << atom.x
                     << "\t\t" << atom.y
                     << "\t\t" << atom.z << endl;
            }

            /* Si diffraction parameters */
            cout << endl;
            cout << "Si111 at 8 KeV. Incidence at the Bragg angle:" << endl;

            auto bragg = Bragg_angle (&*cryst, energy, 1, 1, 1, &err);
            cout << "  Bragg angle: Rad: " << bragg << " Deg: " << bragg * RADEG << endl;

            auto q = Q_scattering_amplitude (&*cryst, energy, 1, 1, 1, rel_angle, &err);
            cout << "  Q Scattering amplitude: " << q << endl;

            Atomic_Factors (14, energy, q, debye_temp_factor, &f0, &fp, &fpp, &err);
            cout << "  Atomic factors (Z = 14) f0, fp, fpp: " << f0 << " " << fp << " i*" << fpp << endl;

            auto FH = Crystal_F_H_StructureFactor (&*cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle, &err);
            cout << "  FH(1,1,1) structure factor: (" << FH.re << ", " << FH.im << ")" << endl;

            auto F0 = Crystal_F_H_StructureFactor (&*cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle, &err);
            cout << "  F0=FH(0,0,0) structure factor: (" << F0.re << ", " << F0.im << ")" << endl;

        } else
            throw errorstring(__LINE__, *err);


        /* Diamond diffraction parameters */
        cryst.reset(Crystal_GetCrystal("Diamond", nullptr, &err));
        if (cryst) {

            cout << endl << "Diamond 111 at 8 KeV. Incidence at the Bragg angle:" << endl;

            auto bragg = Bragg_angle (&*cryst, energy, 1, 1, 1, &err);
            cout << "  Bragg angle: Rad: " << bragg << " Deg: " << bragg * RADEG << endl;

            auto q = Q_scattering_amplitude (&*cryst, energy, 1, 1, 1, rel_angle, &err);
            cout << "  Q Scattering amplitude: " << q << endl;

            Atomic_Factors (6, energy, q, debye_temp_factor, &f0, &fp, &fpp, &err);
            cout << "  Atomic factors (Z = 6) f0, fp, fpp: " << f0 << " " << fp << " i*" << fpp << endl;

            auto FH = Crystal_F_H_StructureFactor (&*cryst, energy, 1, 1, 1, debye_temp_factor, rel_angle, &err);
            cout << "  FH(1,1,1) structure factor: (" << FH.re << ", " << FH.im << ")" << endl;

            auto F0 = Crystal_F_H_StructureFactor (&*cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle, &err);
            cout << "  F0=FH(0,0,0) structure factor: (" << F0.re << ", " << F0.im << ")" << endl;

            auto FHbar = Crystal_F_H_StructureFactor (&*cryst, energy, -1, -1, -1, debye_temp_factor, rel_angle, &err);
            auto dw = 1.0e10 * 2. * (R_E / cryst->volume) * (KEV2ANGST * KEV2ANGST/ (energy * energy)) * sqrt(c_abs(c_mul(FH, FHbar))) / PI / sin(2*bragg);
            cout << "  Darwin width: " << 1.0e6 * dw << " micro-radians" << endl;

        }

        /* Alpha Quartz diffraction parameters */
        cryst.reset(Crystal_GetCrystal("AlphaQuartz", nullptr, &err));
        if (cryst) {
            cout << endl << "Alpha Quartz 020 at 8 KeV. Incidence at the Bragg angle:" << endl;

            auto bragg = Bragg_angle (&*cryst, energy, 0, 2, 0, &err);
            cout << "  Bragg angle: Rad: " << bragg << " Deg: " << bragg * RADEG << endl;

            auto q = Q_scattering_amplitude (&*cryst, energy, 0, 2, 0, rel_angle, &err);
            cout << "  Q Scattering amplitude: " << q << endl;

            Atomic_Factors (8, energy, q, debye_temp_factor, &f0, &fp, &fpp, &err);
            cout << "  Atomic factors (Z = 8) f0, fp, fpp: " << f0 << " " << fp << " i*" << fpp << endl;

            auto FH = Crystal_F_H_StructureFactor (&*cryst, energy, 0, 2, 0, debye_temp_factor, rel_angle, &err);
            cout << "  FH(0,2,0) structure factor: (" << FH.re << ", " << FH.im << ")" << endl;

            auto F0 = Crystal_F_H_StructureFactor (&*cryst, energy, 0, 0, 0, debye_temp_factor, rel_angle, &err);
            cout << "  F0=FH(0,0,0) structure factor: (" << F0.re << ", " << F0.im << ")" << endl;
        }

        /* Muscovite diffraction parameters */
        cryst.reset(Crystal_GetCrystal("Muscovite", nullptr, &err));
        if (cryst) {
            cout << endl << "Muscovite 331 at 8 KeV. Incidence at the Bragg angle:" << endl;

            auto bragg = Bragg_angle (&*cryst, energy, 3, 3, 1, &err);
            cout << "  Bragg angle: Rad: " << bragg << " Deg: " << bragg * RADEG << endl;

            auto q = Q_scattering_amplitude (&*cryst, energy, 3, 3, 1, rel_angle, &err);
            cout << "  Q Scattering amplitude: " << q << endl;

            Atomic_Factors (19, energy, q, debye_temp_factor, &f0, &fp, &fpp, &err);
            cout << "  Atomic factors (Z = 19) f0, fp, fpp: " << f0 << " " << fp << " i*" << fpp << endl;

            auto FH = Crystal_F_H_StructureFactor (&*cryst, energy, 3, 3, 1, debye_temp_factor, rel_angle, &err);
            cout << "  FH(3,3,1) structure factor: (" << FH.re << ", " << FH.im << ")" << endl;

            auto F0 = Crystal_F_H_StructureFactor (&*cryst, energy, 3, 3, 1, debye_temp_factor, rel_angle, &err);
            cout << "  F0=FH(0,0,0) structure factor: (" << F0.re << ", " << F0.im << ")" << endl;
        }

        cout << endl;
        cout << "List of available crystals:" << endl;
        auto crystalNames = Crystal_GetCrystalsList(nullptr, nullptr, &err); //returns array of c strings with unknown length
        for (int i = 0; crystalNames[i] != nullptr; i++) {
            cout << "  Crystal " <<  i << ": " << crystalNames[i] << endl;
            xrlFree(crystalNames[i]);
        }
        xrlFree(crystalNames);
        cout << endl;

        /* compoundDataNIST tests */
        auto cdn = unique_ptr<compoundDataNIST>(GetCompoundDataNISTByName("Uranium Monocarbide", &err));
        if (cdn) {
            cout << "Uranium Monocarbide" << endl;
            cout << "  Name: " << cdn->name << endl;
            cout << "  Density: " << cdn->density << " g/cm3" << endl;
            for (int i = 0; i < cdn->nElements; i++)
                cout << "  Element " << cdn->Elements[i] << ": " << cdn->massFractions[i]*100.0 << "%" << endl;
        }

        cdn.reset(GetCompoundDataNISTByIndex(NIST_COMPOUND_BRAIN_ICRP, &err));
        if (cdn) {
            cout << "NIST_COMPOUND_BRAIN_ICRP" << endl;
            cout << "  Name: " << cdn->name << endl;
            cout << "  Density: " << cdn->density << " g/cm3" << endl;
            for (int i = 0 ; i < cdn->nElements ; i++)
                cout << "  Element " << cdn->Elements[i] << ": " << cdn->massFractions[i]*100.0 << "%" << endl;
        }

        cout << "List of available NIST compounds:" << endl;
        auto nistCompounds = GetCompoundDataNISTList(nullptr, &err);  //returns array of c strings with unknown length
        for (int i = 0; nistCompounds[i] != nullptr; i++) {
            cout << "  Compound " << i << ": " << nistCompounds[i] << endl;
            xrlFree(nistCompounds[i]);
        }
        xrlFree(nistCompounds);

        cout << endl;

        /* radioNuclideData tests */
        auto rnd = unique_ptr<radioNuclideData>(GetRadioNuclideDataByName("109Cd", &err));
        cout << "109Cd" << endl;
        cout << "  Name: " << rnd->name << endl;
        cout << "  Z: " << rnd->Z << endl;
        cout << "  A: " << rnd->A << endl;
        cout << "  N: " << rnd->N << endl;
        cout << "  Z_xray: " << rnd->Z_xray << endl;
        cout << "  X-rays:" << endl;
        for (int i = 0; i < rnd->nXrays; i++)
            cout << "  "
                 << LineEnergy(rnd->Z_xray, rnd->XrayLines[i], &err) << " keV -> "
                 << rnd->XrayIntensities[i] << endl;
        cout << "  Gamma rays:" << endl;
        for (int i = 0; i < rnd->nGammas; i++)
            cout << "  " << rnd->GammaEnergies[i] << " keV -> " << rnd->GammaIntensities[i] << endl;

        rnd.reset(GetRadioNuclideDataByIndex(RADIO_NUCLIDE_125I, &err));
        cout << "RADIO_NUCLIDE_125I" << endl;
                cout << "  Name: " << rnd->name << endl;
        cout << "  Z: " << rnd->Z << endl;
        cout << "  A: " << rnd->A << endl;
        cout << "  N: " << rnd->N << endl;
        cout << "  Z_xray: " << rnd->Z_xray << endl;
        cout << "  X-rays:" << endl;
        for (int i = 0; i < rnd->nXrays; i++)
            cout << "  "
                 << LineEnergy(rnd->Z_xray, rnd->XrayLines[i], &err)
                 << " keV -> " << rnd->XrayIntensities[i] << endl;
        cout << "  Gamma rays:" << endl;
        for (int i = 0; i < rnd->nGammas; i++)
            cout << "  " << rnd->GammaEnergies[i] << " keV -> " << rnd->GammaIntensities[i] << endl;

        auto radioNuclides = GetRadioNuclideDataList(nullptr, &err);
        cout << "List of available radionuclides:" << endl;
        for (int i = 0; radioNuclides[i] != nullptr; i++) {
                    cout << "  Radionuclide " << i << ": " << radioNuclides[i] << endl;
                    xrlFree(radioNuclides[i]);
                }
                xrlFree(radioNuclides);
    }
    catch (string msg) {
        cout << msg.data();
        return 1;

    }
    cout << "--------------------------- END OF XRLEXAMPLE6 -------------------------------" << endl;
    return 0;
}
