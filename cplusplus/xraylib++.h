/* Copyright (c) 2020, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef XRAYLIB_PLUSPLUS_H
#define XRAYLIB_PLUSPLUS_H

#include <xraylib.h>
#include <stdexcept>
#include <complex>
#include <vector>

using _compoundDataPod = struct compoundData;
using _radioNuclideDataPod = struct radioNuclideData;
using _compoundDataNISTPod = struct compoundDataNIST;


#define _XRL_FUNCTION(_name) \
    template<typename... T> \
    double _name(const std::string &compound, const T... args) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(compound.c_str(), args..., &error); \
        _process_error(error); \
        return rv; \
    } \
    template<typename ...T> \
    double _name(const T... args) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(args..., &error); \
        _process_error(error); \
        return rv; \
    }


namespace xrlpp {
    void _process_error(xrl_error *error) {
        if (!error)
            return;
        switch (error->code) {
            case XRL_ERROR_MEMORY:
                throw std::bad_alloc();
            case XRL_ERROR_INVALID_ARGUMENT:
                throw std::invalid_argument(error->message);
            default:
                throw std::runtime_error(error->message);
        }
    }

    void XrayInit(void) {
        ::XRayInit();
    }

    std::complex<double> Refractive_Index(const std::string &compound, double E, double density) {
        xrl_error *error = nullptr;
        xrlComplex rv = ::Refractive_Index(compound.c_str(), E, density, &error);
        _process_error(error);
        return std::complex<double>(rv.re, rv.im);
    }

    int SymbolToAtomicNumber(const std::string &symbol) {
        xrl_error *error = nullptr;
        int rv = ::SymbolToAtomicNumber(symbol.c_str(), &error);
        _process_error(error);
        return rv;
    }

    std::string AtomicNumberToSymbol(int Z) {
        xrl_error *error = nullptr;
        char *rv = ::AtomicNumberToSymbol(Z, &error);
        _process_error(error);
        std::string rv2(rv);
        ::xrlFree(rv);
        return rv2;
    }

    class compoundData {
        public:
        const int nElements;
        const std::vector<int> Elements;
        const std::vector<double> massFractions;
        const double nAtomsAll;
        const std::vector<double> nAtoms;
        const double molarMass;

        friend compoundData CompoundParser(const std::string &compoundString);

        private:
        compoundData(_compoundDataPod *cd) :
            nElements(cd->nElements), 
            Elements(cd->Elements, cd->Elements + cd->nElements),
            massFractions(cd->massFractions, cd->massFractions + cd->nElements),
            nAtomsAll(cd->nAtomsAll),
            nAtoms(cd->nAtoms, cd->nAtoms + cd->nElements),
            molarMass(cd->molarMass)
        {}
    };

    class radioNuclideData {
        public:
	    const std::string name;
	    const int Z;
	    const int A;
	    const int N;
	    const int Z_xray;
	    const int nXrays;
	    const std::vector<int> XrayLines;
	    const std::vector<double> XrayIntensities;
	    const int nGammas;
	    const std::vector<double> GammaEnergies;
	    const std::vector<double> GammaIntensities;

        friend radioNuclideData GetRadioNuclideDataByName(const std::string &radioNuclideString);
        friend radioNuclideData GetRadioNuclideDataByIndex(int radioNuclideIndex);

        private:
        radioNuclideData(_radioNuclideDataPod *rnd) :
            name(rnd->name),
            Z(rnd->Z),
            A(rnd->A),
            N(rnd->N),
            Z_xray(rnd->Z_xray),
            nXrays(rnd->nXrays),
            XrayLines(rnd->XrayLines, rnd->XrayLines + rnd->nXrays),
            XrayIntensities(rnd->XrayIntensities, rnd->XrayIntensities+ rnd->nXrays),
            nGammas(rnd->nGammas),
            GammaEnergies(rnd->GammaEnergies, rnd->GammaEnergies + rnd->nGammas),
            GammaIntensities(rnd->GammaIntensities, rnd->GammaIntensities + rnd->nGammas)
        {}
    };

    class compoundDataNIST {
        public:
        const std::string name;
        const int nElements;
        const std::vector<int> Elements;
        const std::vector<double> massFractions;
        const double density;

        friend compoundDataNIST GetCompoundDataNISTByName(const std::string &compoundString);
        friend compoundDataNIST GetCompoundDataNISTByIndex(int compoundIndex);

        private:
        compoundDataNIST(_compoundDataNISTPod *cdn) :
            name(cdn->name),
            nElements(cdn->nElements), 
            Elements(cdn->Elements, cdn->Elements + cdn->nElements),
            massFractions(cdn->massFractions, cdn->massFractions + cdn->nElements),
            density(cdn->density)
        {}
    };

    namespace Crystal {
        class Atom {
            public:
            const int Zatom;
            const double fraction;
            const double x, y, z;

            friend std::vector<Atom> _create_atom_vector(Crystal_Atom *atoms, int n_atom);

            private:
            Atom(const Crystal_Atom &atom) :
                Zatom(atom.Zatom),
                fraction(atom.fraction),
                x(atom.x),
                y(atom.y),
                z(atom.z)
            {}
        };

        std::vector<Atom> _create_atom_vector(Crystal_Atom *atoms, int n_atom) {
            std::vector<Atom> rv;

            for (int i = 0 ; i < n_atom ; i++)
                rv.push_back(atoms[i]);

            return rv;
        }

        class Struct {
            public:
            const std::string name;
            const double a, b, c;
            const double alpha, beta, gamma;
            const double volume;
            const int n_atom;
            const std::vector<Atom> atom;

            double Bragg_angle(double energy, int i_miller, int j_miller, int k_miller) {
                xrl_error *error = nullptr;
                double rv = ::Bragg_angle(cs, energy, i_miller, j_miller, k_miller, &error);
                _process_error(error);
                return rv;
            }

            double Q_scattering_amplitude(double energy, int i_miller, int j_miller, int k_miller, double rel_angle) {
                xrl_error *error = nullptr;
                double rv = ::Q_scattering_amplitude(cs, energy, i_miller, j_miller, k_miller, rel_angle, &error);
                _process_error(error);
                return rv;
            }

            std::complex<double> F_H_StructureFactor(double energy, int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle) {
                xrl_error *error = nullptr;
                xrlComplex rv = ::Crystal_F_H_StructureFactor(cs, energy, i_miller, j_miller, k_miller, debye_factor, rel_angle, &error);
                _process_error(error);
                return std::complex<double>(rv.re, rv.im);
            }

            std::complex<double> F_H_StructureFactor_Partial(double energy, int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle, int f0_flag, int f_prime_flag, int f_prime2_flag) {
                xrl_error *error = nullptr;
                xrlComplex rv = ::Crystal_F_H_StructureFactor_Partial(cs, energy, i_miller, j_miller, k_miller, debye_factor, rel_angle, f0_flag, f_prime_flag, f_prime2_flag, &error);
                _process_error(error);
                return std::complex<double>(rv.re, rv.im);
            }

            double UnitCellVolume(void) {
                xrl_error *error = nullptr;
                double rv = ::Crystal_UnitCellVolume(cs, &error);
                _process_error(error);
                return rv;
            }

            double dSpacing(int i_miller, int j_miller, int k_miller) {
                xrl_error *error = nullptr;
                double rv = ::Crystal_dSpacing(cs, i_miller, j_miller, k_miller, &error);
                _process_error(error);
                return rv;
            }

            int AddCrystal(void) {
                xrl_error *error = nullptr;
                int rv = ::Crystal_AddCrystal(cs, nullptr, &error);
                _process_error(error);
                return rv;
            }

            // constructor -> this needs to generate the underlying cs pointer!
            Struct(const std::string &name, double a, double b, double c, double alpha, double beta, double gamma, double volume, const std::vector<Atom> &atoms) :
                name(name),
                a(a),
                b(b),
                c(c),
                alpha(alpha),
                beta(beta),
                gamma(gamma),
                volume(volume),
                n_atom(atoms.size()),
                atom(atoms)
            {
                cs = (Crystal_Struct *) xrl_malloc(sizeof(Crystal_Struct));
                cs->name = xrl_strdup(name.c_str());
                cs->a = a;
                cs->b = b;
                cs->c = c;
                cs->alpha = alpha;
                cs->beta = beta;
                cs->gamma = gamma;
                cs->volume = volume;
                cs->n_atom = n_atom;
                cs->atom = (Crystal_Atom *) xrl_malloc(sizeof(Crystal_Atom) * n_atom);
                for (int i = 0 ; i < n_atom ; i++) {
                    cs->atom[i].Zatom = atoms[i].Zatom;
                    cs->atom[i].fraction = atoms[i].fraction;
                    cs->atom[i].x = atoms[i].x;
                    cs->atom[i].y = atoms[i].y;
                    cs->atom[i].z = atoms[i].z;
                }
            }

            // copy constructor
            Struct(const Struct &_struct) :
                name(_struct.name),
                a(_struct.a),
                b(_struct.b),
                c(_struct.c),
                alpha(_struct.alpha),
                beta(_struct.beta),
                gamma(_struct.gamma),
                volume(_struct.volume),
                n_atom(_struct.n_atom),
                atom(_struct.atom) {

                xrl_error *error = nullptr;
                cs = ::Crystal_MakeCopy(_struct.cs, &error);
                _process_error(error);
            }

            // destructor
            ~Struct() {
                ::Crystal_Free(cs);
            }

            friend Struct GetCrystal(const std::string &material);

            private:
            Crystal_Struct *cs;

            Struct(Crystal_Struct *_struct) :
                name(_struct->name),
                a(_struct->a),
                b(_struct->b),
                c(_struct->c),
                alpha(_struct->alpha),
                beta(_struct->beta),
                gamma(_struct->gamma),
                volume(_struct->volume),
                n_atom(_struct->n_atom),
                atom(_create_atom_vector(_struct->atom, _struct->n_atom)),
                cs(_struct)
            {}

        };

        Struct GetCrystal(const std::string &material) {
            xrl_error *error = nullptr;
            Crystal_Struct *cs = ::Crystal_GetCrystal(material.c_str(), nullptr, &error);
            _process_error(error);
            Struct rv(cs);
            return rv;
        }

        double Bragg_angle(Struct &cs, double energy, int i_miller, int j_miller, int k_miller) {
            return cs.Bragg_angle(energy, i_miller, j_miller, k_miller);
        }

        double Q_scattering_amplitude(Struct &cs, double energy, int i_miller, int j_miller, int k_miller, double rel_angle) {
            return cs.Q_scattering_amplitude(energy, i_miller, j_miller, k_miller, rel_angle);
        }

        int Atomic_Factors(int Z, double energy, double q, double debye_factor, double *f0, double *f_prime, double *f_prime2) {
            xrl_error *error = nullptr;
            int rv = ::Atomic_Factors(Z, energy, q, debye_factor, f0, f_prime, f_prime2, &error);
            _process_error(error);
            return rv;
        }

        std::complex<double> F_H_StructureFactor(Struct &cs, double energy, int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle) {
            return cs.F_H_StructureFactor(energy, i_miller, j_miller, k_miller, debye_factor, rel_angle);
        }

        std::complex<double> F_H_StructureFactor_Partial(Struct &cs, double energy, int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle, int f0_flag, int f_prime_flag, int f_prime2_flag) {
            return cs.F_H_StructureFactor_Partial(energy, i_miller, j_miller, k_miller, debye_factor, rel_angle, f0_flag, f_prime_flag, f_prime2_flag);
        }

        double UnitCellVolume(Struct &cs) {
            return cs.UnitCellVolume();
        }
            
        double dSpacing(Struct &cs, int i_miller, int j_miller, int k_miller) {
            return cs.dSpacing(i_miller, j_miller, k_miller);
        }

        std::vector<std::string> GetCrystalsList(void) {
            std::vector<std::string> rv;
            xrl_error *error = nullptr;
            int nCrystals;
            char **list = ::Crystal_GetCrystalsList(nullptr, &nCrystals, &error);
            _process_error(error);
            for (int i = 0 ; i < nCrystals ; i++) {
                rv.push_back(list[i]);
                ::xrlFree(list[i]);
            }
            ::xrlFree(list);
            return rv;
        }

        int AddCrystal(Struct &cs) {
            return cs.AddCrystal();
        }
    } // end namespace Crystal

    compoundData CompoundParser(const std::string &compoundString) {
        xrl_error *error = nullptr;
        _compoundDataPod *cd = ::CompoundParser(compoundString.c_str(), &error);
        _process_error(error);
        compoundData rv(cd);
        ::FreeCompoundData(cd);
        return rv;
    }

    radioNuclideData GetRadioNuclideDataByName(const std::string &radioNuclideString) {
        xrl_error *error = nullptr;
        _radioNuclideDataPod *rnd = ::GetRadioNuclideDataByName(radioNuclideString.c_str(), &error);
        _process_error(error);
        radioNuclideData rv(rnd);
        ::FreeRadioNuclideData(rnd);
        return rv;
    }
    
    radioNuclideData GetRadioNuclideDataByIndex(int radioNuclideIndex) {
        xrl_error *error = nullptr;
        _radioNuclideDataPod *rnd = ::GetRadioNuclideDataByIndex(radioNuclideIndex, &error);
        _process_error(error);
        radioNuclideData rv(rnd);
        ::FreeRadioNuclideData(rnd);
        return rv;
    }

    std::vector<std::string> GetRadioNuclideDataList(void) {
        std::vector<std::string> rv;
        xrl_error *error = nullptr;
        int nRadioNuclides;
        char **list = ::GetRadioNuclideDataList(&nRadioNuclides, &error);
        _process_error(error);
        for (int i = 0 ; i < nRadioNuclides ; i++) {
            rv.push_back(list[i]);
            ::xrlFree(list[i]);
        }
        ::xrlFree(list);
        return rv;
    }

    compoundDataNIST GetCompoundDataNISTByName(const std::string &compoundString) {
        xrl_error *error = nullptr;
        _compoundDataNISTPod *cdn = ::GetCompoundDataNISTByName(compoundString.c_str(), &error);
        _process_error(error);
        compoundDataNIST rv(cdn);
        ::FreeCompoundDataNIST(cdn);
        return rv;
    }
    
    compoundDataNIST GetCompoundDataNISTByIndex(int compoundIndex) {
        xrl_error *error = nullptr;
        _compoundDataNISTPod *cdn = ::GetCompoundDataNISTByIndex(compoundIndex, &error);
        _process_error(error);
        compoundDataNIST rv(cdn);
        ::FreeCompoundDataNIST(cdn);
        return rv;
    }

    std::vector<std::string> GetCompoundDataNISTList(void) {
        std::vector<std::string> rv;
        xrl_error *error = nullptr;
        int nCompounds;
        char **list = ::GetCompoundDataNISTList(&nCompounds, &error);
        _process_error(error);
        for (int i = 0 ; i < nCompounds; i++) {
            rv.push_back(list[i]);
            ::xrlFree(list[i]);
        }
        ::xrlFree(list);
        return rv;
    }

    /* the macros below are sorted by number and types of input arguments */
    /* 1 int */
    _XRL_FUNCTION(AtomicWeight)
    _XRL_FUNCTION(ElementDensity)
    /* 1 double */
    _XRL_FUNCTION(CS_KN)
    _XRL_FUNCTION(DCS_Thoms)
    /* 2 ints */
    _XRL_FUNCTION(AtomicLevelWidth)
    _XRL_FUNCTION(AugerRate)
    _XRL_FUNCTION(AugerYield)
    _XRL_FUNCTION(CosKronTransProb)
    _XRL_FUNCTION(EdgeEnergy)
    _XRL_FUNCTION(ElectronConfig)
    _XRL_FUNCTION(FluorYield)
    _XRL_FUNCTION(JumpFactor)
    _XRL_FUNCTION(LineEnergy)
    _XRL_FUNCTION(RadRate)
    /* 2 double */
    _XRL_FUNCTION(ComptonEnergy)
    _XRL_FUNCTION(DCS_KN)
    _XRL_FUNCTION(DCSP_Thoms)
    _XRL_FUNCTION(MomentTransf)
    /* 1 int, 1 double*/
    _XRL_FUNCTION(ComptonProfile)
    _XRL_FUNCTION(CS_Compt)
    _XRL_FUNCTION(CS_Energy)
    _XRL_FUNCTION(CS_Photo)
    _XRL_FUNCTION(CS_Photo_Total)
    _XRL_FUNCTION(CS_Rayl)
    _XRL_FUNCTION(CS_Total)
    _XRL_FUNCTION(CS_Total_Kissel)
    _XRL_FUNCTION(CSb_Compt)
    _XRL_FUNCTION(CSb_Photo)
    _XRL_FUNCTION(CSb_Photo_Total)
    _XRL_FUNCTION(CSb_Rayl)
    _XRL_FUNCTION(CSb_Total)
    _XRL_FUNCTION(CSb_Total_Kissel)
    _XRL_FUNCTION(FF_Rayl)
    _XRL_FUNCTION(SF_Compt)
    _XRL_FUNCTION(Fi)
    _XRL_FUNCTION(Fii)
    /* 2 int, 1 double */
    _XRL_FUNCTION(ComptonProfile_Partial)
    _XRL_FUNCTION(CS_FluorLine_Kissel)
    _XRL_FUNCTION(CSb_FluorLine_Kissel)
    _XRL_FUNCTION(CS_FluorLine_Kissel_Cascade)
    _XRL_FUNCTION(CSb_FluorLine_Kissel_Cascade)
    _XRL_FUNCTION(CS_FluorLine_Kissel_no_Cascade)
    _XRL_FUNCTION(CSb_FluorLine_Kissel_no_Cascade)
    _XRL_FUNCTION(CS_FluorLine_Kissel_Nonradiative_Cascade)
    _XRL_FUNCTION(CSb_FluorLine_Kissel_Nonradiative_Cascade)
    _XRL_FUNCTION(CS_FluorLine_Kissel_Radiative_Cascade)
    _XRL_FUNCTION(CSb_FluorLine_Kissel_Radiative_Cascade)
    _XRL_FUNCTION(CS_FluorShell_Kissel)
    _XRL_FUNCTION(CSb_FluorShell_Kissel)
    _XRL_FUNCTION(CS_FluorShell_Kissel_Cascade)
    _XRL_FUNCTION(CSb_FluorShell_Kissel_Cascade)
    _XRL_FUNCTION(CS_FluorShell_Kissel_no_Cascade)
    _XRL_FUNCTION(CSb_FluorShell_Kissel_no_Cascade)
    _XRL_FUNCTION(CS_FluorShell_Kissel_Nonradiative_Cascade)
    _XRL_FUNCTION(CSb_FluorShell_Kissel_Nonradiative_Cascade)
    _XRL_FUNCTION(CS_FluorShell_Kissel_Radiative_Cascade)
    _XRL_FUNCTION(CSb_FluorShell_Kissel_Radiative_Cascade)
    _XRL_FUNCTION(CS_FluorLine)
    _XRL_FUNCTION(CSb_FluorLine)
    _XRL_FUNCTION(CS_FluorShell)
    _XRL_FUNCTION(CSb_FluorShell)
    /* 1 int, 2 double */
    _XRL_FUNCTION(CS_Photo_Partial)
    _XRL_FUNCTION(CSb_Photo_Partial)
    _XRL_FUNCTION(DCS_Compt)
    _XRL_FUNCTION(DCS_Rayl)
    _XRL_FUNCTION(DCSb_Compt)
    _XRL_FUNCTION(DCSb_Rayl)
    /* 1 int 3 double args */
    _XRL_FUNCTION(DCSP_Rayl)
    _XRL_FUNCTION(DCSP_Compt)
    _XRL_FUNCTION(DCSPb_Rayl)
    _XRL_FUNCTION(DCSPb_Compt)
    /* 3 double args */
    _XRL_FUNCTION(DCSP_KN)
    /* 1 string, 1 double */
    _XRL_FUNCTION(CS_Total_CP)
    _XRL_FUNCTION(CS_Photo_CP)
    _XRL_FUNCTION(CS_Rayl_CP)
    _XRL_FUNCTION(CS_Compt_CP)
    _XRL_FUNCTION(CS_Energy_CP)
    _XRL_FUNCTION(CS_Photo_Total_CP)
    _XRL_FUNCTION(CS_Total_Kissel_CP)
    _XRL_FUNCTION(CSb_Total_CP)
    _XRL_FUNCTION(CSb_Photo_CP)
    _XRL_FUNCTION(CSb_Rayl_CP)
    _XRL_FUNCTION(CSb_Compt_CP)
    _XRL_FUNCTION(CSb_Photo_Total_CP)
    _XRL_FUNCTION(CSb_Total_Kissel_CP)
    /* 1 string, 2 double */
    _XRL_FUNCTION(DCS_Rayl_CP)
    _XRL_FUNCTION(DCS_Compt_CP)
    _XRL_FUNCTION(DCSb_Rayl_CP)
    _XRL_FUNCTION(DCSb_Compt_CP)
    _XRL_FUNCTION(Refractive_Index_Re)
    _XRL_FUNCTION(Refractive_Index_Im)
     /* 1 string, 3 double */
    _XRL_FUNCTION(DCSP_Rayl_CP)
    _XRL_FUNCTION(DCSP_Compt_CP)
    _XRL_FUNCTION(DCSPb_Rayl_CP)
    _XRL_FUNCTION(DCSPb_Compt_CP)
}

#endif
