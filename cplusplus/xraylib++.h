

#ifndef XRAYLIB_PLUSPLUS_H
#define XRAYLIB_PLUSPLUS_H

#include <xraylib.h>
#include <stdexcept>
#include <new>
#include <complex>
#include <vector>

    typedef struct compoundData _compoundDataPod;

#define _XRL_FUNCTION_1I(_name) \
    double _name(int arg1) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(arg1, &error); \
        _process_error(error); \
        return rv; \
    }

#define _XRL_FUNCTION_1D(_name) \
    double _name(double arg1) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(arg1, &error); \
        _process_error(error); \
        return rv; \
    }

#define _XRL_FUNCTION_2ID(_name) \
    double _name(int arg1, double arg2) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(arg1, arg2, &error); \
        _process_error(error); \
        return rv; \
    }

#define _XRL_FUNCTION_2DD(_name) \
    double _name(double arg1, double arg2) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(arg1, arg2, &error); \
        _process_error(error); \
        return rv; \
    }

#define _XRL_FUNCTION_2II(_name) \
    double _name(int arg1, int arg2) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(arg1, arg2, &error); \
        _process_error(error); \
        return rv; \
    }

#define _XRL_FUNCTION_3IDD(_name) \
    double _name(int arg1, double arg2, double arg3) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(arg1, arg2, arg3, &error); \
        _process_error(error); \
        return rv; \
    }

#define _XRL_FUNCTION_3IID(_name) \
    double _name(int arg1, int arg2, double arg3) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(arg1, arg2, arg3, &error); \
        _process_error(error); \
        return rv; \
    }

#define _XRL_FUNCTION_3DDD(_name) \
    double _name(double arg1, double arg2, double arg3) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(arg1, arg2, arg3, &error); \
        _process_error(error); \
        return rv; \
    }

#define _XRL_FUNCTION_4IDDD(_name) \
    double _name(int arg1, double arg2, double arg3, double arg4) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(arg1, arg2, arg3, arg4, &error); \
        _process_error(error); \
        return rv; \
    }

#define _XRL_FUNCTION_2SD(_name) \
    double _name(const std::string &arg1, double arg2) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(arg1.c_str(), arg2, &error); \
        _process_error(error); \
        return rv; \
    }

#define _XRL_FUNCTION_3SDD(_name) \
    double _name(const std::string &arg1, double arg2, double arg3) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(arg1.c_str(), arg2, arg3, &error); \
        _process_error(error); \
        return rv; \
    }

#define _XRL_FUNCTION_4SDDD(_name) \
    double _name(const std::string &arg1, double arg2, double arg3, double arg4) { \
        xrl_error *error = nullptr; \
        double rv = ::_name(arg1.c_str(), arg2, arg3, arg4, &error); \
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
            Elements(std::vector<int>(cd->Elements, cd->Elements + cd->nElements)),
            massFractions(std::vector<double>(cd->massFractions, cd->massFractions + cd->nElements)),
            nAtomsAll(cd->nAtomsAll),
            nAtoms(std::vector<double>(cd->nAtoms, cd->nAtoms + cd->nElements)),
            molarMass(cd->molarMass)
        {}
    };

    compoundData CompoundParser(const std::string &compoundString) {
        xrl_error *error = nullptr;
        _compoundDataPod *cd = ::CompoundParser(compoundString.c_str(), &error);
        _process_error(error);
        compoundData rv(cd);
        ::FreeCompoundData(cd);
        return rv;
    }

    _XRL_FUNCTION_1I(AtomicWeight)
    _XRL_FUNCTION_1I(ElementDensity)
    _XRL_FUNCTION_1D(CS_KN)
    _XRL_FUNCTION_1D(DCS_Thoms)
    _XRL_FUNCTION_2ID(CS_Total)
    _XRL_FUNCTION_2ID(CS_Photo)
    _XRL_FUNCTION_2ID(CS_Rayl)
    _XRL_FUNCTION_2ID(CS_Compt)
    _XRL_FUNCTION_2ID(CS_Energy)
    _XRL_FUNCTION_2ID(CSb_Total)
    _XRL_FUNCTION_2ID(CSb_Photo)
    _XRL_FUNCTION_2ID(CSb_Rayl)
    _XRL_FUNCTION_2ID(CSb_Compt)
    _XRL_FUNCTION_2ID(FF_Rayl)
    _XRL_FUNCTION_2ID(SF_Compt)
    _XRL_FUNCTION_2ID(Fi)
    _XRL_FUNCTION_2ID(Fii)
    _XRL_FUNCTION_2ID(CS_Photo_Total)
    _XRL_FUNCTION_2ID(CSb_Photo_Total)
    _XRL_FUNCTION_2ID(CS_Total_Kissel)
    _XRL_FUNCTION_2ID(CSb_Total_Kissel)
    _XRL_FUNCTION_2II(LineEnergy)
    _XRL_FUNCTION_2II(FluorYield)
    _XRL_FUNCTION_2II(CosKronTransProb)
    _XRL_FUNCTION_2II(EdgeEnergy)
    _XRL_FUNCTION_2II(JumpFactor)
    _XRL_FUNCTION_2II(RadRate)
    _XRL_FUNCTION_2DD(DCS_KN)
    _XRL_FUNCTION_2DD(DCSP_Thoms)
    _XRL_FUNCTION_2DD(MomentTransf)
    _XRL_FUNCTION_2DD(ComptonEnergy)
    _XRL_FUNCTION_3IDD(DCS_Rayl)
    _XRL_FUNCTION_3IDD(DCS_Compt)
    _XRL_FUNCTION_3IDD(DCSb_Rayl)
    _XRL_FUNCTION_3IDD(DCSb_Compt)
    _XRL_FUNCTION_3DDD(DCSP_KN)
    _XRL_FUNCTION_3IID(CS_FluorLine)
    _XRL_FUNCTION_3IID(CSb_FluorLine)
    _XRL_FUNCTION_3IID(CS_Photo_Partial)
    _XRL_FUNCTION_3IID(CSb_Photo_Partial)
    _XRL_FUNCTION_4IDDD(DCSP_Rayl)
    _XRL_FUNCTION_4IDDD(DCSP_Compt)
    _XRL_FUNCTION_4IDDD(DCSPb_Rayl)
    _XRL_FUNCTION_4IDDD(DCSPb_Compt)
    _XRL_FUNCTION_2ID(ComptonProfile)
    _XRL_FUNCTION_3IID(ComptonProfile_Partial)
    _XRL_FUNCTION_2II(ElectronConfig)
    _XRL_FUNCTION_2II(AtomicLevelWidth)
    _XRL_FUNCTION_2II(AugerRate)
    _XRL_FUNCTION_2II(AugerYield)
    _XRL_FUNCTION_3IID(CS_FluorLine_Kissel)
    _XRL_FUNCTION_3IID(CSb_FluorLine_Kissel)
    _XRL_FUNCTION_3IID(CS_FluorLine_Kissel_Cascade)
    _XRL_FUNCTION_3IID(CSb_FluorLine_Kissel_Cascade)
    _XRL_FUNCTION_3IID(CS_FluorLine_Kissel_no_Cascade)
    _XRL_FUNCTION_3IID(CSb_FluorLine_Kissel_no_Cascade)
    _XRL_FUNCTION_3IID(CS_FluorLine_Kissel_Nonradiative_Cascade)
    _XRL_FUNCTION_3IID(CSb_FluorLine_Kissel_Nonradiative_Cascade)
    _XRL_FUNCTION_3IID(CS_FluorLine_Kissel_Radiative_Cascade)
    _XRL_FUNCTION_3IID(CSb_FluorLine_Kissel_Radiative_Cascade)

    _XRL_FUNCTION_2SD(CS_Total_CP)
    _XRL_FUNCTION_2SD(CS_Photo_CP)
    _XRL_FUNCTION_2SD(CS_Rayl_CP)
    _XRL_FUNCTION_2SD(CS_Compt_CP)
    _XRL_FUNCTION_2SD(CS_Energy_CP)
    _XRL_FUNCTION_2SD(CSb_Total_CP)
    _XRL_FUNCTION_2SD(CSb_Photo_CP)
    _XRL_FUNCTION_2SD(CSb_Rayl_CP)
    _XRL_FUNCTION_2SD(CSb_Compt_CP)
    _XRL_FUNCTION_3SDD(DCS_Rayl_CP)
    _XRL_FUNCTION_3SDD(DCS_Compt_CP)
    _XRL_FUNCTION_3SDD(DCSb_Rayl_CP)
    _XRL_FUNCTION_3SDD(DCSb_Compt_CP)
    _XRL_FUNCTION_4SDDD(DCSP_Rayl_CP)
    _XRL_FUNCTION_4SDDD(DCSP_Compt_CP)
    _XRL_FUNCTION_4SDDD(DCSPb_Rayl_CP)
    _XRL_FUNCTION_4SDDD(DCSPb_Compt_CP)
    _XRL_FUNCTION_2SD(CS_Photo_Total_CP)
    _XRL_FUNCTION_2SD(CSb_Photo_Total_CP)
    _XRL_FUNCTION_2SD(CS_Total_Kissel_CP)
    _XRL_FUNCTION_2SD(CSb_Total_Kissel_CP)
    _XRL_FUNCTION_3SDD(Refractive_Index_Re)
    _XRL_FUNCTION_3SDD(Refractive_Index_Im)

}

#endif
