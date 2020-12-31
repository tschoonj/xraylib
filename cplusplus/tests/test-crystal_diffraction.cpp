/* Copyright (c) 2017, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifdef NDEBUG
  #undef NDEBUG
#endif
#define _USE_MATH_DEFINES
#include "xraylib++.h"
#include "xraylib-error-private.h"
#include <cmath>
#include <cassert>
#include <cstring>

int main(int argc, char *argv[]) {

    std::vector<std::string> crystals_list = xrlpp::Crystal::GetCrystalsList();
    assert(crystals_list.size() == 38);

    for (auto crystal_name : crystals_list) {
        auto cs = xrlpp::Crystal::GetCrystal(crystal_name);
        assert(cs.name == crystal_name);
    }

    try {
        xrlpp::Crystal::GetCrystal("non-existent-crystal");
        abort();
    }
    catch (std::invalid_argument &e) {
    }

    auto cs = xrlpp::Crystal::GetCrystal("Diamond");
    auto cs_copy(cs);
    auto *cs_copy2 = new xrlpp::Crystal::Struct(cs_copy);
    delete cs_copy2;

    try {
        xrlpp::Crystal::AddCrystal(cs);
        abort();
    }
    catch (std::invalid_argument &e) {
    }

    try {
        xrlpp::Crystal::AddCrystal(cs_copy);
        abort();
    }
    catch (std::invalid_argument &e) {
    }

    auto *cs_new = new xrlpp::Crystal::Struct("Diamond Copy", cs.a, cs.b, cs.c, cs.alpha, cs.beta, cs.gamma, cs.volume, cs.atom);
    xrlpp::Crystal::AddCrystal(*cs_new);
    delete cs_new;
    crystals_list = xrlpp::Crystal::GetCrystalsList();
    assert(crystals_list.size() == 39);

    double angle = cs.Bragg_angle(10.0, 1, 1, 1);
    assert(fabs(angle - 0.3057795845795849) < 1E-6);

    double tmp = cs.Q_scattering_amplitude(10.0, 1, 1, 1, M_PI_4);
    assert(fabs(tmp - 0.19184445408324474) < 1E-6);

    double f0, f_prime, f_prime2;
    xrlpp::Crystal::Atomic_Factors(26, 10.0, 1.0, 10.0, &f0, &f_prime, &f_prime2);
    assert(fabs(f0 - 65.15) < 1E-6);
    assert(fabs(f_prime + 0.22193271025027966) < 1E-6);
    assert(fabs(f_prime2 - 22.420270655080493) < 1E-6);

    double volume = xrlpp::Crystal::UnitCellVolume(cs);
    assert(fabs(volume - 45.376673902751) < 1E-6);

    double dSpacing = cs.dSpacing(1, 1, 1);
    assert(fabs(dSpacing - 2.0592870875248344) < 1E-6);

    return 0;
}