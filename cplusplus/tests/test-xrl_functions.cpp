/* Copyright (c) 2020, Tom Schoonjans, Marius Schollmeier
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
#include "xraylib++.h"
#include "xraylib-error-private.h"
#include <cmath>
#include <cassert>
#include <cstring>

int main(int argc, char **argv) {

    /* Testing arg types of the _XRL_FUNCTION macro.
     * One example for each case.
     */

    double res {};

    /* begin with 1 int function, e.g. Atomic Weight */
    res = xrlpp::AtomicWeight(26);
    assert(fabs(res - 55.850) < 1E-6);

    res = xrlpp::AtomicWeight(92);
    assert(fabs(res - 238.070) < 1E-6);

    try {
        xrlpp::AtomicWeight(185);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), Z_OUT_OF_RANGE) == 0);
    }


    /*  1 double */
    res = xrlpp::CS_KN(10.0);
    assert(fabs(res - 0.64047) < 1.0e-6);

    try {
        xrlpp::CS_KN(-5.0);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_ENERGY) == 0);
    }


    /* 2 ints */
    res = xrlpp::AtomicLevelWidth(26, K_SHELL);
    assert(fabs(res - 1.19E-3) < 1E-6);

    res = xrlpp::AtomicLevelWidth(92, N7_SHELL);
    assert(fabs(res - 0.31E-3) < 1E-8);

    try {
        xrlpp::AtomicLevelWidth(185, K_SHELL);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), Z_OUT_OF_RANGE) == 0);
    }

    try {
        xrlpp::AtomicLevelWidth(26, -5);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), UNKNOWN_SHELL) == 0);
    }

    try {
        xrlpp::AtomicLevelWidth(26, N3_SHELL);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), INVALID_SHELL) == 0);
    }

    /* 2 double */
    res = xrlpp::DCS_KN(10, 1.5707964);
    assert(fabs(res - 0.0382088) < 1.0e-6);

    try {
        xrlpp::DCS_KN(-10, 1.5707964);
    }
    catch (std::invalid_argument &e)
    {
        assert(strcmp(e.what(), NEGATIVE_ENERGY) == 0);
    }

    /* 1 int, 1 double */
    res = xrlpp::CS_Total(26, 10.0);
    assert(fabs(res - 170.6911133371) < 1.0e-6);

    try {
       xrlpp::CS_Total(26, -10);
       abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_ENERGY) == 0);
    }

    try {
        xrlpp::CS_Total(185, 10.0);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), Z_OUT_OF_RANGE) == 0);
    }


    /* 2 int, 1 double */
    res = xrlpp::CS_FluorLine_Kissel_no_Cascade(26, KA1_LINE, 10);
    assert(fabs(res - 30.87830762) < 1.0e-6);

    try {
        xrlpp::CS_FluorLine_Kissel_no_Cascade(185, KA1_LINE, 10);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), Z_OUT_OF_RANGE) == 0);
    }

    try {
        xrlpp::CS_FluorLine_Kissel_no_Cascade(26, LV_LINE, 10);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), INVALID_LINE) == 0);
    }

    try {
        xrlpp::CS_FluorLine_Kissel_no_Cascade(26, KA1_LINE, -5.0);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_ENERGY) == 0);
    }

    /* 1 int, 2 double */
    res = xrlpp::DCS_Rayl(26, 10.0, 0.7853981633974483);
    assert(fabs(res - 0.17394691) < 1.0e-6);

    try {
        xrlpp::DCS_Rayl(185, 10.0, 0.7853981633974483);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), Z_OUT_OF_RANGE) == 0);
    }

    try {
        xrlpp::DCS_Rayl(26, -10.0, 0.7853981633974483);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_ENERGY) == 0);
    }

    try {
        xrlpp::DCS_Rayl(26, 10.0, -5.);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_Q) == 0);
    }

    /* 1 int, 3 double */
    res = xrlpp::DCSP_Rayl(26, 10.0, 1.5707963267948966, 0.7853981633974483);
    assert(fabs(res - 0.044355521) < 1.0e-9);

    try {
        xrlpp::DCSP_Rayl(185, 10.0, 1.5707963267948966, 0.7853981633974483);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), Z_OUT_OF_RANGE) == 0);
    }

    try {
        xrlpp::DCSP_Rayl(26, -10.0, 1.5707963267948966, 0.7853981633974483);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_ENERGY) == 0);
    }

    try {
        xrlpp::DCSP_Rayl(26, 10.0, -5., 0.7853981633974483);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_Q) == 0);
    }

    /* 3 double */
    res = xrlpp::DCSP_KN(10., 1.5707964, 3.14159);
    assert(fabs(res - 1.4346407663864e-5) < 1.0e-12);

    try {
        xrlpp::DCSP_KN(-10., 1.5707964, 3.14159);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_ENERGY) == 0);
    }

    /* 1 string, 1 double */
    res = xrlpp::CS_Total_CP("FeSO4", 10.0);
    assert(fabs(res - 75.8420901) < 1.0e-6);

    try {
        xrlpp::CS_Total_CP("Auu1", 10.0);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), UNKNOWN_COMPOUND) == 0);
    }

    try {
        xrlpp::CS_Total_CP("FeSO4", -10.0);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_ENERGY) == 0);
    }

    /* 1 string, 2 double */
    res = xrlpp::Refractive_Index_Re("H2O", 1.0, 1.0);
    assert(fabs(res - 0.999763450676632) < 1E-9);

    res = xrlpp::Refractive_Index_Re(std::string("H2O"), 1.0, 1.0);
    assert(fabs(res - 0.999763450676632) < 1E-9);

    try {
        xrlpp::Refractive_Index_Re("", 1.0, 1.0);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), UNKNOWN_COMPOUND) == 0);
    }

    try {
        xrlpp::Refractive_Index_Re("H2O", 0.0, 1.0);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_ENERGY) == 0);
    }

    try {
        xrlpp::Refractive_Index_Re("H2O", 1.0, 0.0);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_DENSITY) == 0);
    }

    /* 1 string, 3 double */
    res = xrlpp::DCSP_Rayl_CP("FeSO4", 10., 1.5707964, 3.14159);
    assert(fabs(res - 3.59519e-13) < 1.0e-18);

    try {
        xrlpp::DCSP_Rayl_CP("Auu1", 10., 1.5707964, 3.14159);
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), UNKNOWN_COMPOUND) == 0);
    }

    try {
        xrlpp::DCSP_Rayl_CP("FeSO4", -10., 1.5707964, 3.14159);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_ENERGY) == 0);
    }

    try {
        xrlpp::DCSP_Rayl_CP("FeSO4", 10., -1.5707964, 3.14159);
        abort();
    }
    catch (std::invalid_argument &e) {
        assert(strcmp(e.what(), NEGATIVE_Q) == 0);
    }

//    try {
//        xrlpp::DCSP_Rayl_CP("FeSO4", 10., 1.5707964, -3.14159);
//        abort();
//    }
//    catch (std::invalid_argument &e) {
//        assert(strcmp(e.what(), "") == 0);
//    }


    return 0;
}

