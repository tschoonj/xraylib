#include <iostream>

#include "xraylib_test.h"

#include <gtest/gtest.h>

/* build command on iMac:
 * /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ -std=gnu++11 -stdlib=libc++ -I/usr/local/xraylib/include/xraylib -I/usr/local/include -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.15.sdk -L/usr/local/xraylib/lib -lxrl -L/usr/local/lib -lgtest  main.cpp
 */

using namespace std;


TEST(DataLookup, LineEnergy) {

    // check proper return value
    EXPECT_DOUBLE_EQ(8.0478, xrlcpp::LineEnergy(29, KA1_LINE));

    // check that proper exception is thrown
    try {
        xrlcpp::LineEnergy(129, KA1_LINE);
        FAIL() << "Expected std::invalid_argumrnt";
    }
    catch (std::invalid_argument const & err) {
        EXPECT_EQ(err.what(), std::string("Z out of range"));
    }
    catch (...) {
        FAIL() << "Expected std::invalid_argument";
    }
}

TEST(DataLookup, FluorYield) {
    EXPECT_DOUBLE_EQ(0.4538, xrlcpp::FluorYield(29, K_SHELL));
}

TEST(DataLookup, CS_Total_CP) {

    // check proper return value
    EXPECT_DOUBLE_EQ(75.842090088141489, xrlcpp::CS_Total_CP("FeSO4", 10.0));

    // check that proper exception is thrown
    try {
        xrlcpp::CS_Total_CP("FeSO4", -10.0);
        FAIL() << "Expected std::invalid_argument";
    }
    catch (std::invalid_argument const & err) {
        EXPECT_EQ(err.what(), std::string("CS_Total_CPEnergy must be strictly positive"));
    }
    catch (...) {
        FAIL() << "Expected std::invalid_argument";
    }
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


//int main()
//{
//    try {
//    cout << "Line Energy: " << xrlcpp::LineEnergy(29, KA1_LINE) << " keV."  <<  endl;
//    cout << "Flourescence yield: " << xrlcpp::FluorYield(29, K_SHELL) << endl;

//    cout << "Total attenuation cross section of FeSO4 at 10.0 keV: " << xrlcpp::CS_Total_CP("FeSO4", -10.0) << " cm^2/g." << endl;

////    cout << "Symbol of element 26 is: " << AtomicNumberToSymbol(26, &err) << endl;


//    auto compound = xrlcpp::CompoundParser("C17H20N4NaO9P");
//    cout << "C17H20N4NaO9P contains " << compound->nAtomsAll << " atoms, " << compound->nElements
//         << " elements and has a molar mass of " << compound->molarMass << " g/mol" << endl;
//    for (int i = 0 ; i < compound->nElements ; i++)
//        cout << "  Element "
//             << compound->Elements[i] << ": "
//             << compound->massFractions[i] * 100.0 << "% and "
//             << compound->nAtoms[i] << " atoms" << endl;


//    }
//    catch(std::exception( &e)) {
//        cout << endl << "Error! " << e.what() << endl;
//        return 1;
//    }

//    return 0;
//}
