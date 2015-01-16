
unit xraylib;
interface
uses strings;


{
  Automatically converted by H2Pas 1.0.0 from xraylib.h
  The following command line parameters were used:
    -D
    xraylib.h
}

  const
    External_library='libxrl.7.dylib'; {Setup as you need}

  type
  Pchar  = ^char;
  type
    xrlComplex = record
        re : double;
        im : double;
      end;
{$IFDEF FPC}
{$PACKRECORDS C}
{$ENDIF}


  {
  Copyright (c) 2009, 2010, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen
  All rights reserved.
  
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
      * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.
  
  THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS F
   }
{ C++ extern C conditionnal removed }

  const
    XRAYLIB_MAJOR = 3;    
    XRAYLIB_MINOR = 1;    
    XRAYLIB_MICRO = 0;    
{$ifndef PI}

  const
    PI = 3.1415926535897932384626433832795;    
{$endif}
{$ifndef TWOPI}

  const
    TWOPI = 2*PI;    
{$endif}

  const
    RADEG = 180.0/PI;    
    DEGRAD = PI/180.0;    
  {
   *
   * values taken from physics.nist.gov
   *
    }
  { Avogadro number (mol-1 * barn-1 * cm2)  }    AVOGNUM = 0.602214129;    
  { keV to angstrom-1 conversion factor  }
    KEV2ANGST = 12.39841930;    
  { electron rest mass (keV)  }
    MEC2 = 510.998928;    
  { square of classical electron radius (barn)  }
    RE2 = 0.079407877;    
  { Classical electron radius (m)  }
    R_E = 2.8179403267e-15;    
  {
   * Siegbahn notation
   * according to Table VIII.2 from Nomenclature system for X-ray spectroscopy
   * Linegroups -> usage is discouraged
   *
    }

  const
    K_SHELL = 0;    
    L1_SHELL = 1;    
    L2_SHELL = 2;    
    L3_SHELL = 3;    
    M1_SHELL = 4;    
    M2_SHELL = 5;    
    M3_SHELL = 6;    
    M4_SHELL = 7;    
    M5_SHELL = 8;    
    N1_SHELL = 9;    
    N2_SHELL = 10;    
    N3_SHELL = 11;    
    N4_SHELL = 12;    
    N5_SHELL = 13;    
    N6_SHELL = 14;    
    N7_SHELL = 15;    
    O1_SHELL = 16;    
    O2_SHELL = 17;    
    O3_SHELL = 18;    
    O4_SHELL = 19;    
    O5_SHELL = 20;    
    O6_SHELL = 21;    
    O7_SHELL = 22;    
    P1_SHELL = 23;    
    P2_SHELL = 24;    
    P3_SHELL = 25;    
    P4_SHELL = 26;    
    P5_SHELL = 27;    
    Q1_SHELL = 28;    
    Q2_SHELL = 29;    
    Q3_SHELL = 30;    
    KA_LINE = 0;    
    KB_LINE = 1;    
    LA_LINE = 2;    
    LB_LINE = 3;    
  { single lines  }
    KL1_LINE = -(1);    
    KL2_LINE = -(2);    
    KL3_LINE = -(3);    
    KM1_LINE = -(4);    
    KM2_LINE = -(5);    
    KM3_LINE = -(6);    
    KM4_LINE = -(7);    
    KM5_LINE = -(8);    
    KN1_LINE = -(9);    
    KN2_LINE = -(10);    
    KN3_LINE = -(11);    
    KN4_LINE = -(12);    
    KN5_LINE = -(13);    
    KN6_LINE = -(14);    
    KN7_LINE = -(15);    
    KO_LINE = -(16);    
    KO1_LINE = -(17);    
    KO2_LINE = -(18);    
    KO3_LINE = -(19);    
    KO4_LINE = -(20);    
    KO5_LINE = -(21);    
    KO6_LINE = -(22);    
    KO7_LINE = -(23);    
    KP_LINE = -(24);    
    KP1_LINE = -(25);    
    KP2_LINE = -(26);    
    KP3_LINE = -(27);    
    KP4_LINE = -(28);    
    KP5_LINE = -(29);    
    L1L2_LINE = -(30);    
    L1L3_LINE = -(31);    
    L1M1_LINE = -(32);    
    L1M2_LINE = -(33);    
    L1M3_LINE = -(34);    
    L1M4_LINE = -(35);    
    L1M5_LINE = -(36);    
    L1N1_LINE = -(37);    
    L1N2_LINE = -(38);    
    L1N3_LINE = -(39);    
    L1N4_LINE = -(40);    
    L1N5_LINE = -(41);    
    L1N6_LINE = -(42);    
    L1N67_LINE = -(43);    
    L1N7_LINE = -(44);    
    L1O1_LINE = -(45);    
    L1O2_LINE = -(46);    
    L1O3_LINE = -(47);    
    L1O4_LINE = -(48);    
    L1O45_LINE = -(49);    
    L1O5_LINE = -(50);    
    L1O6_LINE = -(51);    
    L1O7_LINE = -(52);    
    L1P1_LINE = -(53);    
    L1P2_LINE = -(54);    
    L1P23_LINE = -(55);    
    L1P3_LINE = -(56);    
    L1P4_LINE = -(57);    
    L1P5_LINE = -(58);    
    L2L3_LINE = -(59);    
    L2M1_LINE = -(60);    
    L2M2_LINE = -(61);    
    L2M3_LINE = -(62);    
    L2M4_LINE = -(63);    
    L2M5_LINE = -(64);    
    L2N1_LINE = -(65);    
    L2N2_LINE = -(66);    
    L2N3_LINE = -(67);    
    L2N4_LINE = -(68);    
    L2N5_LINE = -(69);    
    L2N6_LINE = -(70);    
    L2N7_LINE = -(71);    
    L2O1_LINE = -(72);    
    L2O2_LINE = -(73);    
    L2O3_LINE = -(74);    
    L2O4_LINE = -(75);    
    L2O5_LINE = -(76);    
    L2O6_LINE = -(77);    
    L2O7_LINE = -(78);    
    L2P1_LINE = -(79);    
    L2P2_LINE = -(80);    
    L2P23_LINE = -(81);    
    L2P3_LINE = -(82);    
    L2P4_LINE = -(83);    
    L2P5_LINE = -(84);    
    L2Q1_LINE = -(85);    
    L3M1_LINE = -(86);    
    L3M2_LINE = -(87);    
    L3M3_LINE = -(88);    
    L3M4_LINE = -(89);    
    L3M5_LINE = -(90);    
    L3N1_LINE = -(91);    
    L3N2_LINE = -(92);    
    L3N3_LINE = -(93);    
    L3N4_LINE = -(94);    
    L3N5_LINE = -(95);    
    L3N6_LINE = -(96);    
    L3N7_LINE = -(97);    
    L3O1_LINE = -(98);    
    L3O2_LINE = -(99);    
    L3O3_LINE = -(100);    
    L3O4_LINE = -(101);    
    L3O45_LINE = -(102);    
    L3O5_LINE = -(103);    
    L3O6_LINE = -(104);    
    L3O7_LINE = -(105);    
    L3P1_LINE = -(106);    
    L3P2_LINE = -(107);    
    L3P23_LINE = -(108);    
    L3P3_LINE = -(109);    
    L3P4_LINE = -(110);    
    L3P45_LINE = -(111);    
    L3P5_LINE = -(112);    
    L3Q1_LINE = -(113);    
    M1M2_LINE = -(114);    
    M1M3_LINE = -(115);    
    M1M4_LINE = -(116);    
    M1M5_LINE = -(117);    
    M1N1_LINE = -(118);    
    M1N2_LINE = -(119);    
    M1N3_LINE = -(120);    
    M1N4_LINE = -(121);    
    M1N5_LINE = -(122);    
    M1N6_LINE = -(123);    
    M1N7_LINE = -(124);    
    M1O1_LINE = -(125);    
    M1O2_LINE = -(126);    
    M1O3_LINE = -(127);    
    M1O4_LINE = -(128);    
    M1O5_LINE = -(129);    
    M1O6_LINE = -(130);    
    M1O7_LINE = -(131);    
    M1P1_LINE = -(132);    
    M1P2_LINE = -(133);    
    M1P3_LINE = -(134);    
    M1P4_LINE = -(135);    
    M1P5_LINE = -(136);    
    M2M3_LINE = -(137);    
    M2M4_LINE = -(138);    
    M2M5_LINE = -(139);    
    M2N1_LINE = -(140);    
    M2N2_LINE = -(141);    
    M2N3_LINE = -(142);    
    M2N4_LINE = -(143);    
    M2N5_LINE = -(144);    
    M2N6_LINE = -(145);    
    M2N7_LINE = -(146);    
    M2O1_LINE = -(147);    
    M2O2_LINE = -(148);    
    M2O3_LINE = -(149);    
    M2O4_LINE = -(150);    
    M2O5_LINE = -(151);    
    M2O6_LINE = -(152);    
    M2O7_LINE = -(153);    
    M2P1_LINE = -(154);    
    M2P2_LINE = -(155);    
    M2P3_LINE = -(156);    
    M2P4_LINE = -(157);    
    M2P5_LINE = -(158);    
    M3M4_LINE = -(159);    
    M3M5_LINE = -(160);    
    M3N1_LINE = -(161);    
    M3N2_LINE = -(162);    
    M3N3_LINE = -(163);    
    M3N4_LINE = -(164);    
    M3N5_LINE = -(165);    
    M3N6_LINE = -(166);    
    M3N7_LINE = -(167);    
    M3O1_LINE = -(168);    
    M3O2_LINE = -(169);    
    M3O3_LINE = -(170);    
    M3O4_LINE = -(171);    
    M3O5_LINE = -(172);    
    M3O6_LINE = -(173);    
    M3O7_LINE = -(174);    
    M3P1_LINE = -(175);    
    M3P2_LINE = -(176);    
    M3P3_LINE = -(177);    
    M3P4_LINE = -(178);    
    M3P5_LINE = -(179);    
    M3Q1_LINE = -(180);    
    M4M5_LINE = -(181);    
    M4N1_LINE = -(182);    
    M4N2_LINE = -(183);    
    M4N3_LINE = -(184);    
    M4N4_LINE = -(185);    
    M4N5_LINE = -(186);    
    M4N6_LINE = -(187);    
    M4N7_LINE = -(188);    
    M4O1_LINE = -(189);    
    M4O2_LINE = -(190);    
    M4O3_LINE = -(191);    
    M4O4_LINE = -(192);    
    M4O5_LINE = -(193);    
    M4O6_LINE = -(194);    
    M4O7_LINE = -(195);    
    M4P1_LINE = -(196);    
    M4P2_LINE = -(197);    
    M4P3_LINE = -(198);    
    M4P4_LINE = -(199);    
    M4P5_LINE = -(200);    
    M5N1_LINE = -(201);    
    M5N2_LINE = -(202);    
    M5N3_LINE = -(203);    
    M5N4_LINE = -(204);    
    M5N5_LINE = -(205);    
    M5N6_LINE = -(206);    
    M5N7_LINE = -(207);    
    M5O1_LINE = -(208);    
    M5O2_LINE = -(209);    
    M5O3_LINE = -(210);    
    M5O4_LINE = -(211);    
    M5O5_LINE = -(212);    
    M5O6_LINE = -(213);    
    M5O7_LINE = -(214);    
    M5P1_LINE = -(215);    
    M5P2_LINE = -(216);    
    M5P3_LINE = -(217);    
    M5P4_LINE = -(218);    
    M5P5_LINE = -(219);    
    N1N2_LINE = -(220);    
    N1N3_LINE = -(221);    
    N1N4_LINE = -(222);    
    N1N5_LINE = -(223);    
    N1N6_LINE = -(224);    
    N1N7_LINE = -(225);    
    N1O1_LINE = -(226);    
    N1O2_LINE = -(227);    
    N1O3_LINE = -(228);    
    N1O4_LINE = -(229);    
    N1O5_LINE = -(230);    
    N1O6_LINE = -(231);    
    N1O7_LINE = -(232);    
    N1P1_LINE = -(233);    
    N1P2_LINE = -(234);    
    N1P3_LINE = -(235);    
    N1P4_LINE = -(236);    
    N1P5_LINE = -(237);    
    N2N3_LINE = -(238);    
    N2N4_LINE = -(239);    
    N2N5_LINE = -(240);    
    N2N6_LINE = -(241);    
    N2N7_LINE = -(242);    
    N2O1_LINE = -(243);    
    N2O2_LINE = -(244);    
    N2O3_LINE = -(245);    
    N2O4_LINE = -(246);    
    N2O5_LINE = -(247);    
    N2O6_LINE = -(248);    
    N2O7_LINE = -(249);    
    N2P1_LINE = -(250);    
    N2P2_LINE = -(251);    
    N2P3_LINE = -(252);    
    N2P4_LINE = -(253);    
    N2P5_LINE = -(254);    
    N3N4_LINE = -(255);    
    N3N5_LINE = -(256);    
    N3N6_LINE = -(257);    
    N3N7_LINE = -(258);    
    N3O1_LINE = -(259);    
    N3O2_LINE = -(260);    
    N3O3_LINE = -(261);    
    N3O4_LINE = -(262);    
    N3O5_LINE = -(263);    
    N3O6_LINE = -(264);    
    N3O7_LINE = -(265);    
    N3P1_LINE = -(266);    
    N3P2_LINE = -(267);    
    N3P3_LINE = -(268);    
    N3P4_LINE = -(269);    
    N3P5_LINE = -(270);    
    N4N5_LINE = -(271);    
    N4N6_LINE = -(272);    
    N4N7_LINE = -(273);    
    N4O1_LINE = -(274);    
    N4O2_LINE = -(275);    
    N4O3_LINE = -(276);    
    N4O4_LINE = -(277);    
    N4O5_LINE = -(278);    
    N4O6_LINE = -(279);    
    N4O7_LINE = -(280);    
    N4P1_LINE = -(281);    
    N4P2_LINE = -(282);    
    N4P3_LINE = -(283);    
    N4P4_LINE = -(284);    
    N4P5_LINE = -(285);    
    N5N6_LINE = -(286);    
    N5N7_LINE = -(287);    
    N5O1_LINE = -(288);    
    N5O2_LINE = -(289);    
    N5O3_LINE = -(290);    
    N5O4_LINE = -(291);    
    N5O5_LINE = -(292);    
    N5O6_LINE = -(293);    
    N5O7_LINE = -(294);    
    N5P1_LINE = -(295);    
    N5P2_LINE = -(296);    
    N5P3_LINE = -(297);    
    N5P4_LINE = -(298);    
    N5P5_LINE = -(299);    
    N6N7_LINE = -(300);    
    N6O1_LINE = -(301);    
    N6O2_LINE = -(302);    
    N6O3_LINE = -(303);    
    N6O4_LINE = -(304);    
    N6O5_LINE = -(305);    
    N6O6_LINE = -(306);    
    N6O7_LINE = -(307);    
    N6P1_LINE = -(308);    
    N6P2_LINE = -(309);    
    N6P3_LINE = -(310);    
    N6P4_LINE = -(311);    
    N6P5_LINE = -(312);    
    N7O1_LINE = -(313);    
    N7O2_LINE = -(314);    
    N7O3_LINE = -(315);    
    N7O4_LINE = -(316);    
    N7O5_LINE = -(317);    
    N7O6_LINE = -(318);    
    N7O7_LINE = -(319);    
    N7P1_LINE = -(320);    
    N7P2_LINE = -(321);    
    N7P3_LINE = -(322);    
    N7P4_LINE = -(323);    
    N7P5_LINE = -(324);    
    O1O2_LINE = -(325);    
    O1O3_LINE = -(326);    
    O1O4_LINE = -(327);    
    O1O5_LINE = -(328);    
    O1O6_LINE = -(329);    
    O1O7_LINE = -(330);    
    O1P1_LINE = -(331);    
    O1P2_LINE = -(332);    
    O1P3_LINE = -(333);    
    O1P4_LINE = -(334);    
    O1P5_LINE = -(335);    
    O2O3_LINE = -(336);    
    O2O4_LINE = -(337);    
    O2O5_LINE = -(338);    
    O2O6_LINE = -(339);    
    O2O7_LINE = -(340);    
    O2P1_LINE = -(341);    
    O2P2_LINE = -(342);    
    O2P3_LINE = -(343);    
    O2P4_LINE = -(344);    
    O2P5_LINE = -(345);    
    O3O4_LINE = -(346);    
    O3O5_LINE = -(347);    
    O3O6_LINE = -(348);    
    O3O7_LINE = -(349);    
    O3P1_LINE = -(350);    
    O3P2_LINE = -(351);    
    O3P3_LINE = -(352);    
    O3P4_LINE = -(353);    
    O3P5_LINE = -(354);    
    O4O5_LINE = -(355);    
    O4O6_LINE = -(356);    
    O4O7_LINE = -(357);    
    O4P1_LINE = -(358);    
    O4P2_LINE = -(359);    
    O4P3_LINE = -(360);    
    O4P4_LINE = -(361);    
    O4P5_LINE = -(362);    
    O5O6_LINE = -(363);    
    O5O7_LINE = -(364);    
    O5P1_LINE = -(365);    
    O5P2_LINE = -(366);    
    O5P3_LINE = -(367);    
    O5P4_LINE = -(368);    
    O5P5_LINE = -(369);    
    O6O7_LINE = -(370);    
    O6P4_LINE = -(371);    
    O6P5_LINE = -(372);    
    O7P4_LINE = -(373);    
    O7P5_LINE = -(374);    
    P1P2_LINE = -(375);    
    P1P3_LINE = -(376);    
    P1P4_LINE = -(377);    
    P1P5_LINE = -(378);    
    P2P3_LINE = -(379);    
    P2P4_LINE = -(380);    
    P2P5_LINE = -(381);    
    P3P4_LINE = -(382);    
    P3P5_LINE = -(383);    
    KA1_LINE = KL3_LINE;    
    KA2_LINE = KL2_LINE;    
    KA3_LINE = KL1_LINE;    
    KB1_LINE = KM3_LINE;    
    KB2_LINE = KN3_LINE;    
    KB3_LINE = KM2_LINE;    
    KB4_LINE = KN5_LINE;    
    KB5_LINE = KM5_LINE;    
    LA1_LINE = L3M5_LINE;    
    LA2_LINE = L3M4_LINE;    
    LB1_LINE = L2M4_LINE;    
    LB2_LINE = L3N5_LINE;    
    LB3_LINE = L1M3_LINE;    
    LB4_LINE = L1M2_LINE;    
    LB5_LINE = L3O45_LINE;    
    LB6_LINE = L3N1_LINE;    
    LB7_LINE = L3O1_LINE;    
    LB9_LINE = L1M5_LINE;    
    LB10_LINE = L1M4_LINE;    
    LB15_LINE = L3N4_LINE;    
    LB17_LINE = L2M3_LINE;    
    LG1_LINE = L2N4_LINE;    
    LG2_LINE = L1N2_LINE;    
    LG3_LINE = L1N3_LINE;    
    LG4_LINE = L1O3_LINE;    
    LG5_LINE = L2N1_LINE;    
    LG6_LINE = L2O4_LINE;    
    LG8_LINE = L2O1_LINE;    
    LE_LINE = L2M1_LINE;    
    LH_LINE = L2M1_LINE;    
    LL_LINE = L3M1_LINE;    
    LS_LINE = L3M3_LINE;    
    LT_LINE = L3M2_LINE;    
    LU_LINE = L3N6_LINE;    
    LV_LINE = L2N6_LINE;    
    MA1_LINE = M5N7_LINE;    
    MA2_LINE = M5N6_LINE;    
    MB_LINE = M4N6_LINE;    
    MG_LINE = M3N5_LINE;    
    F1_TRANS = 0;    
    F12_TRANS = 1;    
    F13_TRANS = 2;    
    FP13_TRANS = 3;    
    F23_TRANS = 4;    
    FL12_TRANS = 1;    
    FL13_TRANS = 2;    
    FLP13_TRANS = 3;    
    FL23_TRANS = 4;    
    FM12_TRANS = 5;    
    FM13_TRANS = 6;    
    FM14_TRANS = 7;    
    FM15_TRANS = 8;    
    FM23_TRANS = 9;    
    FM24_TRANS = 10;    
    FM25_TRANS = 11;    
    FM34_TRANS = 12;    
    FM35_TRANS = 13;    
    FM45_TRANS = 14;    
    ZMAX = 120;    
    MENDEL_MAX = 107;    
    CRYSTALARRAY_MAX = 100;    
    MAXFILENAMESIZE = 1000;    
    SHELLNUM = 28;    
    SHELLNUM_K = 31;    
    SHELLNUM_C = 29;    
    LINENUM = 383;    
    TRANSNUM = 15;    
    AUGERNUM = 996;    
    SHELLNUM_A = 9;    
  { Delta for size increase of Crystal_Array.crystal array  }
    N_NEW_CRYSTAL = 10;    

    K_L1L1_AUGER = 0;    
    K_L1L2_AUGER = 1;    
    K_L1L3_AUGER = 2;    
    K_L1M1_AUGER = 3;    
    K_L1M2_AUGER = 4;    
    K_L1M3_AUGER = 5;    
    K_L1M4_AUGER = 6;    
    K_L1M5_AUGER = 7;    
    K_L1N1_AUGER = 8;    
    K_L1N2_AUGER = 9;    
    K_L1N3_AUGER = 10;    
    K_L1N4_AUGER = 11;    
    K_L1N5_AUGER = 12;    
    K_L1N6_AUGER = 13;    
    K_L1N7_AUGER = 14;    
    K_L1O1_AUGER = 15;    
    K_L1O2_AUGER = 16;    
    K_L1O3_AUGER = 17;    
    K_L1O4_AUGER = 18;    
    K_L1O5_AUGER = 19;    
    K_L1O6_AUGER = 20;    
    K_L1O7_AUGER = 21;    
    K_L1P1_AUGER = 22;    
    K_L1P2_AUGER = 23;    
    K_L1P3_AUGER = 24;    
    K_L1P4_AUGER = 25;    
    K_L1P5_AUGER = 26;    
    K_L1Q1_AUGER = 27;    
    K_L1Q2_AUGER = 28;    
    K_L1Q3_AUGER = 29;    
    K_L2L1_AUGER = 30;    
    K_L2L2_AUGER = 31;    
    K_L2L3_AUGER = 32;    
    K_L2M1_AUGER = 33;    
    K_L2M2_AUGER = 34;    
    K_L2M3_AUGER = 35;    
    K_L2M4_AUGER = 36;    
    K_L2M5_AUGER = 37;    
    K_L2N1_AUGER = 38;    
    K_L2N2_AUGER = 39;    
    K_L2N3_AUGER = 40;    
    K_L2N4_AUGER = 41;    
    K_L2N5_AUGER = 42;    
    K_L2N6_AUGER = 43;    
    K_L2N7_AUGER = 44;    
    K_L2O1_AUGER = 45;    
    K_L2O2_AUGER = 46;    
    K_L2O3_AUGER = 47;    
    K_L2O4_AUGER = 48;    
    K_L2O5_AUGER = 49;    
    K_L2O6_AUGER = 50;    
    K_L2O7_AUGER = 51;    
    K_L2P1_AUGER = 52;    
    K_L2P2_AUGER = 53;    
    K_L2P3_AUGER = 54;    
    K_L2P4_AUGER = 55;    
    K_L2P5_AUGER = 56;    
    K_L2Q1_AUGER = 57;    
    K_L2Q2_AUGER = 58;    
    K_L2Q3_AUGER = 59;    
    K_L3L1_AUGER = 60;    
    K_L3L2_AUGER = 61;    
    K_L3L3_AUGER = 62;    
    K_L3M1_AUGER = 63;    
    K_L3M2_AUGER = 64;    
    K_L3M3_AUGER = 65;    
    K_L3M4_AUGER = 66;    
    K_L3M5_AUGER = 67;    
    K_L3N1_AUGER = 68;    
    K_L3N2_AUGER = 69;    
    K_L3N3_AUGER = 70;    
    K_L3N4_AUGER = 71;    
    K_L3N5_AUGER = 72;    
    K_L3N6_AUGER = 73;    
    K_L3N7_AUGER = 74;    
    K_L3O1_AUGER = 75;    
    K_L3O2_AUGER = 76;    
    K_L3O3_AUGER = 77;    
    K_L3O4_AUGER = 78;    
    K_L3O5_AUGER = 79;    
    K_L3O6_AUGER = 80;    
    K_L3O7_AUGER = 81;    
    K_L3P1_AUGER = 82;    
    K_L3P2_AUGER = 83;    
    K_L3P3_AUGER = 84;    
    K_L3P4_AUGER = 85;    
    K_L3P5_AUGER = 86;    
    K_L3Q1_AUGER = 87;    
    K_L3Q2_AUGER = 88;    
    K_L3Q3_AUGER = 89;    
    K_M1L1_AUGER = 90;    
    K_M1L2_AUGER = 91;    
    K_M1L3_AUGER = 92;    
    K_M1M1_AUGER = 93;    
    K_M1M2_AUGER = 94;    
    K_M1M3_AUGER = 95;    
    K_M1M4_AUGER = 96;    
    K_M1M5_AUGER = 97;    
    K_M1N1_AUGER = 98;    
    K_M1N2_AUGER = 99;    
    K_M1N3_AUGER = 100;    
    K_M1N4_AUGER = 101;    
    K_M1N5_AUGER = 102;    
    K_M1N6_AUGER = 103;    
    K_M1N7_AUGER = 104;    
    K_M1O1_AUGER = 105;    
    K_M1O2_AUGER = 106;    
    K_M1O3_AUGER = 107;    
    K_M1O4_AUGER = 108;    
    K_M1O5_AUGER = 109;    
    K_M1O6_AUGER = 110;    
    K_M1O7_AUGER = 111;    
    K_M1P1_AUGER = 112;    
    K_M1P2_AUGER = 113;    
    K_M1P3_AUGER = 114;    
    K_M1P4_AUGER = 115;    
    K_M1P5_AUGER = 116;    
    K_M1Q1_AUGER = 117;    
    K_M1Q2_AUGER = 118;    
    K_M1Q3_AUGER = 119;    
    K_M2L1_AUGER = 120;    
    K_M2L2_AUGER = 121;    
    K_M2L3_AUGER = 122;    
    K_M2M1_AUGER = 123;    
    K_M2M2_AUGER = 124;    
    K_M2M3_AUGER = 125;    
    K_M2M4_AUGER = 126;    
    K_M2M5_AUGER = 127;    
    K_M2N1_AUGER = 128;    
    K_M2N2_AUGER = 129;    
    K_M2N3_AUGER = 130;    
    K_M2N4_AUGER = 131;    
    K_M2N5_AUGER = 132;    
    K_M2N6_AUGER = 133;    
    K_M2N7_AUGER = 134;    
    K_M2O1_AUGER = 135;    
    K_M2O2_AUGER = 136;    
    K_M2O3_AUGER = 137;    
    K_M2O4_AUGER = 138;    
    K_M2O5_AUGER = 139;    
    K_M2O6_AUGER = 140;    
    K_M2O7_AUGER = 141;    
    K_M2P1_AUGER = 142;    
    K_M2P2_AUGER = 143;    
    K_M2P3_AUGER = 144;    
    K_M2P4_AUGER = 145;    
    K_M2P5_AUGER = 146;    
    K_M2Q1_AUGER = 147;    
    K_M2Q2_AUGER = 148;    
    K_M2Q3_AUGER = 149;    
    K_M3L1_AUGER = 150;    
    K_M3L2_AUGER = 151;    
    K_M3L3_AUGER = 152;    
    K_M3M1_AUGER = 153;    
    K_M3M2_AUGER = 154;    
    K_M3M3_AUGER = 155;    
    K_M3M4_AUGER = 156;    
    K_M3M5_AUGER = 157;    
    K_M3N1_AUGER = 158;    
    K_M3N2_AUGER = 159;    
    K_M3N3_AUGER = 160;    
    K_M3N4_AUGER = 161;    
    K_M3N5_AUGER = 162;    
    K_M3N6_AUGER = 163;    
    K_M3N7_AUGER = 164;    
    K_M3O1_AUGER = 165;    
    K_M3O2_AUGER = 166;    
    K_M3O3_AUGER = 167;    
    K_M3O4_AUGER = 168;    
    K_M3O5_AUGER = 169;    
    K_M3O6_AUGER = 170;    
    K_M3O7_AUGER = 171;    
    K_M3P1_AUGER = 172;    
    K_M3P2_AUGER = 173;    
    K_M3P3_AUGER = 174;    
    K_M3P4_AUGER = 175;    
    K_M3P5_AUGER = 176;    
    K_M3Q1_AUGER = 177;    
    K_M3Q2_AUGER = 178;    
    K_M3Q3_AUGER = 179;    
    K_M4L1_AUGER = 180;    
    K_M4L2_AUGER = 181;    
    K_M4L3_AUGER = 182;    
    K_M4M1_AUGER = 183;    
    K_M4M2_AUGER = 184;    
    K_M4M3_AUGER = 185;    
    K_M4M4_AUGER = 186;    
    K_M4M5_AUGER = 187;    
    K_M4N1_AUGER = 188;    
    K_M4N2_AUGER = 189;    
    K_M4N3_AUGER = 190;    
    K_M4N4_AUGER = 191;    
    K_M4N5_AUGER = 192;    
    K_M4N6_AUGER = 193;    
    K_M4N7_AUGER = 194;    
    K_M4O1_AUGER = 195;    
    K_M4O2_AUGER = 196;    
    K_M4O3_AUGER = 197;    
    K_M4O4_AUGER = 198;    
    K_M4O5_AUGER = 199;    
    K_M4O6_AUGER = 200;    
    K_M4O7_AUGER = 201;    
    K_M4P1_AUGER = 202;    
    K_M4P2_AUGER = 203;    
    K_M4P3_AUGER = 204;    
    K_M4P4_AUGER = 205;    
    K_M4P5_AUGER = 206;    
    K_M4Q1_AUGER = 207;    
    K_M4Q2_AUGER = 208;    
    K_M4Q3_AUGER = 209;    
    K_M5L1_AUGER = 210;    
    K_M5L2_AUGER = 211;    
    K_M5L3_AUGER = 212;    
    K_M5M1_AUGER = 213;    
    K_M5M2_AUGER = 214;    
    K_M5M3_AUGER = 215;    
    K_M5M4_AUGER = 216;    
    K_M5M5_AUGER = 217;    
    K_M5N1_AUGER = 218;    
    K_M5N2_AUGER = 219;    
    K_M5N3_AUGER = 220;    
    K_M5N4_AUGER = 221;    
    K_M5N5_AUGER = 222;    
    K_M5N6_AUGER = 223;    
    K_M5N7_AUGER = 224;    
    K_M5O1_AUGER = 225;    
    K_M5O2_AUGER = 226;    
    K_M5O3_AUGER = 227;    
    K_M5O4_AUGER = 228;    
    K_M5O5_AUGER = 229;    
    K_M5O6_AUGER = 230;    
    K_M5O7_AUGER = 231;    
    K_M5P1_AUGER = 232;    
    K_M5P2_AUGER = 233;    
    K_M5P3_AUGER = 234;    
    K_M5P4_AUGER = 235;    
    K_M5P5_AUGER = 236;    
    K_M5Q1_AUGER = 237;    
    K_M5Q2_AUGER = 238;    
    K_M5Q3_AUGER = 239;    
    L1_L2L2_AUGER = 240;    
    L1_L2L3_AUGER = 241;    
    L1_L2M1_AUGER = 242;    
    L1_L2M2_AUGER = 243;    
    L1_L2M3_AUGER = 244;    
    L1_L2M4_AUGER = 245;    
    L1_L2M5_AUGER = 246;    
    L1_L2N1_AUGER = 247;    
    L1_L2N2_AUGER = 248;    
    L1_L2N3_AUGER = 249;    
    L1_L2N4_AUGER = 250;    
    L1_L2N5_AUGER = 251;    
    L1_L2N6_AUGER = 252;    
    L1_L2N7_AUGER = 253;    
    L1_L2O1_AUGER = 254;    
    L1_L2O2_AUGER = 255;    
    L1_L2O3_AUGER = 256;    
    L1_L2O4_AUGER = 257;    
    L1_L2O5_AUGER = 258;    
    L1_L2O6_AUGER = 259;    
    L1_L2O7_AUGER = 260;    
    L1_L2P1_AUGER = 261;    
    L1_L2P2_AUGER = 262;    
    L1_L2P3_AUGER = 263;    
    L1_L2P4_AUGER = 264;    
    L1_L2P5_AUGER = 265;    
    L1_L2Q1_AUGER = 266;    
    L1_L2Q2_AUGER = 267;    
    L1_L2Q3_AUGER = 268;    
    L1_L3L2_AUGER = 269;    
    L1_L3L3_AUGER = 270;    
    L1_L3M1_AUGER = 271;    
    L1_L3M2_AUGER = 272;    
    L1_L3M3_AUGER = 273;    
    L1_L3M4_AUGER = 274;    
    L1_L3M5_AUGER = 275;    
    L1_L3N1_AUGER = 276;    
    L1_L3N2_AUGER = 277;    
    L1_L3N3_AUGER = 278;    
    L1_L3N4_AUGER = 279;    
    L1_L3N5_AUGER = 280;    
    L1_L3N6_AUGER = 281;    
    L1_L3N7_AUGER = 282;    
    L1_L3O1_AUGER = 283;    
    L1_L3O2_AUGER = 284;    
    L1_L3O3_AUGER = 285;    
    L1_L3O4_AUGER = 286;    
    L1_L3O5_AUGER = 287;    
    L1_L3O6_AUGER = 288;    
    L1_L3O7_AUGER = 289;    
    L1_L3P1_AUGER = 290;    
    L1_L3P2_AUGER = 291;    
    L1_L3P3_AUGER = 292;    
    L1_L3P4_AUGER = 293;    
    L1_L3P5_AUGER = 294;    
    L1_L3Q1_AUGER = 295;    
    L1_L3Q2_AUGER = 296;    
    L1_L3Q3_AUGER = 297;    
    L1_M1L2_AUGER = 298;    
    L1_M1L3_AUGER = 299;    
    L1_M1M1_AUGER = 300;    
    L1_M1M2_AUGER = 301;    
    L1_M1M3_AUGER = 302;    
    L1_M1M4_AUGER = 303;    
    L1_M1M5_AUGER = 304;    
    L1_M1N1_AUGER = 305;    
    L1_M1N2_AUGER = 306;    
    L1_M1N3_AUGER = 307;    
    L1_M1N4_AUGER = 308;    
    L1_M1N5_AUGER = 309;    
    L1_M1N6_AUGER = 310;    
    L1_M1N7_AUGER = 311;    
    L1_M1O1_AUGER = 312;    
    L1_M1O2_AUGER = 313;    
    L1_M1O3_AUGER = 314;    
    L1_M1O4_AUGER = 315;    
    L1_M1O5_AUGER = 316;    
    L1_M1O6_AUGER = 317;    
    L1_M1O7_AUGER = 318;    
    L1_M1P1_AUGER = 319;    
    L1_M1P2_AUGER = 320;    
    L1_M1P3_AUGER = 321;    
    L1_M1P4_AUGER = 322;    
    L1_M1P5_AUGER = 323;    
    L1_M1Q1_AUGER = 324;    
    L1_M1Q2_AUGER = 325;    
    L1_M1Q3_AUGER = 326;    
    L1_M2L2_AUGER = 327;    
    L1_M2L3_AUGER = 328;    
    L1_M2M1_AUGER = 329;    
    L1_M2M2_AUGER = 330;    
    L1_M2M3_AUGER = 331;    
    L1_M2M4_AUGER = 332;    
    L1_M2M5_AUGER = 333;    
    L1_M2N1_AUGER = 334;    
    L1_M2N2_AUGER = 335;    
    L1_M2N3_AUGER = 336;    
    L1_M2N4_AUGER = 337;    
    L1_M2N5_AUGER = 338;    
    L1_M2N6_AUGER = 339;    
    L1_M2N7_AUGER = 340;    
    L1_M2O1_AUGER = 341;    
    L1_M2O2_AUGER = 342;    
    L1_M2O3_AUGER = 343;    
    L1_M2O4_AUGER = 344;    
    L1_M2O5_AUGER = 345;    
    L1_M2O6_AUGER = 346;    
    L1_M2O7_AUGER = 347;    
    L1_M2P1_AUGER = 348;    
    L1_M2P2_AUGER = 349;    
    L1_M2P3_AUGER = 350;    
    L1_M2P4_AUGER = 351;    
    L1_M2P5_AUGER = 352;    
    L1_M2Q1_AUGER = 353;    
    L1_M2Q2_AUGER = 354;    
    L1_M2Q3_AUGER = 355;    
    L1_M3L2_AUGER = 356;    
    L1_M3L3_AUGER = 357;    
    L1_M3M1_AUGER = 358;    
    L1_M3M2_AUGER = 359;    
    L1_M3M3_AUGER = 360;    
    L1_M3M4_AUGER = 361;    
    L1_M3M5_AUGER = 362;    
    L1_M3N1_AUGER = 363;    
    L1_M3N2_AUGER = 364;    
    L1_M3N3_AUGER = 365;    
    L1_M3N4_AUGER = 366;    
    L1_M3N5_AUGER = 367;    
    L1_M3N6_AUGER = 368;    
    L1_M3N7_AUGER = 369;    
    L1_M3O1_AUGER = 370;    
    L1_M3O2_AUGER = 371;    
    L1_M3O3_AUGER = 372;    
    L1_M3O4_AUGER = 373;    
    L1_M3O5_AUGER = 374;    
    L1_M3O6_AUGER = 375;    
    L1_M3O7_AUGER = 376;    
    L1_M3P1_AUGER = 377;    
    L1_M3P2_AUGER = 378;    
    L1_M3P3_AUGER = 379;    
    L1_M3P4_AUGER = 380;    
    L1_M3P5_AUGER = 381;    
    L1_M3Q1_AUGER = 382;    
    L1_M3Q2_AUGER = 383;    
    L1_M3Q3_AUGER = 384;    
    L1_M4L2_AUGER = 385;    
    L1_M4L3_AUGER = 386;    
    L1_M4M1_AUGER = 387;    
    L1_M4M2_AUGER = 388;    
    L1_M4M3_AUGER = 389;    
    L1_M4M4_AUGER = 390;    
    L1_M4M5_AUGER = 391;    
    L1_M4N1_AUGER = 392;    
    L1_M4N2_AUGER = 393;    
    L1_M4N3_AUGER = 394;    
    L1_M4N4_AUGER = 395;    
    L1_M4N5_AUGER = 396;    
    L1_M4N6_AUGER = 397;    
    L1_M4N7_AUGER = 398;    
    L1_M4O1_AUGER = 399;    
    L1_M4O2_AUGER = 400;    
    L1_M4O3_AUGER = 401;    
    L1_M4O4_AUGER = 402;    
    L1_M4O5_AUGER = 403;    
    L1_M4O6_AUGER = 404;    
    L1_M4O7_AUGER = 405;    
    L1_M4P1_AUGER = 406;    
    L1_M4P2_AUGER = 407;    
    L1_M4P3_AUGER = 408;    
    L1_M4P4_AUGER = 409;    
    L1_M4P5_AUGER = 410;    
    L1_M4Q1_AUGER = 411;    
    L1_M4Q2_AUGER = 412;    
    L1_M4Q3_AUGER = 413;    
    L1_M5L2_AUGER = 414;    
    L1_M5L3_AUGER = 415;    
    L1_M5M1_AUGER = 416;    
    L1_M5M2_AUGER = 417;    
    L1_M5M3_AUGER = 418;    
    L1_M5M4_AUGER = 419;    
    L1_M5M5_AUGER = 420;    
    L1_M5N1_AUGER = 421;    
    L1_M5N2_AUGER = 422;    
    L1_M5N3_AUGER = 423;    
    L1_M5N4_AUGER = 424;    
    L1_M5N5_AUGER = 425;    
    L1_M5N6_AUGER = 426;    
    L1_M5N7_AUGER = 427;    
    L1_M5O1_AUGER = 428;    
    L1_M5O2_AUGER = 429;    
    L1_M5O3_AUGER = 430;    
    L1_M5O4_AUGER = 431;    
    L1_M5O5_AUGER = 432;    
    L1_M5O6_AUGER = 433;    
    L1_M5O7_AUGER = 434;    
    L1_M5P1_AUGER = 435;    
    L1_M5P2_AUGER = 436;    
    L1_M5P3_AUGER = 437;    
    L1_M5P4_AUGER = 438;    
    L1_M5P5_AUGER = 439;    
    L1_M5Q1_AUGER = 440;    
    L1_M5Q2_AUGER = 441;    
    L1_M5Q3_AUGER = 442;    
    L2_L3L3_AUGER = 443;    
    L2_L3M1_AUGER = 444;    
    L2_L3M2_AUGER = 445;    
    L2_L3M3_AUGER = 446;    
    L2_L3M4_AUGER = 447;    
    L2_L3M5_AUGER = 448;    
    L2_L3N1_AUGER = 449;    
    L2_L3N2_AUGER = 450;    
    L2_L3N3_AUGER = 451;    
    L2_L3N4_AUGER = 452;    
    L2_L3N5_AUGER = 453;    
    L2_L3N6_AUGER = 454;    
    L2_L3N7_AUGER = 455;    
    L2_L3O1_AUGER = 456;    
    L2_L3O2_AUGER = 457;    
    L2_L3O3_AUGER = 458;    
    L2_L3O4_AUGER = 459;    
    L2_L3O5_AUGER = 460;    
    L2_L3O6_AUGER = 461;    
    L2_L3O7_AUGER = 462;    
    L2_L3P1_AUGER = 463;    
    L2_L3P2_AUGER = 464;    
    L2_L3P3_AUGER = 465;    
    L2_L3P4_AUGER = 466;    
    L2_L3P5_AUGER = 467;    
    L2_L3Q1_AUGER = 468;    
    L2_L3Q2_AUGER = 469;    
    L2_L3Q3_AUGER = 470;    
    L2_M1L3_AUGER = 471;    
    L2_M1M1_AUGER = 472;    
    L2_M1M2_AUGER = 473;    
    L2_M1M3_AUGER = 474;    
    L2_M1M4_AUGER = 475;    
    L2_M1M5_AUGER = 476;    
    L2_M1N1_AUGER = 477;    
    L2_M1N2_AUGER = 478;    
    L2_M1N3_AUGER = 479;    
    L2_M1N4_AUGER = 480;    
    L2_M1N5_AUGER = 481;    
    L2_M1N6_AUGER = 482;    
    L2_M1N7_AUGER = 483;    
    L2_M1O1_AUGER = 484;    
    L2_M1O2_AUGER = 485;    
    L2_M1O3_AUGER = 486;    
    L2_M1O4_AUGER = 487;    
    L2_M1O5_AUGER = 488;    
    L2_M1O6_AUGER = 489;    
    L2_M1O7_AUGER = 490;    
    L2_M1P1_AUGER = 491;    
    L2_M1P2_AUGER = 492;    
    L2_M1P3_AUGER = 493;    
    L2_M1P4_AUGER = 494;    
    L2_M1P5_AUGER = 495;    
    L2_M1Q1_AUGER = 496;    
    L2_M1Q2_AUGER = 497;    
    L2_M1Q3_AUGER = 498;    
    L2_M2L3_AUGER = 499;    
    L2_M2M1_AUGER = 500;    
    L2_M2M2_AUGER = 501;    
    L2_M2M3_AUGER = 502;    
    L2_M2M4_AUGER = 503;    
    L2_M2M5_AUGER = 504;    
    L2_M2N1_AUGER = 505;    
    L2_M2N2_AUGER = 506;    
    L2_M2N3_AUGER = 507;    
    L2_M2N4_AUGER = 508;    
    L2_M2N5_AUGER = 509;    
    L2_M2N6_AUGER = 510;    
    L2_M2N7_AUGER = 511;    
    L2_M2O1_AUGER = 512;    
    L2_M2O2_AUGER = 513;    
    L2_M2O3_AUGER = 514;    
    L2_M2O4_AUGER = 515;    
    L2_M2O5_AUGER = 516;    
    L2_M2O6_AUGER = 517;    
    L2_M2O7_AUGER = 518;    
    L2_M2P1_AUGER = 519;    
    L2_M2P2_AUGER = 520;    
    L2_M2P3_AUGER = 521;    
    L2_M2P4_AUGER = 522;    
    L2_M2P5_AUGER = 523;    
    L2_M2Q1_AUGER = 524;    
    L2_M2Q2_AUGER = 525;    
    L2_M2Q3_AUGER = 526;    
    L2_M3L3_AUGER = 527;    
    L2_M3M1_AUGER = 528;    
    L2_M3M2_AUGER = 529;    
    L2_M3M3_AUGER = 530;    
    L2_M3M4_AUGER = 531;    
    L2_M3M5_AUGER = 532;    
    L2_M3N1_AUGER = 533;    
    L2_M3N2_AUGER = 534;    
    L2_M3N3_AUGER = 535;    
    L2_M3N4_AUGER = 536;    
    L2_M3N5_AUGER = 537;    
    L2_M3N6_AUGER = 538;    
    L2_M3N7_AUGER = 539;    
    L2_M3O1_AUGER = 540;    
    L2_M3O2_AUGER = 541;    
    L2_M3O3_AUGER = 542;    
    L2_M3O4_AUGER = 543;    
    L2_M3O5_AUGER = 544;    
    L2_M3O6_AUGER = 545;    
    L2_M3O7_AUGER = 546;    
    L2_M3P1_AUGER = 547;    
    L2_M3P2_AUGER = 548;    
    L2_M3P3_AUGER = 549;    
    L2_M3P4_AUGER = 550;    
    L2_M3P5_AUGER = 551;    
    L2_M3Q1_AUGER = 552;    
    L2_M3Q2_AUGER = 553;    
    L2_M3Q3_AUGER = 554;    
    L2_M4L3_AUGER = 555;    
    L2_M4M1_AUGER = 556;    
    L2_M4M2_AUGER = 557;    
    L2_M4M3_AUGER = 558;    
    L2_M4M4_AUGER = 559;    
    L2_M4M5_AUGER = 560;    
    L2_M4N1_AUGER = 561;    
    L2_M4N2_AUGER = 562;    
    L2_M4N3_AUGER = 563;    
    L2_M4N4_AUGER = 564;    
    L2_M4N5_AUGER = 565;    
    L2_M4N6_AUGER = 566;    
    L2_M4N7_AUGER = 567;    
    L2_M4O1_AUGER = 568;    
    L2_M4O2_AUGER = 569;    
    L2_M4O3_AUGER = 570;    
    L2_M4O4_AUGER = 571;    
    L2_M4O5_AUGER = 572;    
    L2_M4O6_AUGER = 573;    
    L2_M4O7_AUGER = 574;    
    L2_M4P1_AUGER = 575;    
    L2_M4P2_AUGER = 576;    
    L2_M4P3_AUGER = 577;    
    L2_M4P4_AUGER = 578;    
    L2_M4P5_AUGER = 579;    
    L2_M4Q1_AUGER = 580;    
    L2_M4Q2_AUGER = 581;    
    L2_M4Q3_AUGER = 582;    
    L2_M5L3_AUGER = 583;    
    L2_M5M1_AUGER = 584;    
    L2_M5M2_AUGER = 585;    
    L2_M5M3_AUGER = 586;    
    L2_M5M4_AUGER = 587;    
    L2_M5M5_AUGER = 588;    
    L2_M5N1_AUGER = 589;    
    L2_M5N2_AUGER = 590;    
    L2_M5N3_AUGER = 591;    
    L2_M5N4_AUGER = 592;    
    L2_M5N5_AUGER = 593;    
    L2_M5N6_AUGER = 594;    
    L2_M5N7_AUGER = 595;    
    L2_M5O1_AUGER = 596;    
    L2_M5O2_AUGER = 597;    
    L2_M5O3_AUGER = 598;    
    L2_M5O4_AUGER = 599;    
    L2_M5O5_AUGER = 600;    
    L2_M5O6_AUGER = 601;    
    L2_M5O7_AUGER = 602;    
    L2_M5P1_AUGER = 603;    
    L2_M5P2_AUGER = 604;    
    L2_M5P3_AUGER = 605;    
    L2_M5P4_AUGER = 606;    
    L2_M5P5_AUGER = 607;    
    L2_M5Q1_AUGER = 608;    
    L2_M5Q2_AUGER = 609;    
    L2_M5Q3_AUGER = 610;    
    L3_M1M1_AUGER = 611;    
    L3_M1M2_AUGER = 612;    
    L3_M1M3_AUGER = 613;    
    L3_M1M4_AUGER = 614;    
    L3_M1M5_AUGER = 615;    
    L3_M1N1_AUGER = 616;    
    L3_M1N2_AUGER = 617;    
    L3_M1N3_AUGER = 618;    
    L3_M1N4_AUGER = 619;    
    L3_M1N5_AUGER = 620;    
    L3_M1N6_AUGER = 621;    
    L3_M1N7_AUGER = 622;    
    L3_M1O1_AUGER = 623;    
    L3_M1O2_AUGER = 624;    
    L3_M1O3_AUGER = 625;    
    L3_M1O4_AUGER = 626;    
    L3_M1O5_AUGER = 627;    
    L3_M1O6_AUGER = 628;    
    L3_M1O7_AUGER = 629;    
    L3_M1P1_AUGER = 630;    
    L3_M1P2_AUGER = 631;    
    L3_M1P3_AUGER = 632;    
    L3_M1P4_AUGER = 633;    
    L3_M1P5_AUGER = 634;    
    L3_M1Q1_AUGER = 635;    
    L3_M1Q2_AUGER = 636;    
    L3_M1Q3_AUGER = 637;    
    L3_M2M1_AUGER = 638;    
    L3_M2M2_AUGER = 639;    
    L3_M2M3_AUGER = 640;    
    L3_M2M4_AUGER = 641;    
    L3_M2M5_AUGER = 642;    
    L3_M2N1_AUGER = 643;    
    L3_M2N2_AUGER = 644;    
    L3_M2N3_AUGER = 645;    
    L3_M2N4_AUGER = 646;    
    L3_M2N5_AUGER = 647;    
    L3_M2N6_AUGER = 648;    
    L3_M2N7_AUGER = 649;    
    L3_M2O1_AUGER = 650;    
    L3_M2O2_AUGER = 651;    
    L3_M2O3_AUGER = 652;    
    L3_M2O4_AUGER = 653;    
    L3_M2O5_AUGER = 654;    
    L3_M2O6_AUGER = 655;    
    L3_M2O7_AUGER = 656;    
    L3_M2P1_AUGER = 657;    
    L3_M2P2_AUGER = 658;    
    L3_M2P3_AUGER = 659;    
    L3_M2P4_AUGER = 660;    
    L3_M2P5_AUGER = 661;    
    L3_M2Q1_AUGER = 662;    
    L3_M2Q2_AUGER = 663;    
    L3_M2Q3_AUGER = 664;    
    L3_M3M1_AUGER = 665;    
    L3_M3M2_AUGER = 666;    
    L3_M3M3_AUGER = 667;    
    L3_M3M4_AUGER = 668;    
    L3_M3M5_AUGER = 669;    
    L3_M3N1_AUGER = 670;    
    L3_M3N2_AUGER = 671;    
    L3_M3N3_AUGER = 672;    
    L3_M3N4_AUGER = 673;    
    L3_M3N5_AUGER = 674;    
    L3_M3N6_AUGER = 675;    
    L3_M3N7_AUGER = 676;    
    L3_M3O1_AUGER = 677;    
    L3_M3O2_AUGER = 678;    
    L3_M3O3_AUGER = 679;    
    L3_M3O4_AUGER = 680;    
    L3_M3O5_AUGER = 681;    
    L3_M3O6_AUGER = 682;    
    L3_M3O7_AUGER = 683;    
    L3_M3P1_AUGER = 684;    
    L3_M3P2_AUGER = 685;    
    L3_M3P3_AUGER = 686;    
    L3_M3P4_AUGER = 687;    
    L3_M3P5_AUGER = 688;    
    L3_M3Q1_AUGER = 689;    
    L3_M3Q2_AUGER = 690;    
    L3_M3Q3_AUGER = 691;    
    L3_M4M1_AUGER = 692;    
    L3_M4M2_AUGER = 693;    
    L3_M4M3_AUGER = 694;    
    L3_M4M4_AUGER = 695;    
    L3_M4M5_AUGER = 696;    
    L3_M4N1_AUGER = 697;    
    L3_M4N2_AUGER = 698;    
    L3_M4N3_AUGER = 699;    
    L3_M4N4_AUGER = 700;    
    L3_M4N5_AUGER = 701;    
    L3_M4N6_AUGER = 702;    
    L3_M4N7_AUGER = 703;    
    L3_M4O1_AUGER = 704;    
    L3_M4O2_AUGER = 705;    
    L3_M4O3_AUGER = 706;    
    L3_M4O4_AUGER = 707;    
    L3_M4O5_AUGER = 708;    
    L3_M4O6_AUGER = 709;    
    L3_M4O7_AUGER = 710;    
    L3_M4P1_AUGER = 711;    
    L3_M4P2_AUGER = 712;    
    L3_M4P3_AUGER = 713;    
    L3_M4P4_AUGER = 714;    
    L3_M4P5_AUGER = 715;    
    L3_M4Q1_AUGER = 716;    
    L3_M4Q2_AUGER = 717;    
    L3_M4Q3_AUGER = 718;    
    L3_M5M1_AUGER = 719;    
    L3_M5M2_AUGER = 720;    
    L3_M5M3_AUGER = 721;    
    L3_M5M4_AUGER = 722;    
    L3_M5M5_AUGER = 723;    
    L3_M5N1_AUGER = 724;    
    L3_M5N2_AUGER = 725;    
    L3_M5N3_AUGER = 726;    
    L3_M5N4_AUGER = 727;    
    L3_M5N5_AUGER = 728;    
    L3_M5N6_AUGER = 729;    
    L3_M5N7_AUGER = 730;    
    L3_M5O1_AUGER = 731;    
    L3_M5O2_AUGER = 732;    
    L3_M5O3_AUGER = 733;    
    L3_M5O4_AUGER = 734;    
    L3_M5O5_AUGER = 735;    
    L3_M5O6_AUGER = 736;    
    L3_M5O7_AUGER = 737;    
    L3_M5P1_AUGER = 738;    
    L3_M5P2_AUGER = 739;    
    L3_M5P3_AUGER = 740;    
    L3_M5P4_AUGER = 741;    
    L3_M5P5_AUGER = 742;    
    L3_M5Q1_AUGER = 743;    
    L3_M5Q2_AUGER = 744;    
    L3_M5Q3_AUGER = 745;    
    M1_M2M2_AUGER = 746;    
    M1_M2M3_AUGER = 747;    
    M1_M2M4_AUGER = 748;    
    M1_M2M5_AUGER = 749;    
    M1_M2N1_AUGER = 750;    
    M1_M2N2_AUGER = 751;    
    M1_M2N3_AUGER = 752;    
    M1_M2N4_AUGER = 753;    
    M1_M2N5_AUGER = 754;    
    M1_M2N6_AUGER = 755;    
    M1_M2N7_AUGER = 756;    
    M1_M2O1_AUGER = 757;    
    M1_M2O2_AUGER = 758;    
    M1_M2O3_AUGER = 759;    
    M1_M2O4_AUGER = 760;    
    M1_M2O5_AUGER = 761;    
    M1_M2O6_AUGER = 762;    
    M1_M2O7_AUGER = 763;    
    M1_M2P1_AUGER = 764;    
    M1_M2P2_AUGER = 765;    
    M1_M2P3_AUGER = 766;    
    M1_M2P4_AUGER = 767;    
    M1_M2P5_AUGER = 768;    
    M1_M2Q1_AUGER = 769;    
    M1_M2Q2_AUGER = 770;    
    M1_M2Q3_AUGER = 771;    
    M1_M3M2_AUGER = 772;    
    M1_M3M3_AUGER = 773;    
    M1_M3M4_AUGER = 774;    
    M1_M3M5_AUGER = 775;    
    M1_M3N1_AUGER = 776;    
    M1_M3N2_AUGER = 777;    
    M1_M3N3_AUGER = 778;    
    M1_M3N4_AUGER = 779;    
    M1_M3N5_AUGER = 780;    
    M1_M3N6_AUGER = 781;    
    M1_M3N7_AUGER = 782;    
    M1_M3O1_AUGER = 783;    
    M1_M3O2_AUGER = 784;    
    M1_M3O3_AUGER = 785;    
    M1_M3O4_AUGER = 786;    
    M1_M3O5_AUGER = 787;    
    M1_M3O6_AUGER = 788;    
    M1_M3O7_AUGER = 789;    
    M1_M3P1_AUGER = 790;    
    M1_M3P2_AUGER = 791;    
    M1_M3P3_AUGER = 792;    
    M1_M3P4_AUGER = 793;    
    M1_M3P5_AUGER = 794;    
    M1_M3Q1_AUGER = 795;    
    M1_M3Q2_AUGER = 796;    
    M1_M3Q3_AUGER = 797;    
    M1_M4M2_AUGER = 798;    
    M1_M4M3_AUGER = 799;    
    M1_M4M4_AUGER = 800;    
    M1_M4M5_AUGER = 801;    
    M1_M4N1_AUGER = 802;    
    M1_M4N2_AUGER = 803;    
    M1_M4N3_AUGER = 804;    
    M1_M4N4_AUGER = 805;    
    M1_M4N5_AUGER = 806;    
    M1_M4N6_AUGER = 807;    
    M1_M4N7_AUGER = 808;    
    M1_M4O1_AUGER = 809;    
    M1_M4O2_AUGER = 810;    
    M1_M4O3_AUGER = 811;    
    M1_M4O4_AUGER = 812;    
    M1_M4O5_AUGER = 813;    
    M1_M4O6_AUGER = 814;    
    M1_M4O7_AUGER = 815;    
    M1_M4P1_AUGER = 816;    
    M1_M4P2_AUGER = 817;    
    M1_M4P3_AUGER = 818;    
    M1_M4P4_AUGER = 819;    
    M1_M4P5_AUGER = 820;    
    M1_M4Q1_AUGER = 821;    
    M1_M4Q2_AUGER = 822;    
    M1_M4Q3_AUGER = 823;    
    M1_M5M2_AUGER = 824;    
    M1_M5M3_AUGER = 825;    
    M1_M5M4_AUGER = 826;    
    M1_M5M5_AUGER = 827;    
    M1_M5N1_AUGER = 828;    
    M1_M5N2_AUGER = 829;    
    M1_M5N3_AUGER = 830;    
    M1_M5N4_AUGER = 831;    
    M1_M5N5_AUGER = 832;    
    M1_M5N6_AUGER = 833;    
    M1_M5N7_AUGER = 834;    
    M1_M5O1_AUGER = 835;    
    M1_M5O2_AUGER = 836;    
    M1_M5O3_AUGER = 837;    
    M1_M5O4_AUGER = 838;    
    M1_M5O5_AUGER = 839;    
    M1_M5O6_AUGER = 840;    
    M1_M5O7_AUGER = 841;    
    M1_M5P1_AUGER = 842;    
    M1_M5P2_AUGER = 843;    
    M1_M5P3_AUGER = 844;    
    M1_M5P4_AUGER = 845;    
    M1_M5P5_AUGER = 846;    
    M1_M5Q1_AUGER = 847;    
    M1_M5Q2_AUGER = 848;    
    M1_M5Q3_AUGER = 849;    
    M2_M3M3_AUGER = 850;    
    M2_M3M4_AUGER = 851;    
    M2_M3M5_AUGER = 852;    
    M2_M3N1_AUGER = 853;    
    M2_M3N2_AUGER = 854;    
    M2_M3N3_AUGER = 855;    
    M2_M3N4_AUGER = 856;    
    M2_M3N5_AUGER = 857;    
    M2_M3N6_AUGER = 858;    
    M2_M3N7_AUGER = 859;    
    M2_M3O1_AUGER = 860;    
    M2_M3O2_AUGER = 861;    
    M2_M3O3_AUGER = 862;    
    M2_M3O4_AUGER = 863;    
    M2_M3O5_AUGER = 864;    
    M2_M3O6_AUGER = 865;    
    M2_M3O7_AUGER = 866;    
    M2_M3P1_AUGER = 867;    
    M2_M3P2_AUGER = 868;    
    M2_M3P3_AUGER = 869;    
    M2_M3P4_AUGER = 870;    
    M2_M3P5_AUGER = 871;    
    M2_M3Q1_AUGER = 872;    
    M2_M3Q2_AUGER = 873;    
    M2_M3Q3_AUGER = 874;    
    M2_M4M3_AUGER = 875;    
    M2_M4M4_AUGER = 876;    
    M2_M4M5_AUGER = 877;    
    M2_M4N1_AUGER = 878;    
    M2_M4N2_AUGER = 879;    
    M2_M4N3_AUGER = 880;    
    M2_M4N4_AUGER = 881;    
    M2_M4N5_AUGER = 882;    
    M2_M4N6_AUGER = 883;    
    M2_M4N7_AUGER = 884;    
    M2_M4O1_AUGER = 885;    
    M2_M4O2_AUGER = 886;    
    M2_M4O3_AUGER = 887;    
    M2_M4O4_AUGER = 888;    
    M2_M4O5_AUGER = 889;    
    M2_M4O6_AUGER = 890;    
    M2_M4O7_AUGER = 891;    
    M2_M4P1_AUGER = 892;    
    M2_M4P2_AUGER = 893;    
    M2_M4P3_AUGER = 894;    
    M2_M4P4_AUGER = 895;    
    M2_M4P5_AUGER = 896;    
    M2_M4Q1_AUGER = 897;    
    M2_M4Q2_AUGER = 898;    
    M2_M4Q3_AUGER = 899;    
    M2_M5M3_AUGER = 900;    
    M2_M5M4_AUGER = 901;    
    M2_M5M5_AUGER = 902;    
    M2_M5N1_AUGER = 903;    
    M2_M5N2_AUGER = 904;    
    M2_M5N3_AUGER = 905;    
    M2_M5N4_AUGER = 906;    
    M2_M5N5_AUGER = 907;    
    M2_M5N6_AUGER = 908;    
    M2_M5N7_AUGER = 909;    
    M2_M5O1_AUGER = 910;    
    M2_M5O2_AUGER = 911;    
    M2_M5O3_AUGER = 912;    
    M2_M5O4_AUGER = 913;    
    M2_M5O5_AUGER = 914;    
    M2_M5O6_AUGER = 915;    
    M2_M5O7_AUGER = 916;    
    M2_M5P1_AUGER = 917;    
    M2_M5P2_AUGER = 918;    
    M2_M5P3_AUGER = 919;    
    M2_M5P4_AUGER = 920;    
    M2_M5P5_AUGER = 921;    
    M2_M5Q1_AUGER = 922;    
    M2_M5Q2_AUGER = 923;    
    M2_M5Q3_AUGER = 924;    
    M3_M4M4_AUGER = 925;    
    M3_M4M5_AUGER = 926;    
    M3_M4N1_AUGER = 927;    
    M3_M4N2_AUGER = 928;    
    M3_M4N3_AUGER = 929;    
    M3_M4N4_AUGER = 930;    
    M3_M4N5_AUGER = 931;    
    M3_M4N6_AUGER = 932;    
    M3_M4N7_AUGER = 933;    
    M3_M4O1_AUGER = 934;    
    M3_M4O2_AUGER = 935;    
    M3_M4O3_AUGER = 936;    
    M3_M4O4_AUGER = 937;    
    M3_M4O5_AUGER = 938;    
    M3_M4O6_AUGER = 939;    
    M3_M4O7_AUGER = 940;    
    M3_M4P1_AUGER = 941;    
    M3_M4P2_AUGER = 942;    
    M3_M4P3_AUGER = 943;    
    M3_M4P4_AUGER = 944;    
    M3_M4P5_AUGER = 945;    
    M3_M4Q1_AUGER = 946;    
    M3_M4Q2_AUGER = 947;    
    M3_M4Q3_AUGER = 948;    
    M3_M5M4_AUGER = 949;    
    M3_M5M5_AUGER = 950;    
    M3_M5N1_AUGER = 951;    
    M3_M5N2_AUGER = 952;    
    M3_M5N3_AUGER = 953;    
    M3_M5N4_AUGER = 954;    
    M3_M5N5_AUGER = 955;    
    M3_M5N6_AUGER = 956;    
    M3_M5N7_AUGER = 957;    
    M3_M5O1_AUGER = 958;    
    M3_M5O2_AUGER = 959;    
    M3_M5O3_AUGER = 960;    
    M3_M5O4_AUGER = 961;    
    M3_M5O5_AUGER = 962;    
    M3_M5O6_AUGER = 963;    
    M3_M5O7_AUGER = 964;    
    M3_M5P1_AUGER = 965;    
    M3_M5P2_AUGER = 966;    
    M3_M5P3_AUGER = 967;    
    M3_M5P4_AUGER = 968;    
    M3_M5P5_AUGER = 969;    
    M3_M5Q1_AUGER = 970;    
    M3_M5Q2_AUGER = 971;    
    M3_M5Q3_AUGER = 972;    
    M4_M5M5_AUGER = 973;    
    M4_M5N1_AUGER = 974;    
    M4_M5N2_AUGER = 975;    
    M4_M5N3_AUGER = 976;    
    M4_M5N4_AUGER = 977;    
    M4_M5N5_AUGER = 978;    
    M4_M5N6_AUGER = 979;    
    M4_M5N7_AUGER = 980;    
    M4_M5O1_AUGER = 981;    
    M4_M5O2_AUGER = 982;    
    M4_M5O3_AUGER = 983;    
    M4_M5O4_AUGER = 984;    
    M4_M5O5_AUGER = 985;    
    M4_M5O6_AUGER = 986;    
    M4_M5O7_AUGER = 987;    
    M4_M5P1_AUGER = 988;    
    M4_M5P2_AUGER = 989;    
    M4_M5P3_AUGER = 990;    
    M4_M5P4_AUGER = 991;    
    M4_M5P5_AUGER = 992;    
    M4_M5Q1_AUGER = 993;    
    M4_M5Q2_AUGER = 994;    
    M4_M5Q3_AUGER = 995;    




  procedure XRayInit;cdecl;external External_library name 'XRayInit';

  { Error Handling  }
  procedure SetHardExit(hard_exit:Integer);cdecl;external External_library name 'SetHardExit';

  procedure SetExitStatus(exit_status:Integer);cdecl;external External_library name 'SetExitStatus';

  function GetExitStatus:Integer;cdecl;external External_library name 'GetExitStatus';

  procedure SetErrorMessages(status:Integer);cdecl;external External_library name 'SetErrorMessages';

  function GetErrorMessages:Integer;cdecl;external External_library name 'GetErrorMessages';

  { Atomic weights  }
  function AtomicWeight(Z:Integer):double;cdecl;external External_library name 'AtomicWeight';

  { Density of pure atomic element  }
  function ElementDensity(Z:Integer):double;cdecl;external External_library name 'ElementDensity';

  { Cross sections (cm2/g)  }
  function CS_Total(Z:Integer; E:double):double;cdecl;external External_library name 'CS_Total';

  function CS_Photo(Z:Integer; E:double):double;cdecl;external External_library name 'CS_Photo';

  function CS_Rayl(Z:Integer; E:double):double;cdecl;external External_library name 'CS_Rayl';

  function CS_Compt(Z:Integer; E:double):double;cdecl;external External_library name 'CS_Compt';

  function CS_KN(E:double):double;cdecl;external External_library name 'CS_KN';

  function CS_Energy(Z:Integer; E:double):double;cdecl;external External_library name 'CS_Energy';

  { barn/atom  }
  function CSb_Total(Z:Integer; E:double):double;cdecl;external External_library name 'CSb_Total';

  function CSb_Photo(Z:Integer; E:double):double;cdecl;external External_library name 'CSb_Photo';

  function CSb_Rayl(Z:Integer; E:double):double;cdecl;external External_library name 'CSb_Rayl';

  function CSb_Compt(Z:Integer; E:double):double;cdecl;external External_library name 'CSb_Compt';

  { Unpolarized differential scattering cross sections  }
  function DCS_Thoms(theta:double):double;cdecl;external External_library name 'DCS_Thoms';

  function DCS_KN(E:double; theta:double):double;cdecl;external External_library name 'DCS_KN';

  function DCS_Rayl(Z:Integer; E:double; theta:double):double;cdecl;external External_library name 'DCS_Rayl';

  function DCS_Compt(Z:Integer; E:double; theta:double):double;cdecl;external External_library name 'DCS_Compt';

  function DCSb_Rayl(Z:Integer; E:double; theta:double):double;cdecl;external External_library name 'DCSb_Rayl';

  function DCSb_Compt(Z:Integer; E:double; theta:double):double;cdecl;external External_library name 'DCSb_Compt';

  { Polarized differential scattering cross sections  }
  function DCSP_Thoms(theta:double; phi:double):double;cdecl;external External_library name 'DCSP_Thoms';

  function DCSP_KN(E:double; theta:double; phi:double):double;cdecl;external External_library name 'DCSP_KN';

  function DCSP_Rayl(Z:Integer; E:double; theta:double; phi:double):double;cdecl;external External_library name 'DCSP_Rayl';

  function DCSP_Compt(Z:Integer; E:double; theta:double; phi:double):double;cdecl;external External_library name 'DCSP_Compt';

  function DCSPb_Rayl(Z:Integer; E:double; theta:double; phi:double):double;cdecl;external External_library name 'DCSPb_Rayl';

  function DCSPb_Compt(Z:Integer; E:double; theta:double; phi:double):double;cdecl;external External_library name 'DCSPb_Compt';

  { Scattering factors  }
  function FF_Rayl(Z:Integer; q:double):double;cdecl;external External_library name 'FF_Rayl';

  function SF_Compt(Z:Integer; q:double):double;cdecl;external External_library name 'SF_Compt';

  function MomentTransf(E:double; theta:double):double;cdecl;external External_library name 'MomentTransf';

  { X-ray fluorescent line energy  }
  function LineEnergy(Z:Integer; line:Integer):double;cdecl;external External_library name 'LineEnergy';

  { Fluorescence yield  }
  function FluorYield(Z:Integer; shell:Integer):double;cdecl;external External_library name 'FluorYield';

  { Coster-Kronig transition Probability  }
  function CosKronTransProb(Z:Integer; trans:Integer):double;cdecl;external External_library name 'CosKronTransProb';

  { Absorption-edge energies  }
  function EdgeEnergy(Z:Integer; shell:Integer):double;cdecl;external External_library name 'EdgeEnergy';

  { Jump ratio  }
  function JumpFactor(Z:Integer; shell:Integer):double;cdecl;external External_library name 'JumpFactor';

  { Fluorescent-lines cross sections  }
  function CS_FluorLine(Z:Integer; line:Integer; E:double):double;cdecl;external External_library name 'CS_FluorLine';

  function CSb_FluorLine(Z:Integer; line:Integer; E:double):double;cdecl;external External_library name 'CSb_FluorLine';

  { Fractional radiative rate  }
  function RadRate(Z:Integer; line:Integer):double;cdecl;external External_library name 'RadRate';

  { Photon energy after Compton scattering  }
  function ComptonEnergy(E0:double; theta:double):double;cdecl;external External_library name 'ComptonEnergy';

  { Anomalous Scattering Factors  }
  function Fi(Z:Integer; E:double):double;cdecl;external External_library name 'Fi';

  function Fii(Z:Integer; E:double):double;cdecl;external External_library name 'Fii';

  { Kissel Photoelectric cross sections  }
  function CS_Photo_Total(Z:Integer; E:double):double;cdecl;external External_library name 'CS_Photo_Total';

  function CSb_Photo_Total(Z:Integer; E:double):double;cdecl;external External_library name 'CSb_Photo_Total';

  function CS_Photo_Partial(Z:Integer; shell:Integer; E:double):double;cdecl;external External_library name 'CS_Photo_Partial';

  function CSb_Photo_Partial(Z:Integer; shell:Integer; E:double):double;cdecl;external External_library name 'CSb_Photo_Partial';

  { XRF cross sections using Kissel partial photoelectric cross sections  }
  function CS_FluorLine_Kissel(Z:Integer; line:Integer; E:double):double;cdecl;external External_library name 'CS_FluorLine_Kissel';

  function CSb_FluorLine_Kissel(Z:Integer; line:Integer; E:double):double;cdecl;external External_library name 'CSb_FluorLine_Kissel';

  function CS_FluorLine_Kissel_Cascade(Z:Integer; line:Integer; E:double):double;cdecl;external External_library name 'CS_FluorLine_Kissel_Cascade';

  function CSb_FluorLine_Kissel_Cascade(Z:Integer; line:Integer; E:double):double;cdecl;external External_library name 'CSb_FluorLine_Kissel_Cascade';

  function CS_FluorLine_Kissel_Nonradiative_Cascade(Z:Integer; line:Integer; E:double):double;cdecl;external External_library name 'CS_FluorLine_Kissel_Nonradiative_Cascade';

  function CSb_FluorLine_Kissel_Nonradiative_Cascade(Z:Integer; line:Integer; E:double):double;cdecl;external External_library name 'CSb_FluorLine_Kissel_Nonradiative_Cascade';

  function CS_FluorLine_Kissel_Radiative_Cascade(Z:Integer; line:Integer; E:double):double;cdecl;external External_library name 'CS_FluorLine_Kissel_Radiative_Cascade';

  function CSb_FluorLine_Kissel_Radiative_Cascade(Z:Integer; line:Integer; E:double):double;cdecl;external External_library name 'CSb_FluorLine_Kissel_Radiative_Cascade';

  function CS_FluorLine_Kissel_no_Cascade(Z:Integer; line:Integer; E:double):double;cdecl;external External_library name 'CS_FluorLine_Kissel_no_Cascade';

  function CSb_FluorLine_Kissel_no_Cascade(Z:Integer; line:Integer; E:double):double;cdecl;external External_library name 'CSb_FluorLine_Kissel_no_Cascade';

  { Total cross sections (photoionization+Rayleigh+Compton) using Kissel Total photoelectric cross sections  }
  function CS_Total_Kissel(Z:Integer; E:double):double;cdecl;external External_library name 'CS_Total_Kissel';

  function CSb_Total_Kissel(Z:Integer; E:double):double;cdecl;external External_library name 'CSb_Total_Kissel';

  { Electron configuration (according to Kissel)  }
  function ElectronConfig(Z:Integer; shell:Integer):double;cdecl;external External_library name 'ElectronConfig';

  { Cross Section functions using the compound parser  }
(* Const before type ignored *)
  function CS_Total_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CS_Total_CP';

(* Const before type ignored *)
  function CS_Photo_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CS_Photo_CP';

(* Const before type ignored *)
  function CS_Rayl_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CS_Rayl_CP';

(* Const before type ignored *)
  function CS_Compt_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CS_Compt_CP';

(* Const before type ignored *)
  function CSb_Total_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CSb_Total_CP';

(* Const before type ignored *)
  function CSb_Photo_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CSb_Photo_CP';

(* Const before type ignored *)
  function CSb_Rayl_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CSb_Rayl_CP';

(* Const before type ignored *)
  function CSb_Compt_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CSb_Compt_CP';

(* Const before type ignored *)
  function DCS_Rayl_CP(compound:Pchar; E:double; theta:double):double;cdecl;external External_library name 'DCS_Rayl_CP';

(* Const before type ignored *)
  function DCS_Compt_CP(compound:Pchar; E:double; theta:double):double;cdecl;external External_library name 'DCS_Compt_CP';

(* Const before type ignored *)
  function DCSb_Rayl_CP(compound:Pchar; E:double; theta:double):double;cdecl;external External_library name 'DCSb_Rayl_CP';

(* Const before type ignored *)
  function DCSb_Compt_CP(compound:Pchar; E:double; theta:double):double;cdecl;external External_library name 'DCSb_Compt_CP';

(* Const before type ignored *)
  function DCSP_Rayl_CP(compound:Pchar; E:double; theta:double; phi:double):double;cdecl;external External_library name 'DCSP_Rayl_CP';

(* Const before type ignored *)
  function DCSP_Compt_CP(compound:Pchar; E:double; theta:double; phi:double):double;cdecl;external External_library name 'DCSP_Compt_CP';

(* Const before type ignored *)
  function DCSPb_Rayl_CP(compound:Pchar; E:double; theta:double; phi:double):double;cdecl;external External_library name 'DCSPb_Rayl_CP';

(* Const before type ignored *)
  function DCSPb_Compt_CP(compound:Pchar; E:double; theta:double; phi:double):double;cdecl;external External_library name 'DCSPb_Compt_CP';

(* Const before type ignored *)
  function CS_Photo_Total_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CS_Photo_Total_CP';

(* Const before type ignored *)
  function CSb_Photo_Total_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CSb_Photo_Total_CP';

(* Const before type ignored *)
  function CS_Total_Kissel_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CS_Total_Kissel_CP';

(* Const before type ignored *)
  function CSb_Total_Kissel_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CSb_Total_Kissel_CP';

(* Const before type ignored *)
  function CS_Energy_CP(compound:Pchar; E:double):double;cdecl;external External_library name 'CS_Energy_CP';

  { Refractive indices functions  }
(* Const before type ignored *)
  function Refractive_Index_Re(compound:Pchar; E:double; density:double):double;cdecl;external External_library name 'Refractive_Index_Re';

(* Const before type ignored *)
  function Refractive_Index_Im(compound:Pchar; E:double; density:double):double;cdecl;external External_library name 'Refractive_Index_Im';

(* Const before type ignored *)
  function Refractive_Index(compound:Pchar; E:double; density:double):xrlComplex;cdecl;external External_library name 'Refractive_Index';

  { ComptonProfiles  }
  function ComptonProfile(Z:Integer; pz:double):double;cdecl;external External_library name 'ComptonProfile';

  function ComptonProfile_Partial(Z:Integer; shell:Integer; pz:double):double;cdecl;external External_library name 'ComptonProfile_Partial';

  { Atomic level widths  }
  function AtomicLevelWidth(Z:Integer; shell:Integer):double;cdecl;external External_library name 'AtomicLevelWidth';

  { Auger non-radiative rates  }
  function AugerRate(Z:Integer; auger_trans:Integer):double;cdecl;external External_library name 'AugerRate';

  { Auger yield  }
  function AugerYield(Z:Integer; shell:Integer):double;cdecl;external External_library name 'AugerYield';

  { this is giving a lot of trouble with python  }

  function c_abs(x:xrlComplex):double;cdecl;external External_library name 'c_abs';

  function c_mul(x:xrlComplex; y:xrlComplex):xrlComplex;cdecl;external External_library name 'c_mul';

  { Struct for an atom in a crystal.  }
  { Atomic number of atom.  }
  { Fractional contribution. Normally 1.0.  }
  { Atom position in fractions of the unit cell lengths.  }

  type
    Crystal_Atom = record
        Zatom : Integer;
        fraction : double;
        x : double;
        y : double;
        z : double;
      end;
  { Struct for a crystal.  }
  { Name of crystal.  }
  { Unit cell size in Angstroms.  }
  { Unit cell angles in degrees.  }
  { Unit cell volume in Angstroms^3.  }
  { Number of atoms.  }
  { Array of atoms in unit cell.  }

    Crystal_Struct = record
        name : ^char;
        a : double;
        b : double;
        c : double;
        alpha : double;
        beta : double;
        gamma : double;
        volume : double;
        n_atom : Integer;
        atom : ^Crystal_Atom;
      end;
  { Container struct to hold an array of CrystalStructs  }
  { Number of defined crystals.  }
  { Size of .crystal array malloc'd  }

    Crystal_Array = record
        n_crystal : Integer;
        n_alloc : Integer;
        crystal : ^Crystal_Struct;
      end;
  {
   *
   * this header includes the prototype of a function designed to parse
   * chemical formulas and a structure definition which is used to store the information.
   *
    }
  {
   * A compoundData structure will be used to store the results after parsing:
   * 	nElements: the number of different atoms present
   * 	nAtomsAll: the total number of atoms present in the compound
   * 	Elements: an array with length nElements that will contain the atomic
   * 	  numbers of the different elements present in the compound, 
   * 	  in ascending order. The array memory will be allocated using malloc 
   * 	  and should be freed by the user when appropriate.
   *	massFractions: an array with length nElements that will contain
   *	  the atomic mass fractions of the different elements present in the compound,
   *	  in an order corresponding with the Elements array. The sum of the values
   *	  in this array is equal to 1. The array memory will be allocated 
   *	  using malloc and should be freed by the user when appropriate.
   *
   * For SiO2 this would yield a structure with contents:
   *  nElements: 2
   *  nAtomsAll: 3
   *  Elements: 8 14
   *  massFractions: 0.467465  0.532535
   *
   *
   *
    }
  type
    compoundData = record
        nElements : Integer;
        nAtomsAll : double;
        Elements : array of Integer;
        massFractions : array of double;
      end;
    PcompoundData  = ^compoundData;



    function AtomicNumberToSymbol(Z:Integer):string;
    function SymbolToAtomicNumber(symbol:string):Integer;
    function CompoundParser(compound:string):PcompoundData;

implementation

  type
    compoundData_C = record
        nElements : Integer;
        nAtomsAll : double;
        Elements : ^Integer;
        massFractions : ^double;
      end;
    PcompoundData_C  = ^compoundData_C;
  { 
   * FreeCompoundData is used to free the memory allocated
   * by CompoundParser in a compoundData struct. It is recommended
   * to set the value of the struct to NULL after calling this function.
    }

  procedure FreeCompoundData(_para1:PcompoundData_C);cdecl;external External_library name 'FreeCompoundData';

  {
   * The CompoundParser function will parse a string and will return 
   * a pointer to a compoundData structure if successful, otherwise it will return a
   * NULL pointer. After usage, the struct can be freed with FreeCompoundData
    }
(* error 
struct compoundData *CompoundParser(const char compoundString[]);
in declaration at line 74 *)
  function CompoundParser_C(compoundString:PChar):PcompoundData_C;cdecl;external External_library name 'CompoundParser';
    {
     * The AtomicNumberToSymbol function returns a pointer to a string containing the element symbol.
     * If an error occurred, the NULL string is returned.
     * The string should be freed after usage with the xrlFree function
      }
    function AtomicNumberToSymbol_C(Z:Integer):Pchar;cdecl;external External_library name 'AtomicNumberToSymbol';

    {
     * The SymbolToAtomicNumber function returns the atomic number that corresponds with element symbol
     * If the element does not exist, 0 is returned
      }
    function SymbolToAtomicNumber_C(symbol:Pchar):Integer;cdecl;external External_library name 'SymbolToAtomicNumber';

    {
     *  xrlFree frees memory that was dynamically allocated by xraylib. For now it should only be used
     *  in combination with AtomicNumberToSymbol and add_compound_data
      }
    procedure xrlFree(_para1:pointer);cdecl;external External_library name 'xrlFree';
function AtomicNumberToSymbol(Z:Integer):string;
var
	temp:Pchar;
begin
	temp := AtomicNumberToSymbol_C(Z);
	if (temp <> nil) then
	begin
		AtomicNumberToSymbol := strpas(temp);
		xrlFree(temp)
	end
	else
		AtomicNumberToSymbol := '';
end;

function SymbolToAtomicNumber(symbol:string):Integer;
var
	temp:Pchar;
begin
	temp := StrAlloc(length(symbol)+1); 
	StrPCopy(temp, symbol);
	SymbolToAtomicNumber := SymbolToAtomicNumber_C(temp);
	StrDispose(temp);
end;

function CompoundParser(compound:string):PcompoundData;
var
	i:Integer;
	temp:Pchar;
	data_C:PcompoundData_C;
	Elements:^Integer;
	massFractions:^Double;
begin
	temp := StrAlloc(length(compound)+1); 
	StrPCopy(temp, compound);
	data_C := CompoundParser_C(temp);
	if (data_C = nil) then
	begin
		CompoundParser := nil;
		Exit;
	end;

	new(CompoundParser);

	SetLength(CompoundParser^.Elements, data_C^.nElements);
	SetLength(CompoundParser^.massFractions, data_C^.nElements);
	CompoundParser^.nElements := data_C^.nElements;
	CompoundParser^.nAtomsAll:= data_C^.nAtomsAll;
	Elements := data_C^.Elements;
	massFractions := data_C^.massFractions;

	for  i := 0 to CompoundParser^.nElements-1 do
	begin
		CompoundParser^.Elements[i] := Elements^;
		CompoundParser^.massFractions[i] := massFractions^;
		inc(Elements);
		inc(massFractions);
	end;
	FreeCompoundData(data_C)
end;


end.
