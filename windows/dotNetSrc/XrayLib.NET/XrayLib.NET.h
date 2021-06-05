/*
	XrayLib.NET copyright (c) 2010-2021 Matthew Wormington. All rights reserved.
	
	File: XrayLib.NET.h
	Author: Matthew Wormington
	Language: C++/CLI   
	Compiler: Microsoft Visual Studio 2019
	Created: September 4, 2010
	$Version:$
	$Revision:$
	$RevDate:$

	Description:
	Contains the definition of a managed wrapper class around the native 
	XrayLib API.

	"A library for X-ray–matter interaction cross sections for
	X-ray fluorescence applications".
	A. Brunetti, M. Sanchez del Rio, B. Golosio, A. Simionovici, A. Somogyi, 
	Spectrochimica Acta Part B 59 (2004) 1725–1731
	http://ftp.esrf.fr/pub/scisoft/xraylib/

	Notes:
	A singleton pattern has been used so that only one instance of the class is ever
	created. The Instance property provides a global point of access to the instance.
	The implementation is based on the Static Initialization example in the following
	Microsoft article: http://msdn.microsoft.com/en-us/library/ms998558.aspx


	XrayLib copyright (c) 2009, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del 
	Rio, Tom Schoonjans and Teemu Ikonen. All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright notice, this 
	  list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright notice, this 
	  list of conditions and the following disclaimer 
	  in the documentation and/or other materials provided with the distribution.
	* The names of the contributors may not be used to endorse or promote products 
	  derived from this software without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, 
	Tom Schoonjans, Teemu Ikonen and Matthew Wormington ''AS IS'' AND ANY EXPRESS OR IMPLIED 
	WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
	FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio 
	Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY 
	DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
	BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
	OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
	WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
	ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
	OF SUCH DAMAGE.
*/

#pragma once

using namespace System;

/// <summary>
/// A namespace that contains scientific classes.
/// </summary>
namespace Science {

	/// <summary>
	/// A custom exception that inherits from Exception and is thrown by the XrayLib class. 
	/// </summary>
	public ref class XrayLibException: Exception
	{
	public:
		// Overloading the constructor for passing the message associated with the exception
		XrayLibException(System::String ^message) : Exception(message) { }
	};

	/// <summary>
	/// A managed wrapper around the XrayLib library for X-ray matter interaction cross sections 
	/// for X-ray fluorescence applications. 
	/// <para>The XrayLib is a library of X-ray matter interaction data for X-ray fluorescence,
	/// and other, applications. </para>
	/// <para>Notes:</para>
	/// <para>Version 3.4 contains improved error handling for the library functions and exposes the functions
	/// as static methods of the XrayLib class. </para>
	/// <para>References:</para>
	/// <para>1) A library for X-ray matter interaction cross sections for X-ray fluorescence applications.</para>
	/// <para>A. Brunetti, M. Sanchez del Rio, B. Golosio, A. Simionovici, A. Somogyi, 
	///	Spectrochimica Acta Part B 59 (2004) 1725–1731</para>
	/// <para>2) The xraylib library for X-ray–matter interactions.Recent developments.</para>
	/// <para>T.Schoonjans, A.Brunetti, B.Golosio, M.Sanchez del Rio, V.A.Solé, C.Ferrero, L.Vincze, 
	/// Spectrochimica Acta Part B 66 (2011) 776–784</para>
	/// <para>https://github.com/tschoonj/xraylib/</para>

	/// </summary>
	public ref class XrayLib
	{
	private:
		// Only the main lines are included in the following arrays
		static array<System::String^> ^IUPACLineNames = 
			{"KL3","KL2","KM3","KN2","KM2","L3M5","L3M4","L2M4","L3N5",
			"L1M3","L1M2","L3N1","L2N4","L1N2","L1N3","L2M1","L3M1"};

		// In the SiegbahnLineName array, the following notation is used:
		// a - alpha, b - beta, g - gamma, n - Nu, l - eta
		static array<System::String^> ^SiegbahnLineNames = 
			{"Ka1","Ka2","Kb1","Kb2","Kb3","La1","La2","Lb1","Lb2",
			"Lb3","Lb4","Lb6","Lg1","Lg2","Lg3","Ln","Ll"};
		static array<int> ^SiegbahnLines =
			{KA1_LINE,KA2_LINE,KB1_LINE,KB2_LINE,KB3_LINE,LA1_LINE,LA2_LINE,LB1_LINE,LB2_LINE,
			LB3_LINE,LB4_LINE,LB6_LINE,LG1_LINE,LG2_LINE,LG3_LINE,LN_LINE,LL_LINE};

	public:
		#pragma region  Constants
		literal int VERSION_MAJOR = 4;
		literal int VERSION_MINOR = 0;
		literal double PI = 3.14159265359;

		// Values taken from physics.nist.gov
		literal double AVOGNUM = 0.602214129;    // Avogadro number (mol-1 * barn-1 * cm2) 
		literal double KEV2ANGST = 12.39841930;  // keV to Angstrom-1 conversion factor 
		literal double MEC2 = 510.998928;        // Electron rest mass (keV) 
		literal double RE2 = 0.079407877;        // Square of classical electron radius (barn)
		literal double R_E = 2.8179403267e-15;   // Classical electron radius (m)

		// IUPAC lines
		literal int KL1_LINE = -1;
		literal int KL2_LINE = -2;
		literal int KL3_LINE = -3;
		literal int KM1_LINE = -4;
		literal int KM2_LINE = -5;
		literal int KM3_LINE = -6;
		literal int KM4_LINE = -7;
		literal int KM5_LINE = -8;
		literal int KN1_LINE = -9;
		literal int KN2_LINE = -10;
		literal int KN3_LINE = -11;
		literal int KN4_LINE = -12;
		literal int KN5_LINE = -13;
		literal int KN6_LINE = -14;
		literal int KN7_LINE = -15;
		literal int KO_LINE = -16;
		literal int KO1_LINE = -17;
		literal int KO2_LINE = -18;
		literal int KO3_LINE = -19;
		literal int KO4_LINE = -20;
		literal int KO5_LINE = -21;
		literal int KO6_LINE = -22;
		literal int KO7_LINE = -23;
		literal int KP_LINE = -24;
		literal int KP1_LINE = -25;
		literal int KP2_LINE = -26;
		literal int KP3_LINE = -27;
		literal int KP4_LINE = -28;
		literal int KP5_LINE = -29;
		literal int L1L2_LINE = -30;
		literal int L1L3_LINE = -31;
		literal int L1M1_LINE = -32;
		literal int L1M2_LINE = -33;
		literal int L1M3_LINE = -34;
		literal int L1M4_LINE = -35;
		literal int L1M5_LINE = -36;
		literal int L1N1_LINE = -37;
		literal int L1N2_LINE = -38;
		literal int L1N3_LINE = -39;
		literal int L1N4_LINE = -40;
		literal int L1N5_LINE = -41;
		literal int L1N6_LINE = -42;
		literal int L1N67_LINE = -43;
		literal int L1N7_LINE = -44;
		literal int L1O1_LINE = -45;
		literal int L1O2_LINE = -46;
		literal int L1O3_LINE = -47;
		literal int L1O4_LINE = -48;
		literal int L1O45_LINE = -49;
		literal int L1O5_LINE = -50;
		literal int L1O6_LINE = -51;
		literal int L1O7_LINE = -52;
		literal int L1P1_LINE = -53;
		literal int L1P2_LINE = -54;
		literal int L1P23_LINE = -55;
		literal int L1P3_LINE = -56;
		literal int L1P4_LINE = -57;
		literal int L1P5_LINE = -58;
		literal int L2L3_LINE = -59;
		literal int L2M1_LINE = -60;
		literal int L2M2_LINE = -61;
		literal int L2M3_LINE = -62;
		literal int L2M4_LINE = -63;
		literal int L2M5_LINE = -64;
		literal int L2N1_LINE = -65;
		literal int L2N2_LINE = -66;
		literal int L2N3_LINE = -67;
		literal int L2N4_LINE = -68;
		literal int L2N5_LINE = -69;
		literal int L2N6_LINE = -70;
		literal int L2N7_LINE = -71;
		literal int L2O1_LINE = -72;
		literal int L2O2_LINE = -73;
		literal int L2O3_LINE = -74;
		literal int L2O4_LINE = -75;
		literal int L2O5_LINE = -76;
		literal int L2O6_LINE = -77;
		literal int L2O7_LINE = -78;
		literal int L2P1_LINE = -79;
		literal int L2P2_LINE = -80;
		literal int L2P23_LINE = -81;
		literal int L2P3_LINE = -82;
		literal int L2P4_LINE = -83;
		literal int L2P5_LINE = -84;
		literal int L2Q1_LINE = -85;
		literal int L3M1_LINE = -86;
		literal int L3M2_LINE = -87;
		literal int L3M3_LINE = -88;
		literal int L3M4_LINE = -89;
		literal int L3M5_LINE = -90;
		literal int L3N1_LINE = -91;
		literal int L3N2_LINE = -92;
		literal int L3N3_LINE = -93;
		literal int L3N4_LINE = -94;
		literal int L3N5_LINE = -95;
		literal int L3N6_LINE = -96;
		literal int L3N7_LINE = -97;
		literal int L3O1_LINE = -98;
		literal int L3O2_LINE = -99;
		literal int L3O3_LINE = -100;
		literal int L3O4_LINE = -101;
		literal int L3O45_LINE = -102;
		literal int L3O5_LINE = -103;
		literal int L3O6_LINE = -104;
		literal int L3O7_LINE = -105;
		literal int L3P1_LINE = -106;
		literal int L3P2_LINE = -107;
		literal int L3P23_LINE = -108;
		literal int L3P3_LINE = -109;
		literal int L3P4_LINE = -110;
		literal int L3P45_LINE = -111;
		literal int L3P5_LINE = -112;
		literal int L3Q1_LINE = -113;
		literal int M1M2_LINE = -114;
		literal int M1M3_LINE = -115;
		literal int M1M4_LINE = -116;
		literal int M1M5_LINE = -117;
		literal int M1N1_LINE = -118;
		literal int M1N2_LINE = -119;
		literal int M1N3_LINE = -120;
		literal int M1N4_LINE = -121;
		literal int M1N5_LINE = -122;
		literal int M1N6_LINE = -123;
		literal int M1N7_LINE = -124;
		literal int M1O1_LINE = -125;
		literal int M1O2_LINE = -126;
		literal int M1O3_LINE = -127;
		literal int M1O4_LINE = -128;
		literal int M1O5_LINE = -129;
		literal int M1O6_LINE = -130;
		literal int M1O7_LINE = -131;
		literal int M1P1_LINE = -132;
		literal int M1P2_LINE = -133;
		literal int M1P3_LINE = -134;
		literal int M1P4_LINE = -135;
		literal int M1P5_LINE = -136;
		literal int M2M3_LINE = -137;
		literal int M2M4_LINE = -138;
		literal int M2M5_LINE = -139;
		literal int M2N1_LINE = -140;
		literal int M2N2_LINE = -141;
		literal int M2N3_LINE = -142;
		literal int M2N4_LINE = -143;
		literal int M2N5_LINE = -144;
		literal int M2N6_LINE = -145;
		literal int M2N7_LINE = -146;
		literal int M2O1_LINE = -147;
		literal int M2O2_LINE = -148;
		literal int M2O3_LINE = -149;
		literal int M2O4_LINE = -150;
		literal int M2O5_LINE = -151;
		literal int M2O6_LINE = -152;
		literal int M2O7_LINE = -153;
		literal int M2P1_LINE = -154;
		literal int M2P2_LINE = -155;
		literal int M2P3_LINE = -156;
		literal int M2P4_LINE = -157;
		literal int M2P5_LINE = -158;
		literal int M3M4_LINE = -159;
		literal int M3M5_LINE = -160;
		literal int M3N1_LINE = -161;
		literal int M3N2_LINE = -162;
		literal int M3N3_LINE = -163;
		literal int M3N4_LINE = -164;
		literal int M3N5_LINE = -165;
		literal int M3N6_LINE = -166;
		literal int M3N7_LINE = -167;
		literal int M3O1_LINE = -168;
		literal int M3O2_LINE = -169;
		literal int M3O3_LINE = -170;
		literal int M3O4_LINE = -171;
		literal int M3O5_LINE = -172;
		literal int M3O6_LINE = -173;
		literal int M3O7_LINE = -174;
		literal int M3P1_LINE = -175;
		literal int M3P2_LINE = -176;
		literal int M3P3_LINE = -177;
		literal int M3P4_LINE = -178;
		literal int M3P5_LINE = -179;
		literal int M3Q1_LINE = -180;
		literal int M4M5_LINE = -181;
		literal int M4N1_LINE = -182;
		literal int M4N2_LINE = -183;
		literal int M4N3_LINE = -184;
		literal int M4N4_LINE = -185;
		literal int M4N5_LINE = -186;
		literal int M4N6_LINE = -187;
		literal int M4N7_LINE = -188;
		literal int M4O1_LINE = -189;
		literal int M4O2_LINE = -190;
		literal int M4O3_LINE = -191;
		literal int M4O4_LINE = -192;
		literal int M4O5_LINE = -193;
		literal int M4O6_LINE = -194;
		literal int M4O7_LINE = -195;
		literal int M4P1_LINE = -196;
		literal int M4P2_LINE = -197;
		literal int M4P3_LINE = -198;
		literal int M4P4_LINE = -199;
		literal int M4P5_LINE = -200;
		literal int M5N1_LINE = -201;
		literal int M5N2_LINE = -202;
		literal int M5N3_LINE = -203;
		literal int M5N4_LINE = -204;
		literal int M5N5_LINE = -205;
		literal int M5N6_LINE = -206;
		literal int M5N7_LINE = -207;
		literal int M5O1_LINE = -208;
		literal int M5O2_LINE = -209;
		literal int M5O3_LINE = -210;
		literal int M5O4_LINE = -211;
		literal int M5O5_LINE = -212;
		literal int M5O6_LINE = -213;
		literal int M5O7_LINE = -214;
		literal int M5P1_LINE = -215;
		literal int M5P2_LINE = -216;
		literal int M5P3_LINE = -217;
		literal int M5P4_LINE = -218;
		literal int M5P5_LINE = -219;
		literal int N1N2_LINE = -220;
		literal int N1N3_LINE = -221;
		literal int N1N4_LINE = -222;
		literal int N1N5_LINE = -223;
		literal int N1N6_LINE = -224;
		literal int N1N7_LINE = -225;
		literal int N1O1_LINE = -226;
		literal int N1O2_LINE = -227;
		literal int N1O3_LINE = -228;
		literal int N1O4_LINE = -229;
		literal int N1O5_LINE = -230;
		literal int N1O6_LINE = -231;
		literal int N1O7_LINE = -232;
		literal int N1P1_LINE = -233;
		literal int N1P2_LINE = -234;
		literal int N1P3_LINE = -235;
		literal int N1P4_LINE = -236;
		literal int N1P5_LINE = -237;
		literal int N2N3_LINE = -238;
		literal int N2N4_LINE = -239;
		literal int N2N5_LINE = -240;
		literal int N2N6_LINE = -241;
		literal int N2N7_LINE = -242;
		literal int N2O1_LINE = -243;
		literal int N2O2_LINE = -244;
		literal int N2O3_LINE = -245;
		literal int N2O4_LINE = -246;
		literal int N2O5_LINE = -247;
		literal int N2O6_LINE = -248;
		literal int N2O7_LINE = -249;
		literal int N2P1_LINE = -250;
		literal int N2P2_LINE = -251;
		literal int N2P3_LINE = -252;
		literal int N2P4_LINE = -253;
		literal int N2P5_LINE = -254;
		literal int N3N4_LINE = -255;
		literal int N3N5_LINE = -256;
		literal int N3N6_LINE = -257;
		literal int N3N7_LINE = -258;
		literal int N3O1_LINE = -259;
		literal int N3O2_LINE = -260;
		literal int N3O3_LINE = -261;
		literal int N3O4_LINE = -262;
		literal int N3O5_LINE = -263;
		literal int N3O6_LINE = -264;
		literal int N3O7_LINE = -265;
		literal int N3P1_LINE = -266;
		literal int N3P2_LINE = -267;
		literal int N3P3_LINE = -268;
		literal int N3P4_LINE = -269;
		literal int N3P5_LINE = -270;
		literal int N4N5_LINE = -271;
		literal int N4N6_LINE = -272;
		literal int N4N7_LINE = -273;
		literal int N4O1_LINE = -274;
		literal int N4O2_LINE = -275;
		literal int N4O3_LINE = -276;
		literal int N4O4_LINE = -277;
		literal int N4O5_LINE = -278;
		literal int N4O6_LINE = -279;
		literal int N4O7_LINE = -280;
		literal int N4P1_LINE = -281;
		literal int N4P2_LINE = -282;
		literal int N4P3_LINE = -283;
		literal int N4P4_LINE = -284;
		literal int N4P5_LINE = -285;
		literal int N5N6_LINE = -286;
		literal int N5N7_LINE = -287;
		literal int N5O1_LINE = -288;
		literal int N5O2_LINE = -289;
		literal int N5O3_LINE = -290;
		literal int N5O4_LINE = -291;
		literal int N5O5_LINE = -292;
		literal int N5O6_LINE = -293;
		literal int N5O7_LINE = -294;
		literal int N5P1_LINE = -295;
		literal int N5P2_LINE = -296;
		literal int N5P3_LINE = -297;
		literal int N5P4_LINE = -298;
		literal int N5P5_LINE = -299;
		literal int N6N7_LINE = -300;
		literal int N6O1_LINE = -301;
		literal int N6O2_LINE = -302;
		literal int N6O3_LINE = -303;
		literal int N6O4_LINE = -304;
		literal int N6O5_LINE = -305;
		literal int N6O6_LINE = -306;
		literal int N6O7_LINE = -307;
		literal int N6P1_LINE = -308;
		literal int N6P2_LINE = -309;
		literal int N6P3_LINE = -310;
		literal int N6P4_LINE = -311;
		literal int N6P5_LINE = -312;
		literal int N7O1_LINE = -313;
		literal int N7O2_LINE = -314;
		literal int N7O3_LINE = -315;
		literal int N7O4_LINE = -316;
		literal int N7O5_LINE = -317;
		literal int N7O6_LINE = -318;
		literal int N7O7_LINE = -319;
		literal int N7P1_LINE = -320;
		literal int N7P2_LINE = -321;
		literal int N7P3_LINE = -322;
		literal int N7P4_LINE = -323;
		literal int N7P5_LINE = -324;
		literal int O1O2_LINE = -325;
		literal int O1O3_LINE = -326;
		literal int O1O4_LINE = -327;
		literal int O1O5_LINE = -328;
		literal int O1O6_LINE = -329;
		literal int O1O7_LINE = -330;
		literal int O1P1_LINE = -331;
		literal int O1P2_LINE = -332;
		literal int O1P3_LINE = -333;
		literal int O1P4_LINE = -334;
		literal int O1P5_LINE = -335;
		literal int O2O3_LINE = -336;
		literal int O2O4_LINE = -337;
		literal int O2O5_LINE = -338;
		literal int O2O6_LINE = -339;
		literal int O2O7_LINE = -340;
		literal int O2P1_LINE = -341;
		literal int O2P2_LINE = -342;
		literal int O2P3_LINE = -343;
		literal int O2P4_LINE = -344;
		literal int O2P5_LINE = -345;
		literal int O3O4_LINE = -346;
		literal int O3O5_LINE = -347;
		literal int O3O6_LINE = -348;
		literal int O3O7_LINE = -349;
		literal int O3P1_LINE = -350;
		literal int O3P2_LINE = -351;
		literal int O3P3_LINE = -352;
		literal int O3P4_LINE = -353;
		literal int O3P5_LINE = -354;
		literal int O4O5_LINE = -355;
		literal int O4O6_LINE = -356;
		literal int O4O7_LINE = -357;
		literal int O4P1_LINE = -358;
		literal int O4P2_LINE = -359;
		literal int O4P3_LINE = -360;
		literal int O4P4_LINE = -361;
		literal int O4P5_LINE = -362;
		literal int O5O6_LINE = -363;
		literal int O5O7_LINE = -364;
		literal int O5P1_LINE = -365;
		literal int O5P2_LINE = -366;
		literal int O5P3_LINE = -367;
		literal int O5P4_LINE = -368;
		literal int O5P5_LINE = -369;
		literal int O6O7_LINE = -370;
		literal int O6P4_LINE = -371;
		literal int O6P5_LINE = -372;
		literal int O7P4_LINE = -373;
		literal int O7P5_LINE = -374;
		literal int P1P2_LINE = -375;
		literal int P1P3_LINE = -376;
		literal int P1P4_LINE = -377;
		literal int P1P5_LINE = -378;
		literal int P2P3_LINE = -379;
		literal int P2P4_LINE = -380;
		literal int P2P5_LINE = -381;
		literal int P3P4_LINE = -382;
		literal int P3P5_LINE = -383;

		// Siegbahn notation
		// according to Table VIII.2 from Nomenclature system for X-ray spectroscopy
		// Linegroups -> usage is discouraged
		literal int KA_LINE = 0;            // Weighted average of KA1 & KA2
		literal int KB_LINE = 1;            // Weighted average of KB1 & KB3
		literal int LA_LINE = 2;            // LA1
		literal int LB_LINE = 3;            // LB1

  		// Single lines
		literal int KA1_LINE = KL3_LINE;
		literal int KA2_LINE = KL2_LINE;
		literal int KA3_LINE = KL1_LINE;
		literal int KB1_LINE = KM3_LINE;
		literal int KB2_LINE = KN3_LINE;
		literal int KB3_LINE = KM2_LINE;
		literal int KB4_LINE = KN5_LINE;
		literal int KB5_LINE = KM5_LINE;

		literal int LA1_LINE = L3M5_LINE;
		literal int LA2_LINE = L3M4_LINE;
		literal int LB1_LINE = L2M4_LINE;
		literal int LB2_LINE = L3N5_LINE;
		literal int LB3_LINE = L1M3_LINE;
		literal int LB4_LINE = L1M2_LINE;
		literal int LB5_LINE = L3O45_LINE;
		literal int LB6_LINE = L3N1_LINE;
		literal int LB7_LINE = L3O1_LINE;
		literal int LB9_LINE = L1M5_LINE;
		literal int LB10_LINE = L1M4_LINE;
		literal int LB15_LINE = L3N4_LINE;
		literal int LB17_LINE = L2M3_LINE;
		literal int LG1_LINE = L2N4_LINE;
		literal int LG2_LINE = L1N2_LINE;
		literal int LG3_LINE = L1N3_LINE;
		literal int LG4_LINE = L1O3_LINE;
		literal int LG5_LINE = L2N1_LINE;
		literal int LG6_LINE = L2O4_LINE;
		literal int LG8_LINE = L2O1_LINE;
		literal int LE_LINE = L2M1_LINE;
		literal int LH_LINE = L2M1_LINE;
		literal int LN_LINE = L2M1_LINE;
		literal int LL_LINE = L3M1_LINE;
		literal int LS_LINE = L3M3_LINE;
		literal int LT_LINE = L3M2_LINE;
		literal int LU_LINE = L3N6_LINE;
		literal int LV_LINE = L2N6_LINE;

		literal int MA1_LINE = M5N7_LINE;
		literal int MA2_LINE = M5N6_LINE;
		literal int MB_LINE = M4N6_LINE;
		literal int MG_LINE = M3N5_LINE;

		// Shells
		literal int K_SHELL = 0;
		literal int L1_SHELL = 1;
		literal int L2_SHELL = 2;
		literal int L3_SHELL = 3;
		literal int M1_SHELL = 4;
		literal int M2_SHELL = 5;
		literal int M3_SHELL = 6;
		literal int M4_SHELL = 7;
		literal int M5_SHELL = 8;
		literal int N1_SHELL = 9;
		literal int N2_SHELL = 10;
		literal int N3_SHELL = 11;
		literal int N4_SHELL = 12;
		literal int N5_SHELL = 13;
		literal int N6_SHELL = 14;
		literal int N7_SHELL = 15;
		literal int O1_SHELL = 16;
		literal int O2_SHELL = 17;
		literal int O3_SHELL = 18;
		literal int O4_SHELL = 19;
		literal int O5_SHELL = 20;
		literal int O6_SHELL = 21;
		literal int O7_SHELL = 22;
		literal int P1_SHELL = 23;
		literal int P2_SHELL = 24;
		literal int P3_SHELL = 25;
		literal int P4_SHELL = 26;
		literal int P5_SHELL = 27;
		literal int Q1_SHELL = 28;
		literal int Q2_SHELL = 29;
		literal int Q3_SHELL = 30;

		// Transitions
		literal int F1_TRANS = 0;
		literal int F12_TRANS = 1;
		literal int F13_TRANS = 2;
		literal int FP13_TRANS = 3;
		literal int F23_TRANS = 4;

		literal int FL12_TRANS = 1;
		literal int FL13_TRANS = 2;
		literal int FLP13_TRANS = 3;
		literal int FL23_TRANS = 4;
		literal int FM12_TRANS = 5;
		literal int FM13_TRANS = 6;
		literal int FM14_TRANS = 7;
		literal int FM15_TRANS = 8;
		literal int FM23_TRANS = 9;
		literal int FM24_TRANS = 10;
		literal int FM25_TRANS = 11;
		literal int FM34_TRANS = 12;
		literal int FM35_TRANS = 13;
		literal int FM45_TRANS = 14;

		// Auger transitions
		literal int K_L1L1_AUGER = 0;
		literal int K_L1L2_AUGER = 1;
		literal int K_L1L3_AUGER = 2;
		literal int K_L1M1_AUGER = 3;
		literal int K_L1M2_AUGER = 4;
		literal int K_L1M3_AUGER = 5;
		literal int K_L1M4_AUGER = 6;
		literal int K_L1M5_AUGER = 7;
		literal int K_L1N1_AUGER = 8;
		literal int K_L1N2_AUGER = 9;
		literal int K_L1N3_AUGER = 10;
		literal int K_L1N4_AUGER = 11;
		literal int K_L1N5_AUGER = 12;
		literal int K_L1N6_AUGER = 13;
		literal int K_L1N7_AUGER = 14;
		literal int K_L1O1_AUGER = 15;
		literal int K_L1O2_AUGER = 16;
		literal int K_L1O3_AUGER = 17;
		literal int K_L1O4_AUGER = 18;
		literal int K_L1O5_AUGER = 19;
		literal int K_L1O6_AUGER = 20;
		literal int K_L1O7_AUGER = 21;
		literal int K_L1P1_AUGER = 22;
		literal int K_L1P2_AUGER = 23;
		literal int K_L1P3_AUGER = 24;
		literal int K_L1P4_AUGER = 25;
		literal int K_L1P5_AUGER = 26;
		literal int K_L1Q1_AUGER = 27;
		literal int K_L1Q2_AUGER = 28;
		literal int K_L1Q3_AUGER = 29;
		literal int K_L2L1_AUGER = 30;
		literal int K_L2L2_AUGER = 31;
		literal int K_L2L3_AUGER = 32;
		literal int K_L2M1_AUGER = 33;
		literal int K_L2M2_AUGER = 34;
		literal int K_L2M3_AUGER = 35;
		literal int K_L2M4_AUGER = 36;
		literal int K_L2M5_AUGER = 37;
		literal int K_L2N1_AUGER = 38;
		literal int K_L2N2_AUGER = 39;
		literal int K_L2N3_AUGER = 40;
		literal int K_L2N4_AUGER = 41;
		literal int K_L2N5_AUGER = 42;
		literal int K_L2N6_AUGER = 43;
		literal int K_L2N7_AUGER = 44;
		literal int K_L2O1_AUGER = 45;
		literal int K_L2O2_AUGER = 46;
		literal int K_L2O3_AUGER = 47;
		literal int K_L2O4_AUGER = 48;
		literal int K_L2O5_AUGER = 49;
		literal int K_L2O6_AUGER = 50;
		literal int K_L2O7_AUGER = 51;
		literal int K_L2P1_AUGER = 52;
		literal int K_L2P2_AUGER = 53;
		literal int K_L2P3_AUGER = 54;
		literal int K_L2P4_AUGER = 55;
		literal int K_L2P5_AUGER = 56;
		literal int K_L2Q1_AUGER = 57;
		literal int K_L2Q2_AUGER = 58;
		literal int K_L2Q3_AUGER = 59;
		literal int K_L3L1_AUGER = 60;
		literal int K_L3L2_AUGER = 61;
		literal int K_L3L3_AUGER = 62;
		literal int K_L3M1_AUGER = 63;
		literal int K_L3M2_AUGER = 64;
		literal int K_L3M3_AUGER = 65;
		literal int K_L3M4_AUGER = 66;
		literal int K_L3M5_AUGER = 67;
		literal int K_L3N1_AUGER = 68;
		literal int K_L3N2_AUGER = 69;
		literal int K_L3N3_AUGER = 70;
		literal int K_L3N4_AUGER = 71;
		literal int K_L3N5_AUGER = 72;
		literal int K_L3N6_AUGER = 73;
		literal int K_L3N7_AUGER = 74;
		literal int K_L3O1_AUGER = 75;
		literal int K_L3O2_AUGER = 76;
		literal int K_L3O3_AUGER = 77;
		literal int K_L3O4_AUGER = 78;
		literal int K_L3O5_AUGER = 79;
		literal int K_L3O6_AUGER = 80;
		literal int K_L3O7_AUGER = 81;
		literal int K_L3P1_AUGER = 82;
		literal int K_L3P2_AUGER = 83;
		literal int K_L3P3_AUGER = 84;
		literal int K_L3P4_AUGER = 85;
		literal int K_L3P5_AUGER = 86;
		literal int K_L3Q1_AUGER = 87;
		literal int K_L3Q2_AUGER = 88;
		literal int K_L3Q3_AUGER = 89;
		literal int K_M1L1_AUGER = 90;
		literal int K_M1L2_AUGER = 91;
		literal int K_M1L3_AUGER = 92;
		literal int K_M1M1_AUGER = 93;
		literal int K_M1M2_AUGER = 94;
		literal int K_M1M3_AUGER = 95;
		literal int K_M1M4_AUGER = 96;
		literal int K_M1M5_AUGER = 97;
		literal int K_M1N1_AUGER = 98;
		literal int K_M1N2_AUGER = 99;
		literal int K_M1N3_AUGER = 100;
		literal int K_M1N4_AUGER = 101;
		literal int K_M1N5_AUGER = 102;
		literal int K_M1N6_AUGER = 103;
		literal int K_M1N7_AUGER = 104;
		literal int K_M1O1_AUGER = 105;
		literal int K_M1O2_AUGER = 106;
		literal int K_M1O3_AUGER = 107;
		literal int K_M1O4_AUGER = 108;
		literal int K_M1O5_AUGER = 109;
		literal int K_M1O6_AUGER = 110;
		literal int K_M1O7_AUGER = 111;
		literal int K_M1P1_AUGER = 112;
		literal int K_M1P2_AUGER = 113;
		literal int K_M1P3_AUGER = 114;
		literal int K_M1P4_AUGER = 115;
		literal int K_M1P5_AUGER = 116;
		literal int K_M1Q1_AUGER = 117;
		literal int K_M1Q2_AUGER = 118;
		literal int K_M1Q3_AUGER = 119;
		literal int K_M2L1_AUGER = 120;
		literal int K_M2L2_AUGER = 121;
		literal int K_M2L3_AUGER = 122;
		literal int K_M2M1_AUGER = 123;
		literal int K_M2M2_AUGER = 124;
		literal int K_M2M3_AUGER = 125;
		literal int K_M2M4_AUGER = 126;
		literal int K_M2M5_AUGER = 127;
		literal int K_M2N1_AUGER = 128;
		literal int K_M2N2_AUGER = 129;
		literal int K_M2N3_AUGER = 130;
		literal int K_M2N4_AUGER = 131;
		literal int K_M2N5_AUGER = 132;
		literal int K_M2N6_AUGER = 133;
		literal int K_M2N7_AUGER = 134;
		literal int K_M2O1_AUGER = 135;
		literal int K_M2O2_AUGER = 136;
		literal int K_M2O3_AUGER = 137;
		literal int K_M2O4_AUGER = 138;
		literal int K_M2O5_AUGER = 139;
		literal int K_M2O6_AUGER = 140;
		literal int K_M2O7_AUGER = 141;
		literal int K_M2P1_AUGER = 142;
		literal int K_M2P2_AUGER = 143;
		literal int K_M2P3_AUGER = 144;
		literal int K_M2P4_AUGER = 145;
		literal int K_M2P5_AUGER = 146;
		literal int K_M2Q1_AUGER = 147;
		literal int K_M2Q2_AUGER = 148;
		literal int K_M2Q3_AUGER = 149;
		literal int K_M3L1_AUGER = 150;
		literal int K_M3L2_AUGER = 151;
		literal int K_M3L3_AUGER = 152;
		literal int K_M3M1_AUGER = 153;
		literal int K_M3M2_AUGER = 154;
		literal int K_M3M3_AUGER = 155;
		literal int K_M3M4_AUGER = 156;
		literal int K_M3M5_AUGER = 157;
		literal int K_M3N1_AUGER = 158;
		literal int K_M3N2_AUGER = 159;
		literal int K_M3N3_AUGER = 160;
		literal int K_M3N4_AUGER = 161;
		literal int K_M3N5_AUGER = 162;
		literal int K_M3N6_AUGER = 163;
		literal int K_M3N7_AUGER = 164;
		literal int K_M3O1_AUGER = 165;
		literal int K_M3O2_AUGER = 166;
		literal int K_M3O3_AUGER = 167;
		literal int K_M3O4_AUGER = 168;
		literal int K_M3O5_AUGER = 169;
		literal int K_M3O6_AUGER = 170;
		literal int K_M3O7_AUGER = 171;
		literal int K_M3P1_AUGER = 172;
		literal int K_M3P2_AUGER = 173;
		literal int K_M3P3_AUGER = 174;
		literal int K_M3P4_AUGER = 175;
		literal int K_M3P5_AUGER = 176;
		literal int K_M3Q1_AUGER = 177;
		literal int K_M3Q2_AUGER = 178;
		literal int K_M3Q3_AUGER = 179;
		literal int K_M4L1_AUGER = 180;
		literal int K_M4L2_AUGER = 181;
		literal int K_M4L3_AUGER = 182;
		literal int K_M4M1_AUGER = 183;
		literal int K_M4M2_AUGER = 184;
		literal int K_M4M3_AUGER = 185;
		literal int K_M4M4_AUGER = 186;
		literal int K_M4M5_AUGER = 187;
		literal int K_M4N1_AUGER = 188;
		literal int K_M4N2_AUGER = 189;
		literal int K_M4N3_AUGER = 190;
		literal int K_M4N4_AUGER = 191;
		literal int K_M4N5_AUGER = 192;
		literal int K_M4N6_AUGER = 193;
		literal int K_M4N7_AUGER = 194;
		literal int K_M4O1_AUGER = 195;
		literal int K_M4O2_AUGER = 196;
		literal int K_M4O3_AUGER = 197;
		literal int K_M4O4_AUGER = 198;
		literal int K_M4O5_AUGER = 199;
		literal int K_M4O6_AUGER = 200;
		literal int K_M4O7_AUGER = 201;
		literal int K_M4P1_AUGER = 202;
		literal int K_M4P2_AUGER = 203;
		literal int K_M4P3_AUGER = 204;
		literal int K_M4P4_AUGER = 205;
		literal int K_M4P5_AUGER = 206;
		literal int K_M4Q1_AUGER = 207;
		literal int K_M4Q2_AUGER = 208;
		literal int K_M4Q3_AUGER = 209;
		literal int K_M5L1_AUGER = 210;
		literal int K_M5L2_AUGER = 211;
		literal int K_M5L3_AUGER = 212;
		literal int K_M5M1_AUGER = 213;
		literal int K_M5M2_AUGER = 214;
		literal int K_M5M3_AUGER = 215;
		literal int K_M5M4_AUGER = 216;
		literal int K_M5M5_AUGER = 217;
		literal int K_M5N1_AUGER = 218;
		literal int K_M5N2_AUGER = 219;
		literal int K_M5N3_AUGER = 220;
		literal int K_M5N4_AUGER = 221;
		literal int K_M5N5_AUGER = 222;
		literal int K_M5N6_AUGER = 223;
		literal int K_M5N7_AUGER = 224;
		literal int K_M5O1_AUGER = 225;
		literal int K_M5O2_AUGER = 226;
		literal int K_M5O3_AUGER = 227;
		literal int K_M5O4_AUGER = 228;
		literal int K_M5O5_AUGER = 229;
		literal int K_M5O6_AUGER = 230;
		literal int K_M5O7_AUGER = 231;
		literal int K_M5P1_AUGER = 232;
		literal int K_M5P2_AUGER = 233;
		literal int K_M5P3_AUGER = 234;
		literal int K_M5P4_AUGER = 235;
		literal int K_M5P5_AUGER = 236;
		literal int K_M5Q1_AUGER = 237;
		literal int K_M5Q2_AUGER = 238;
		literal int K_M5Q3_AUGER = 239;
		literal int L1_L2L2_AUGER = 240;
		literal int L1_L2L3_AUGER = 241;
		literal int L1_L2M1_AUGER = 242;
		literal int L1_L2M2_AUGER = 243;
		literal int L1_L2M3_AUGER = 244;
		literal int L1_L2M4_AUGER = 245;
		literal int L1_L2M5_AUGER = 246;
		literal int L1_L2N1_AUGER = 247;
		literal int L1_L2N2_AUGER = 248;
		literal int L1_L2N3_AUGER = 249;
		literal int L1_L2N4_AUGER = 250;
		literal int L1_L2N5_AUGER = 251;
		literal int L1_L2N6_AUGER = 252;
		literal int L1_L2N7_AUGER = 253;
		literal int L1_L2O1_AUGER = 254;
		literal int L1_L2O2_AUGER = 255;
		literal int L1_L2O3_AUGER = 256;
		literal int L1_L2O4_AUGER = 257;
		literal int L1_L2O5_AUGER = 258;
		literal int L1_L2O6_AUGER = 259;
		literal int L1_L2O7_AUGER = 260;
		literal int L1_L2P1_AUGER = 261;
		literal int L1_L2P2_AUGER = 262;
		literal int L1_L2P3_AUGER = 263;
		literal int L1_L2P4_AUGER = 264;
		literal int L1_L2P5_AUGER = 265;
		literal int L1_L2Q1_AUGER = 266;
		literal int L1_L2Q2_AUGER = 267;
		literal int L1_L2Q3_AUGER = 268;
		literal int L1_L3L2_AUGER = 269;
		literal int L1_L3L3_AUGER = 270;
		literal int L1_L3M1_AUGER = 271;
		literal int L1_L3M2_AUGER = 272;
		literal int L1_L3M3_AUGER = 273;
		literal int L1_L3M4_AUGER = 274;
		literal int L1_L3M5_AUGER = 275;
		literal int L1_L3N1_AUGER = 276;
		literal int L1_L3N2_AUGER = 277;
		literal int L1_L3N3_AUGER = 278;
		literal int L1_L3N4_AUGER = 279;
		literal int L1_L3N5_AUGER = 280;
		literal int L1_L3N6_AUGER = 281;
		literal int L1_L3N7_AUGER = 282;
		literal int L1_L3O1_AUGER = 283;
		literal int L1_L3O2_AUGER = 284;
		literal int L1_L3O3_AUGER = 285;
		literal int L1_L3O4_AUGER = 286;
		literal int L1_L3O5_AUGER = 287;
		literal int L1_L3O6_AUGER = 288;
		literal int L1_L3O7_AUGER = 289;
		literal int L1_L3P1_AUGER = 290;
		literal int L1_L3P2_AUGER = 291;
		literal int L1_L3P3_AUGER = 292;
		literal int L1_L3P4_AUGER = 293;
		literal int L1_L3P5_AUGER = 294;
		literal int L1_L3Q1_AUGER = 295;
		literal int L1_L3Q2_AUGER = 296;
		literal int L1_L3Q3_AUGER = 297;
		literal int L1_M1L2_AUGER = 298;
		literal int L1_M1L3_AUGER = 299;
		literal int L1_M1M1_AUGER = 300;
		literal int L1_M1M2_AUGER = 301;
		literal int L1_M1M3_AUGER = 302;
		literal int L1_M1M4_AUGER = 303;
		literal int L1_M1M5_AUGER = 304;
		literal int L1_M1N1_AUGER = 305;
		literal int L1_M1N2_AUGER = 306;
		literal int L1_M1N3_AUGER = 307;
		literal int L1_M1N4_AUGER = 308;
		literal int L1_M1N5_AUGER = 309;
		literal int L1_M1N6_AUGER = 310;
		literal int L1_M1N7_AUGER = 311;
		literal int L1_M1O1_AUGER = 312;
		literal int L1_M1O2_AUGER = 313;
		literal int L1_M1O3_AUGER = 314;
		literal int L1_M1O4_AUGER = 315;
		literal int L1_M1O5_AUGER = 316;
		literal int L1_M1O6_AUGER = 317;
		literal int L1_M1O7_AUGER = 318;
		literal int L1_M1P1_AUGER = 319;
		literal int L1_M1P2_AUGER = 320;
		literal int L1_M1P3_AUGER = 321;
		literal int L1_M1P4_AUGER = 322;
		literal int L1_M1P5_AUGER = 323;
		literal int L1_M1Q1_AUGER = 324;
		literal int L1_M1Q2_AUGER = 325;
		literal int L1_M1Q3_AUGER = 326;
		literal int L1_M2L2_AUGER = 327;
		literal int L1_M2L3_AUGER = 328;
		literal int L1_M2M1_AUGER = 329;
		literal int L1_M2M2_AUGER = 330;
		literal int L1_M2M3_AUGER = 331;
		literal int L1_M2M4_AUGER = 332;
		literal int L1_M2M5_AUGER = 333;
		literal int L1_M2N1_AUGER = 334;
		literal int L1_M2N2_AUGER = 335;
		literal int L1_M2N3_AUGER = 336;
		literal int L1_M2N4_AUGER = 337;
		literal int L1_M2N5_AUGER = 338;
		literal int L1_M2N6_AUGER = 339;
		literal int L1_M2N7_AUGER = 340;
		literal int L1_M2O1_AUGER = 341;
		literal int L1_M2O2_AUGER = 342;
		literal int L1_M2O3_AUGER = 343;
		literal int L1_M2O4_AUGER = 344;
		literal int L1_M2O5_AUGER = 345;
		literal int L1_M2O6_AUGER = 346;
		literal int L1_M2O7_AUGER = 347;
		literal int L1_M2P1_AUGER = 348;
		literal int L1_M2P2_AUGER = 349;
		literal int L1_M2P3_AUGER = 350;
		literal int L1_M2P4_AUGER = 351;
		literal int L1_M2P5_AUGER = 352;
		literal int L1_M2Q1_AUGER = 353;
		literal int L1_M2Q2_AUGER = 354;
		literal int L1_M2Q3_AUGER = 355;
		literal int L1_M3L2_AUGER = 356;
		literal int L1_M3L3_AUGER = 357;
		literal int L1_M3M1_AUGER = 358;
		literal int L1_M3M2_AUGER = 359;
		literal int L1_M3M3_AUGER = 360;
		literal int L1_M3M4_AUGER = 361;
		literal int L1_M3M5_AUGER = 362;
		literal int L1_M3N1_AUGER = 363;
		literal int L1_M3N2_AUGER = 364;
		literal int L1_M3N3_AUGER = 365;
		literal int L1_M3N4_AUGER = 366;
		literal int L1_M3N5_AUGER = 367;
		literal int L1_M3N6_AUGER = 368;
		literal int L1_M3N7_AUGER = 369;
		literal int L1_M3O1_AUGER = 370;
		literal int L1_M3O2_AUGER = 371;
		literal int L1_M3O3_AUGER = 372;
		literal int L1_M3O4_AUGER = 373;
		literal int L1_M3O5_AUGER = 374;
		literal int L1_M3O6_AUGER = 375;
		literal int L1_M3O7_AUGER = 376;
		literal int L1_M3P1_AUGER = 377;
		literal int L1_M3P2_AUGER = 378;
		literal int L1_M3P3_AUGER = 379;
		literal int L1_M3P4_AUGER = 380;
		literal int L1_M3P5_AUGER = 381;
		literal int L1_M3Q1_AUGER = 382;
		literal int L1_M3Q2_AUGER = 383;
		literal int L1_M3Q3_AUGER = 384;
		literal int L1_M4L2_AUGER = 385;
		literal int L1_M4L3_AUGER = 386;
		literal int L1_M4M1_AUGER = 387;
		literal int L1_M4M2_AUGER = 388;
		literal int L1_M4M3_AUGER = 389;
		literal int L1_M4M4_AUGER = 390;
		literal int L1_M4M5_AUGER = 391;
		literal int L1_M4N1_AUGER = 392;
		literal int L1_M4N2_AUGER = 393;
		literal int L1_M4N3_AUGER = 394;
		literal int L1_M4N4_AUGER = 395;
		literal int L1_M4N5_AUGER = 396;
		literal int L1_M4N6_AUGER = 397;
		literal int L1_M4N7_AUGER = 398;
		literal int L1_M4O1_AUGER = 399;
		literal int L1_M4O2_AUGER = 400;
		literal int L1_M4O3_AUGER = 401;
		literal int L1_M4O4_AUGER = 402;
		literal int L1_M4O5_AUGER = 403;
		literal int L1_M4O6_AUGER = 404;
		literal int L1_M4O7_AUGER = 405;
		literal int L1_M4P1_AUGER = 406;
		literal int L1_M4P2_AUGER = 407;
		literal int L1_M4P3_AUGER = 408;
		literal int L1_M4P4_AUGER = 409;
		literal int L1_M4P5_AUGER = 410;
		literal int L1_M4Q1_AUGER = 411;
		literal int L1_M4Q2_AUGER = 412;
		literal int L1_M4Q3_AUGER = 413;
		literal int L1_M5L2_AUGER = 414;
		literal int L1_M5L3_AUGER = 415;
		literal int L1_M5M1_AUGER = 416;
		literal int L1_M5M2_AUGER = 417;
		literal int L1_M5M3_AUGER = 418;
		literal int L1_M5M4_AUGER = 419;
		literal int L1_M5M5_AUGER = 420;
		literal int L1_M5N1_AUGER = 421;
		literal int L1_M5N2_AUGER = 422;
		literal int L1_M5N3_AUGER = 423;
		literal int L1_M5N4_AUGER = 424;
		literal int L1_M5N5_AUGER = 425;
		literal int L1_M5N6_AUGER = 426;
		literal int L1_M5N7_AUGER = 427;
		literal int L1_M5O1_AUGER = 428;
		literal int L1_M5O2_AUGER = 429;
		literal int L1_M5O3_AUGER = 430;
		literal int L1_M5O4_AUGER = 431;
		literal int L1_M5O5_AUGER = 432;
		literal int L1_M5O6_AUGER = 433;
		literal int L1_M5O7_AUGER = 434;
		literal int L1_M5P1_AUGER = 435;
		literal int L1_M5P2_AUGER = 436;
		literal int L1_M5P3_AUGER = 437;
		literal int L1_M5P4_AUGER = 438;
		literal int L1_M5P5_AUGER = 439;
		literal int L1_M5Q1_AUGER = 440;
		literal int L1_M5Q2_AUGER = 441;
		literal int L1_M5Q3_AUGER = 442;
		literal int L2_L3L3_AUGER = 443;
		literal int L2_L3M1_AUGER = 444;
		literal int L2_L3M2_AUGER = 445;
		literal int L2_L3M3_AUGER = 446;
		literal int L2_L3M4_AUGER = 447;
		literal int L2_L3M5_AUGER = 448;
		literal int L2_L3N1_AUGER = 449;
		literal int L2_L3N2_AUGER = 450;
		literal int L2_L3N3_AUGER = 451;
		literal int L2_L3N4_AUGER = 452;
		literal int L2_L3N5_AUGER = 453;
		literal int L2_L3N6_AUGER = 454;
		literal int L2_L3N7_AUGER = 455;
		literal int L2_L3O1_AUGER = 456;
		literal int L2_L3O2_AUGER = 457;
		literal int L2_L3O3_AUGER = 458;
		literal int L2_L3O4_AUGER = 459;
		literal int L2_L3O5_AUGER = 460;
		literal int L2_L3O6_AUGER = 461;
		literal int L2_L3O7_AUGER = 462;
		literal int L2_L3P1_AUGER = 463;
		literal int L2_L3P2_AUGER = 464;
		literal int L2_L3P3_AUGER = 465;
		literal int L2_L3P4_AUGER = 466;
		literal int L2_L3P5_AUGER = 467;
		literal int L2_L3Q1_AUGER = 468;
		literal int L2_L3Q2_AUGER = 469;
		literal int L2_L3Q3_AUGER = 470;
		literal int L2_M1L3_AUGER = 471;
		literal int L2_M1M1_AUGER = 472;
		literal int L2_M1M2_AUGER = 473;
		literal int L2_M1M3_AUGER = 474;
		literal int L2_M1M4_AUGER = 475;
		literal int L2_M1M5_AUGER = 476;
		literal int L2_M1N1_AUGER = 477;
		literal int L2_M1N2_AUGER = 478;
		literal int L2_M1N3_AUGER = 479;
		literal int L2_M1N4_AUGER = 480;
		literal int L2_M1N5_AUGER = 481;
		literal int L2_M1N6_AUGER = 482;
		literal int L2_M1N7_AUGER = 483;
		literal int L2_M1O1_AUGER = 484;
		literal int L2_M1O2_AUGER = 485;
		literal int L2_M1O3_AUGER = 486;
		literal int L2_M1O4_AUGER = 487;
		literal int L2_M1O5_AUGER = 488;
		literal int L2_M1O6_AUGER = 489;
		literal int L2_M1O7_AUGER = 490;
		literal int L2_M1P1_AUGER = 491;
		literal int L2_M1P2_AUGER = 492;
		literal int L2_M1P3_AUGER = 493;
		literal int L2_M1P4_AUGER = 494;
		literal int L2_M1P5_AUGER = 495;
		literal int L2_M1Q1_AUGER = 496;
		literal int L2_M1Q2_AUGER = 497;
		literal int L2_M1Q3_AUGER = 498;
		literal int L2_M2L3_AUGER = 499;
		literal int L2_M2M1_AUGER = 500;
		literal int L2_M2M2_AUGER = 501;
		literal int L2_M2M3_AUGER = 502;
		literal int L2_M2M4_AUGER = 503;
		literal int L2_M2M5_AUGER = 504;
		literal int L2_M2N1_AUGER = 505;
		literal int L2_M2N2_AUGER = 506;
		literal int L2_M2N3_AUGER = 507;
		literal int L2_M2N4_AUGER = 508;
		literal int L2_M2N5_AUGER = 509;
		literal int L2_M2N6_AUGER = 510;
		literal int L2_M2N7_AUGER = 511;
		literal int L2_M2O1_AUGER = 512;
		literal int L2_M2O2_AUGER = 513;
		literal int L2_M2O3_AUGER = 514;
		literal int L2_M2O4_AUGER = 515;
		literal int L2_M2O5_AUGER = 516;
		literal int L2_M2O6_AUGER = 517;
		literal int L2_M2O7_AUGER = 518;
		literal int L2_M2P1_AUGER = 519;
		literal int L2_M2P2_AUGER = 520;
		literal int L2_M2P3_AUGER = 521;
		literal int L2_M2P4_AUGER = 522;
		literal int L2_M2P5_AUGER = 523;
		literal int L2_M2Q1_AUGER = 524;
		literal int L2_M2Q2_AUGER = 525;
		literal int L2_M2Q3_AUGER = 526;
		literal int L2_M3L3_AUGER = 527;
		literal int L2_M3M1_AUGER = 528;
		literal int L2_M3M2_AUGER = 529;
		literal int L2_M3M3_AUGER = 530;
		literal int L2_M3M4_AUGER = 531;
		literal int L2_M3M5_AUGER = 532;
		literal int L2_M3N1_AUGER = 533;
		literal int L2_M3N2_AUGER = 534;
		literal int L2_M3N3_AUGER = 535;
		literal int L2_M3N4_AUGER = 536;
		literal int L2_M3N5_AUGER = 537;
		literal int L2_M3N6_AUGER = 538;
		literal int L2_M3N7_AUGER = 539;
		literal int L2_M3O1_AUGER = 540;
		literal int L2_M3O2_AUGER = 541;
		literal int L2_M3O3_AUGER = 542;
		literal int L2_M3O4_AUGER = 543;
		literal int L2_M3O5_AUGER = 544;
		literal int L2_M3O6_AUGER = 545;
		literal int L2_M3O7_AUGER = 546;
		literal int L2_M3P1_AUGER = 547;
		literal int L2_M3P2_AUGER = 548;
		literal int L2_M3P3_AUGER = 549;
		literal int L2_M3P4_AUGER = 550;
		literal int L2_M3P5_AUGER = 551;
		literal int L2_M3Q1_AUGER = 552;
		literal int L2_M3Q2_AUGER = 553;
		literal int L2_M3Q3_AUGER = 554;
		literal int L2_M4L3_AUGER = 555;
		literal int L2_M4M1_AUGER = 556;
		literal int L2_M4M2_AUGER = 557;
		literal int L2_M4M3_AUGER = 558;
		literal int L2_M4M4_AUGER = 559;
		literal int L2_M4M5_AUGER = 560;
		literal int L2_M4N1_AUGER = 561;
		literal int L2_M4N2_AUGER = 562;
		literal int L2_M4N3_AUGER = 563;
		literal int L2_M4N4_AUGER = 564;
		literal int L2_M4N5_AUGER = 565;
		literal int L2_M4N6_AUGER = 566;
		literal int L2_M4N7_AUGER = 567;
		literal int L2_M4O1_AUGER = 568;
		literal int L2_M4O2_AUGER = 569;
		literal int L2_M4O3_AUGER = 570;
		literal int L2_M4O4_AUGER = 571;
		literal int L2_M4O5_AUGER = 572;
		literal int L2_M4O6_AUGER = 573;
		literal int L2_M4O7_AUGER = 574;
		literal int L2_M4P1_AUGER = 575;
		literal int L2_M4P2_AUGER = 576;
		literal int L2_M4P3_AUGER = 577;
		literal int L2_M4P4_AUGER = 578;
		literal int L2_M4P5_AUGER = 579;
		literal int L2_M4Q1_AUGER = 580;
		literal int L2_M4Q2_AUGER = 581;
		literal int L2_M4Q3_AUGER = 582;
		literal int L2_M5L3_AUGER = 583;
		literal int L2_M5M1_AUGER = 584;
		literal int L2_M5M2_AUGER = 585;
		literal int L2_M5M3_AUGER = 586;
		literal int L2_M5M4_AUGER = 587;
		literal int L2_M5M5_AUGER = 588;
		literal int L2_M5N1_AUGER = 589;
		literal int L2_M5N2_AUGER = 590;
		literal int L2_M5N3_AUGER = 591;
		literal int L2_M5N4_AUGER = 592;
		literal int L2_M5N5_AUGER = 593;
		literal int L2_M5N6_AUGER = 594;
		literal int L2_M5N7_AUGER = 595;
		literal int L2_M5O1_AUGER = 596;
		literal int L2_M5O2_AUGER = 597;
		literal int L2_M5O3_AUGER = 598;
		literal int L2_M5O4_AUGER = 599;
		literal int L2_M5O5_AUGER = 600;
		literal int L2_M5O6_AUGER = 601;
		literal int L2_M5O7_AUGER = 602;
		literal int L2_M5P1_AUGER = 603;
		literal int L2_M5P2_AUGER = 604;
		literal int L2_M5P3_AUGER = 605;
		literal int L2_M5P4_AUGER = 606;
		literal int L2_M5P5_AUGER = 607;
		literal int L2_M5Q1_AUGER = 608;
		literal int L2_M5Q2_AUGER = 609;
		literal int L2_M5Q3_AUGER = 610;
		literal int L3_M1M1_AUGER = 611;
		literal int L3_M1M2_AUGER = 612;
		literal int L3_M1M3_AUGER = 613;
		literal int L3_M1M4_AUGER = 614;
		literal int L3_M1M5_AUGER = 615;
		literal int L3_M1N1_AUGER = 616;
		literal int L3_M1N2_AUGER = 617;
		literal int L3_M1N3_AUGER = 618;
		literal int L3_M1N4_AUGER = 619;
		literal int L3_M1N5_AUGER = 620;
		literal int L3_M1N6_AUGER = 621;
		literal int L3_M1N7_AUGER = 622;
		literal int L3_M1O1_AUGER = 623;
		literal int L3_M1O2_AUGER = 624;
		literal int L3_M1O3_AUGER = 625;
		literal int L3_M1O4_AUGER = 626;
		literal int L3_M1O5_AUGER = 627;
		literal int L3_M1O6_AUGER = 628;
		literal int L3_M1O7_AUGER = 629;
		literal int L3_M1P1_AUGER = 630;
		literal int L3_M1P2_AUGER = 631;
		literal int L3_M1P3_AUGER = 632;
		literal int L3_M1P4_AUGER = 633;
		literal int L3_M1P5_AUGER = 634;
		literal int L3_M1Q1_AUGER = 635;
		literal int L3_M1Q2_AUGER = 636;
		literal int L3_M1Q3_AUGER = 637;
		literal int L3_M2M1_AUGER = 638;
		literal int L3_M2M2_AUGER = 639;
		literal int L3_M2M3_AUGER = 640;
		literal int L3_M2M4_AUGER = 641;
		literal int L3_M2M5_AUGER = 642;
		literal int L3_M2N1_AUGER = 643;
		literal int L3_M2N2_AUGER = 644;
		literal int L3_M2N3_AUGER = 645;
		literal int L3_M2N4_AUGER = 646;
		literal int L3_M2N5_AUGER = 647;
		literal int L3_M2N6_AUGER = 648;
		literal int L3_M2N7_AUGER = 649;
		literal int L3_M2O1_AUGER = 650;
		literal int L3_M2O2_AUGER = 651;
		literal int L3_M2O3_AUGER = 652;
		literal int L3_M2O4_AUGER = 653;
		literal int L3_M2O5_AUGER = 654;
		literal int L3_M2O6_AUGER = 655;
		literal int L3_M2O7_AUGER = 656;
		literal int L3_M2P1_AUGER = 657;
		literal int L3_M2P2_AUGER = 658;
		literal int L3_M2P3_AUGER = 659;
		literal int L3_M2P4_AUGER = 660;
		literal int L3_M2P5_AUGER = 661;
		literal int L3_M2Q1_AUGER = 662;
		literal int L3_M2Q2_AUGER = 663;
		literal int L3_M2Q3_AUGER = 664;
		literal int L3_M3M1_AUGER = 665;
		literal int L3_M3M2_AUGER = 666;
		literal int L3_M3M3_AUGER = 667;
		literal int L3_M3M4_AUGER = 668;
		literal int L3_M3M5_AUGER = 669;
		literal int L3_M3N1_AUGER = 670;
		literal int L3_M3N2_AUGER = 671;
		literal int L3_M3N3_AUGER = 672;
		literal int L3_M3N4_AUGER = 673;
		literal int L3_M3N5_AUGER = 674;
		literal int L3_M3N6_AUGER = 675;
		literal int L3_M3N7_AUGER = 676;
		literal int L3_M3O1_AUGER = 677;
		literal int L3_M3O2_AUGER = 678;
		literal int L3_M3O3_AUGER = 679;
		literal int L3_M3O4_AUGER = 680;
		literal int L3_M3O5_AUGER = 681;
		literal int L3_M3O6_AUGER = 682;
		literal int L3_M3O7_AUGER = 683;
		literal int L3_M3P1_AUGER = 684;
		literal int L3_M3P2_AUGER = 685;
		literal int L3_M3P3_AUGER = 686;
		literal int L3_M3P4_AUGER = 687;
		literal int L3_M3P5_AUGER = 688;
		literal int L3_M3Q1_AUGER = 689;
		literal int L3_M3Q2_AUGER = 690;
		literal int L3_M3Q3_AUGER = 691;
		literal int L3_M4M1_AUGER = 692;
		literal int L3_M4M2_AUGER = 693;
		literal int L3_M4M3_AUGER = 694;
		literal int L3_M4M4_AUGER = 695;
		literal int L3_M4M5_AUGER = 696;
		literal int L3_M4N1_AUGER = 697;
		literal int L3_M4N2_AUGER = 698;
		literal int L3_M4N3_AUGER = 699;
		literal int L3_M4N4_AUGER = 700;
		literal int L3_M4N5_AUGER = 701;
		literal int L3_M4N6_AUGER = 702;
		literal int L3_M4N7_AUGER = 703;
		literal int L3_M4O1_AUGER = 704;
		literal int L3_M4O2_AUGER = 705;
		literal int L3_M4O3_AUGER = 706;
		literal int L3_M4O4_AUGER = 707;
		literal int L3_M4O5_AUGER = 708;
		literal int L3_M4O6_AUGER = 709;
		literal int L3_M4O7_AUGER = 710;
		literal int L3_M4P1_AUGER = 711;
		literal int L3_M4P2_AUGER = 712;
		literal int L3_M4P3_AUGER = 713;
		literal int L3_M4P4_AUGER = 714;
		literal int L3_M4P5_AUGER = 715;
		literal int L3_M4Q1_AUGER = 716;
		literal int L3_M4Q2_AUGER = 717;
		literal int L3_M4Q3_AUGER = 718;
		literal int L3_M5M1_AUGER = 719;
		literal int L3_M5M2_AUGER = 720;
		literal int L3_M5M3_AUGER = 721;
		literal int L3_M5M4_AUGER = 722;
		literal int L3_M5M5_AUGER = 723;
		literal int L3_M5N1_AUGER = 724;
		literal int L3_M5N2_AUGER = 725;
		literal int L3_M5N3_AUGER = 726;
		literal int L3_M5N4_AUGER = 727;
		literal int L3_M5N5_AUGER = 728;
		literal int L3_M5N6_AUGER = 729;
		literal int L3_M5N7_AUGER = 730;
		literal int L3_M5O1_AUGER = 731;
		literal int L3_M5O2_AUGER = 732;
		literal int L3_M5O3_AUGER = 733;
		literal int L3_M5O4_AUGER = 734;
		literal int L3_M5O5_AUGER = 735;
		literal int L3_M5O6_AUGER = 736;
		literal int L3_M5O7_AUGER = 737;
		literal int L3_M5P1_AUGER = 738;
		literal int L3_M5P2_AUGER = 739;
		literal int L3_M5P3_AUGER = 740;
		literal int L3_M5P4_AUGER = 741;
		literal int L3_M5P5_AUGER = 742;
		literal int L3_M5Q1_AUGER = 743;
		literal int L3_M5Q2_AUGER = 744;
		literal int L3_M5Q3_AUGER = 745;
		literal int M1_M2M2_AUGER = 746;
		literal int M1_M2M3_AUGER = 747;
		literal int M1_M2M4_AUGER = 748;
		literal int M1_M2M5_AUGER = 749;
		literal int M1_M2N1_AUGER = 750;
		literal int M1_M2N2_AUGER = 751;
		literal int M1_M2N3_AUGER = 752;
		literal int M1_M2N4_AUGER = 753;
		literal int M1_M2N5_AUGER = 754;
		literal int M1_M2N6_AUGER = 755;
		literal int M1_M2N7_AUGER = 756;
		literal int M1_M2O1_AUGER = 757;
		literal int M1_M2O2_AUGER = 758;
		literal int M1_M2O3_AUGER = 759;
		literal int M1_M2O4_AUGER = 760;
		literal int M1_M2O5_AUGER = 761;
		literal int M1_M2O6_AUGER = 762;
		literal int M1_M2O7_AUGER = 763;
		literal int M1_M2P1_AUGER = 764;
		literal int M1_M2P2_AUGER = 765;
		literal int M1_M2P3_AUGER = 766;
		literal int M1_M2P4_AUGER = 767;
		literal int M1_M2P5_AUGER = 768;
		literal int M1_M2Q1_AUGER = 769;
		literal int M1_M2Q2_AUGER = 770;
		literal int M1_M2Q3_AUGER = 771;
		literal int M1_M3M2_AUGER = 772;
		literal int M1_M3M3_AUGER = 773;
		literal int M1_M3M4_AUGER = 774;
		literal int M1_M3M5_AUGER = 775;
		literal int M1_M3N1_AUGER = 776;
		literal int M1_M3N2_AUGER = 777;
		literal int M1_M3N3_AUGER = 778;
		literal int M1_M3N4_AUGER = 779;
		literal int M1_M3N5_AUGER = 780;
		literal int M1_M3N6_AUGER = 781;
		literal int M1_M3N7_AUGER = 782;
		literal int M1_M3O1_AUGER = 783;
		literal int M1_M3O2_AUGER = 784;
		literal int M1_M3O3_AUGER = 785;
		literal int M1_M3O4_AUGER = 786;
		literal int M1_M3O5_AUGER = 787;
		literal int M1_M3O6_AUGER = 788;
		literal int M1_M3O7_AUGER = 789;
		literal int M1_M3P1_AUGER = 790;
		literal int M1_M3P2_AUGER = 791;
		literal int M1_M3P3_AUGER = 792;
		literal int M1_M3P4_AUGER = 793;
		literal int M1_M3P5_AUGER = 794;
		literal int M1_M3Q1_AUGER = 795;
		literal int M1_M3Q2_AUGER = 796;
		literal int M1_M3Q3_AUGER = 797;
		literal int M1_M4M2_AUGER = 798;
		literal int M1_M4M3_AUGER = 799;
		literal int M1_M4M4_AUGER = 800;
		literal int M1_M4M5_AUGER = 801;
		literal int M1_M4N1_AUGER = 802;
		literal int M1_M4N2_AUGER = 803;
		literal int M1_M4N3_AUGER = 804;
		literal int M1_M4N4_AUGER = 805;
		literal int M1_M4N5_AUGER = 806;
		literal int M1_M4N6_AUGER = 807;
		literal int M1_M4N7_AUGER = 808;
		literal int M1_M4O1_AUGER = 809;
		literal int M1_M4O2_AUGER = 810;
		literal int M1_M4O3_AUGER = 811;
		literal int M1_M4O4_AUGER = 812;
		literal int M1_M4O5_AUGER = 813;
		literal int M1_M4O6_AUGER = 814;
		literal int M1_M4O7_AUGER = 815;
		literal int M1_M4P1_AUGER = 816;
		literal int M1_M4P2_AUGER = 817;
		literal int M1_M4P3_AUGER = 818;
		literal int M1_M4P4_AUGER = 819;
		literal int M1_M4P5_AUGER = 820;
		literal int M1_M4Q1_AUGER = 821;
		literal int M1_M4Q2_AUGER = 822;
		literal int M1_M4Q3_AUGER = 823;
		literal int M1_M5M2_AUGER = 824;
		literal int M1_M5M3_AUGER = 825;
		literal int M1_M5M4_AUGER = 826;
		literal int M1_M5M5_AUGER = 827;
		literal int M1_M5N1_AUGER = 828;
		literal int M1_M5N2_AUGER = 829;
		literal int M1_M5N3_AUGER = 830;
		literal int M1_M5N4_AUGER = 831;
		literal int M1_M5N5_AUGER = 832;
		literal int M1_M5N6_AUGER = 833;
		literal int M1_M5N7_AUGER = 834;
		literal int M1_M5O1_AUGER = 835;
		literal int M1_M5O2_AUGER = 836;
		literal int M1_M5O3_AUGER = 837;
		literal int M1_M5O4_AUGER = 838;
		literal int M1_M5O5_AUGER = 839;
		literal int M1_M5O6_AUGER = 840;
		literal int M1_M5O7_AUGER = 841;
		literal int M1_M5P1_AUGER = 842;
		literal int M1_M5P2_AUGER = 843;
		literal int M1_M5P3_AUGER = 844;
		literal int M1_M5P4_AUGER = 845;
		literal int M1_M5P5_AUGER = 846;
		literal int M1_M5Q1_AUGER = 847;
		literal int M1_M5Q2_AUGER = 848;
		literal int M1_M5Q3_AUGER = 849;
		literal int M2_M3M3_AUGER = 850;
		literal int M2_M3M4_AUGER = 851;
		literal int M2_M3M5_AUGER = 852;
		literal int M2_M3N1_AUGER = 853;
		literal int M2_M3N2_AUGER = 854;
		literal int M2_M3N3_AUGER = 855;
		literal int M2_M3N4_AUGER = 856;
		literal int M2_M3N5_AUGER = 857;
		literal int M2_M3N6_AUGER = 858;
		literal int M2_M3N7_AUGER = 859;
		literal int M2_M3O1_AUGER = 860;
		literal int M2_M3O2_AUGER = 861;
		literal int M2_M3O3_AUGER = 862;
		literal int M2_M3O4_AUGER = 863;
		literal int M2_M3O5_AUGER = 864;
		literal int M2_M3O6_AUGER = 865;
		literal int M2_M3O7_AUGER = 866;
		literal int M2_M3P1_AUGER = 867;
		literal int M2_M3P2_AUGER = 868;
		literal int M2_M3P3_AUGER = 869;
		literal int M2_M3P4_AUGER = 870;
		literal int M2_M3P5_AUGER = 871;
		literal int M2_M3Q1_AUGER = 872;
		literal int M2_M3Q2_AUGER = 873;
		literal int M2_M3Q3_AUGER = 874;
		literal int M2_M4M3_AUGER = 875;
		literal int M2_M4M4_AUGER = 876;
		literal int M2_M4M5_AUGER = 877;
		literal int M2_M4N1_AUGER = 878;
		literal int M2_M4N2_AUGER = 879;
		literal int M2_M4N3_AUGER = 880;
		literal int M2_M4N4_AUGER = 881;
		literal int M2_M4N5_AUGER = 882;
		literal int M2_M4N6_AUGER = 883;
		literal int M2_M4N7_AUGER = 884;
		literal int M2_M4O1_AUGER = 885;
		literal int M2_M4O2_AUGER = 886;
		literal int M2_M4O3_AUGER = 887;
		literal int M2_M4O4_AUGER = 888;
		literal int M2_M4O5_AUGER = 889;
		literal int M2_M4O6_AUGER = 890;
		literal int M2_M4O7_AUGER = 891;
		literal int M2_M4P1_AUGER = 892;
		literal int M2_M4P2_AUGER = 893;
		literal int M2_M4P3_AUGER = 894;
		literal int M2_M4P4_AUGER = 895;
		literal int M2_M4P5_AUGER = 896;
		literal int M2_M4Q1_AUGER = 897;
		literal int M2_M4Q2_AUGER = 898;
		literal int M2_M4Q3_AUGER = 899;
		literal int M2_M5M3_AUGER = 900;
		literal int M2_M5M4_AUGER = 901;
		literal int M2_M5M5_AUGER = 902;
		literal int M2_M5N1_AUGER = 903;
		literal int M2_M5N2_AUGER = 904;
		literal int M2_M5N3_AUGER = 905;
		literal int M2_M5N4_AUGER = 906;
		literal int M2_M5N5_AUGER = 907;
		literal int M2_M5N6_AUGER = 908;
		literal int M2_M5N7_AUGER = 909;
		literal int M2_M5O1_AUGER = 910;
		literal int M2_M5O2_AUGER = 911;
		literal int M2_M5O3_AUGER = 912;
		literal int M2_M5O4_AUGER = 913;
		literal int M2_M5O5_AUGER = 914;
		literal int M2_M5O6_AUGER = 915;
		literal int M2_M5O7_AUGER = 916;
		literal int M2_M5P1_AUGER = 917;
		literal int M2_M5P2_AUGER = 918;
		literal int M2_M5P3_AUGER = 919;
		literal int M2_M5P4_AUGER = 920;
		literal int M2_M5P5_AUGER = 921;
		literal int M2_M5Q1_AUGER = 922;
		literal int M2_M5Q2_AUGER = 923;
		literal int M2_M5Q3_AUGER = 924;
		literal int M3_M4M4_AUGER = 925;
		literal int M3_M4M5_AUGER = 926;
		literal int M3_M4N1_AUGER = 927;
		literal int M3_M4N2_AUGER = 928;
		literal int M3_M4N3_AUGER = 929;
		literal int M3_M4N4_AUGER = 930;
		literal int M3_M4N5_AUGER = 931;
		literal int M3_M4N6_AUGER = 932;
		literal int M3_M4N7_AUGER = 933;
		literal int M3_M4O1_AUGER = 934;
		literal int M3_M4O2_AUGER = 935;
		literal int M3_M4O3_AUGER = 936;
		literal int M3_M4O4_AUGER = 937;
		literal int M3_M4O5_AUGER = 938;
		literal int M3_M4O6_AUGER = 939;
		literal int M3_M4O7_AUGER = 940;
		literal int M3_M4P1_AUGER = 941;
		literal int M3_M4P2_AUGER = 942;
		literal int M3_M4P3_AUGER = 943;
		literal int M3_M4P4_AUGER = 944;
		literal int M3_M4P5_AUGER = 945;
		literal int M3_M4Q1_AUGER = 946;
		literal int M3_M4Q2_AUGER = 947;
		literal int M3_M4Q3_AUGER = 948;
		literal int M3_M5M4_AUGER = 949;
		literal int M3_M5M5_AUGER = 950;
		literal int M3_M5N1_AUGER = 951;
		literal int M3_M5N2_AUGER = 952;
		literal int M3_M5N3_AUGER = 953;
		literal int M3_M5N4_AUGER = 954;
		literal int M3_M5N5_AUGER = 955;
		literal int M3_M5N6_AUGER = 956;
		literal int M3_M5N7_AUGER = 957;
		literal int M3_M5O1_AUGER = 958;
		literal int M3_M5O2_AUGER = 959;
		literal int M3_M5O3_AUGER = 960;
		literal int M3_M5O4_AUGER = 961;
		literal int M3_M5O5_AUGER = 962;
		literal int M3_M5O6_AUGER = 963;
		literal int M3_M5O7_AUGER = 964;
		literal int M3_M5P1_AUGER = 965;
		literal int M3_M5P2_AUGER = 966;
		literal int M3_M5P3_AUGER = 967;
		literal int M3_M5P4_AUGER = 968;
		literal int M3_M5P5_AUGER = 969;
		literal int M3_M5Q1_AUGER = 970;
		literal int M3_M5Q2_AUGER = 971;
		literal int M3_M5Q3_AUGER = 972;
		literal int M4_M5M5_AUGER = 973;
		literal int M4_M5N1_AUGER = 974;
		literal int M4_M5N2_AUGER = 975;
		literal int M4_M5N3_AUGER = 976;
		literal int M4_M5N4_AUGER = 977;
		literal int M4_M5N5_AUGER = 978;
		literal int M4_M5N6_AUGER = 979;
		literal int M4_M5N7_AUGER = 980;
		literal int M4_M5O1_AUGER = 981;
		literal int M4_M5O2_AUGER = 982;
		literal int M4_M5O3_AUGER = 983;
		literal int M4_M5O4_AUGER = 984;
		literal int M4_M5O5_AUGER = 985;
		literal int M4_M5O6_AUGER = 986;
		literal int M4_M5O7_AUGER = 987;
		literal int M4_M5P1_AUGER = 988;
		literal int M4_M5P2_AUGER = 989;
		literal int M4_M5P3_AUGER = 990;
		literal int M4_M5P4_AUGER = 991;
		literal int M4_M5P5_AUGER = 992;
		literal int M4_M5Q1_AUGER = 993;
		literal int M4_M5Q2_AUGER = 994;
		literal int M4_M5Q3_AUGER = 995;

		// Radionuclides
		literal int RADIONUCLIDE_55FE = 0;
		literal int RADIONUCLIDE_57CO = 1;
		literal int RADIONUCLIDE_109CD = 2;
		literal int RADIONUCLIDE_125I = 3;
		literal int RADIONUCLIDE_137CS = 4;
		literal int RADIONUCLIDE_133BA = 5;
		literal int RADIONUCLIDE_153GD = 6;
		literal int RADIONUCLIDE_238PU = 7;
		literal int RADIONUCLIDE_241AM = 8;
		literal int RADIONUCLIDE_244CM = 9;
		#pragma endregion

		/// <summary>
		/// Initialize the library.
		/// </summary>
		static void XrayInit();

		/// <summary>
		/// Gets the atomic weight of the element with the specified atomic number.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <returns>Atomic weight</returns>
		static double AtomicWeight(int Z);

		/// <summary>
		/// Gets the density of a pure atomic element.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <returns>Density (g/cm3)</returns>
		static double ElementDensity(int Z);

		/// <summary>
		/// Gets element information for the specified atomic number.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <returns>Element information</returns>
		static Science::ElementData GetElementData(int Z);

		// Cross sections 
		/// <summary>
		/// Calculates the total cross section.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Total(int Z, double E);

		/// <summary>
		/// Calculates the photoelectric absorption cross section.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Photo(int Z, double E);

		/// <summary>
		/// Calculates the Rayleigh scattering cross section.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Rayl(int Z, double E);

		/// <summary>
		/// Calculates the Compton scattering cross section.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Compt(int Z, double E);

		/// <summary>
		/// Calculates the mass energy-absorption coefficient.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Energy(int Z, double E);

		/// <summary>
		/// Calculates the total cross section.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_Total(int Z, double E);

		/// <summary>
		/// Calculates the photoelectric absorption cross section.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_Photo(int Z, double E);

		/// <summary>
		/// Calculates the Rayleigh scattering cross section.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_Rayl(int Z, double E);

		/// <summary>
		/// Calculates the Compton scattering cross section.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_Compt(int Z, double E);

		/// <summary>
		/// Calculates the total Klein-Nishina cross section.
		/// </summary>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CS_KN(double E);

		// Unpolarized differential scattering cross sections
		/// <summary>
		/// Calculates the Thomson differential scattering cross section.
		/// </summary>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <returns>Cross section (barn)</returns>
		static double DCS_Thoms(double theta);

		/// <summary>
		/// Calculates the Klein-Nishina differential scattering cross section.
		/// </summary>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <returns>Cross section (barn)</returns>
		static double DCS_KN(double E, double theta);

		/// <summary>
		/// Calculates the Rayleigh differential scattering cross section.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <returns>Cross section (cm2/g/sterad)</returns>
		static double DCS_Rayl(int Z, double E, double theta);

		/// <summary>
		/// Calculates the Compton differential scattering cross section.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <returns>Cross section (cm2/g/sterad)</returns>
		static double DCS_Compt(int Z, double E, double theta);

		/// <summary>
		/// Calculates the Rayleigh differential scattering cross section.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <returns>Cross section (barn)</returns>
		static double DCSb_Rayl(int Z, double E, double theta);

		/// <summary>
		/// Calculates the Compton differential scattering cross section.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <returns>Cross section (barn)</returns>
		static double DCSb_Compt(int Z, double E, double theta);

		// Polarized differential scattering cross sections
		/// <summary>
		/// Calculates the Thomson differential scattering cross section for polarized beam.
		/// </summary>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <param name="phi">Scattering azimuthal angle (rad)</param>
		/// <returns>Cross section (barn)</returns>
		static double DCSP_Thoms(double theta, double phi);

		// Polarized differential scattering cross sections
		/// <summary>
		/// Calculates the Klein-Nishina differential scattering cross section for polarized beam.
		/// </summary>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <param name="phi">Scattering azimuthal angle (rad)</param>
		/// <returns>Cross section (barn)</returns>
		static double DCSP_KN(double E, double theta, double phi);

		/// <summary>
		/// Calculates the Rayleigh differential scattering cross section for polarized beam.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <param name="phi">Scattering azimuthal angle (rad)</param>
		/// <returns>Cross section (cm2/g/sterad)</returns>
		static double DCSP_Rayl(int Z, double E, double theta, double phi);

		/// <summary>
		/// Calculates the Compton differential scattering cross section for polarized beam.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <param name="phi">Scattering azimuthal angle (rad)</param>
		/// <returns>Cross section (cm2/g/sterad)</returns>
		static double DCSP_Compt(int Z, double E, double theta, double phi);

		/// <summary>
		/// Calculates the Rayleigh differential scattering cross section for polarized beam.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <param name="phi">Scattering azimuthal angle (rad)</param>
		/// <returns>Cross section (barn)</returns>
		static double DCSPb_Rayl(int Z, double E, double theta, double phi);

		/// <summary>
		/// Calculates the Compton differential scattering cross section for polarized beam.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <param name="phi">Scattering azimuthal angle (rad)</param>
		/// <returns>Cross section (barn)</returns>
		static double DCSPb_Compt(int Z, double E, double theta, double phi);

		// Scattering factors
		/// <summary>
		/// Calculates the Atomic form factor for Rayleigh scattering.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="q">Momentum transfer</param>
		/// <returns>Form factor</returns> 
		static double FF_Rayl(int Z, double q);

		/// <summary>
		/// Calculates the Incoherent scattering function for Compton scattering.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="q">Momentum transfer</param>
		/// <returns>Form factor</returns> 
		static double  SF_Compt(int Z, double q);

		/// <summary>
		/// Calculates the Momentum transfer for X-ray photon scattering.
		/// </summary>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <returns>Momentum transfer (1/A)</returns> 
		static double  MomentTransf(double E, double theta);

		/// <summary>
		/// Gets X-ray fluorescent line energy.
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Emission line ID</param>
		/// <returns>X-ray fluorescent line energy (keV)</returns> 
		static double LineEnergy(int Z, int line);

		/// <summary>
		/// Gets the fluorescence yield 
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <returns>Fluorescence yield</returns> 
		static double FluorYield(int Z, int shell);

		/// <summary>
		/// Gets the Coster-Kronig transition probability
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="trans">Atomic transition ID</param>
		/// <returns>Transition probability</returns> 
		static double CosKronTransProb(int Z, int trans);

		/// <summary>
		/// Gets the absorption-edge energy     
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <returns>Edge energy (keV)</returns> 
		static double EdgeEnergy(int Z, int shell);

		/// <summary>
		/// Gets the jump ratio     
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <returns>Jump ratio</returns> 
		static double JumpFactor(int Z, int shell);
		
		/// <summary>
		/// Calculates the fluorescent line cross section    
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns> 
		static double CS_FluorLine(int Z, int line, double E);

		/// <summary>
		/// Calculates the fluorescent line cross section    
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns> 
		static double CSb_FluorLine(int Z, int line, double E);
		
		/// <summary>
		/// Calculates the fluorescent cross section for an entire shell   
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns> 
		static double CS_FluorShell(int Z, int shell, double E);

		/// <summary>
		/// Calculates the fluorescent cross section for an entire shell   
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns> 
		static double CSb_FluorShell(int Z, int shell, double E);
		
		/// <summary>
		/// Gets the fractional radiative rate    
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <returns>Fractional radiative rate</returns> 
		static double RadRate(int Z, int line);

		/// <summary>
		/// Calculates the photon energy after Compton scattering   
		/// </summary>
		/// <param name="E0">Photon energy before scattering (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <returns>Energy after scattering (keV)</returns> 
		static double ComptonEnergy(double E0, double theta);

		// Anomalous scattering factors
		/// <summary>
		/// Calculates the real-part of the anomalous scattering factor 
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Real-part of anomalous scattering factor</returns> 
		static double Fi(int Z, double E);
		
		/// <summary>
		/// Calculates the imaginary-part of the anomalous scattering factor 
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Imaginary-part of anomalous scattering factor</returns> 
		static double Fii(int Z, double E);

		// Kissel Photoelectric cross sections
		/// <summary>
		/// Calculates the total photoelectric absorption cross section using Kissel partial photoelectric cross sections.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Photo_Total(int Z, double E);

		/// <summary>
		/// Calculates the total photoelectric absorption cross section using Kissel partial photoelectric cross sections.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_Photo_Total(int Z, double E);

		/// <summary>
		/// Calculates the partial photoelectric absorption cross section using Kissel partial photoelectric cross sections.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Photo_Partial(int Z, int shell, double E);

		/// <summary>
		/// Calculates the partial photoelectric absorption cross section using Kissel partial photoelectric cross sections.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_Photo_Partial(int Z, int shell, double E);

		// XRF cross sections using Kissel partial photoelectric cross sections
		// Total cross sections (photoionization + Rayleigh + Compton) using Kissel total photoelectric cross sections
		/// <summary>
		/// Calculates the total cross section using Kissel partial photoelectric cross sections.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Total_Kissel(int Z, double E);

		// Total cross sections (photoionization + Rayleigh + Compton) using Kissel total photoelectric cross sections
		/// <summary>
		/// Calculates the total cross section using Kissel partial photoelectric cross sections.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_Total_Kissel(int Z, double E);
		
		/// <summary>
		/// Calculates the fluorescent line cross section using Kissel partial photoelectric cross sections   
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns> 
		static double CS_FluorLine_Kissel(int Z, int line, double E);

		/// <summary>
		/// Calculates the fluorescent line cross section using Kissel partial photoelectric cross sections   
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns> 
		static double CSb_FluorLine_Kissel(int Z, int line, double E);
		
		/// <summary>
		/// Calculates the fluorescent line cross section including cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns> 
		static double CS_FluorLine_Kissel_Cascade(int Z, int line, double E);

		/// <summary>
		/// Calculates the fluorescent line cross section including cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns> 
		static double CSb_FluorLine_Kissel_Cascade(int Z, int line, double E);

		/// <summary>
		/// Calculates the fluorescent line cross section with non-radiative cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_FluorLine_Kissel_Nonradiative_Cascade(int Z, int line, double E);

		/// <summary>
		/// Calculates the fluorescent line cross section with non-radiative cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_FluorLine_Kissel_Nonradiative_Cascade(int Z, int line, double E);

		/// <summary>
		/// Calculates the fluorescent line cross section with radiative cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_FluorLine_Kissel_Radiative_Cascade(int Z, int line, double E);

		/// <summary>
		/// Calculates the fluorescent line cross section with non-radiative cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_FluorLine_Kissel_Radiative_Cascade(int Z, int line, double E);

		/// <summary>
		/// Calculates the fluorescent line cross section without cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_FluorLine_Kissel_No_Cascade(int Z, int line, double E);

		/// <summary>
		/// Calculates the fluorescent line cross section without cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Atomic line ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_FluorLine_Kissel_No_Cascade(int Z, int line, double E);
		
		/// <summary>
		/// Calculates the fluorescent shell cross section using Kissel partial photoelectric cross sections   
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns> 
		static double CS_FluorShell_Kissel(int Z, int shell, double E);

		/// <summary>
		/// Calculates the fluorescent shell cross section using Kissel partial photoelectric cross sections   
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns> 
		static double CSb_FluorShell_Kissel(int Z, int shell, double E);

		/// <summary>
		/// Calculates the fluorescent shell cross section including cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns> 
		static double CS_FluorShell_Kissel_Cascade(int Z, int shell, double E);

		/// <summary>
		/// Calculates the fluorescent shell cross section including cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns> 
		static double CSb_FluorShell_Kissel_Cascade(int Z, int shell, double E);

		/// <summary>
		/// Calculates the fluorescent shell cross section with non-radiative cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_FluorShell_Kissel_Nonradiative_Cascade(int Z, int shell, double E);

		/// <summary>
		/// Calculates the fluorescent shell cross section with non-radiative cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_FluorShell_Kissel_Nonradiative_Cascade(int Z, int shell, double E);

		/// <summary>
		/// Calculates the fluorescent shell cross section with radiative cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_FluorShell_Kissel_Radiative_Cascade(int Z, int shell, double E);

		/// <summary>
		/// Calculates the fluorescent shell cross section with non-radiative cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_FluorShell_Kissel_Radiative_Cascade(int Z, int shell, double E);

		/// <summary>
		/// Calculates the fluorescent shell cross section without cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_FluorShell_Kissel_No_Cascade(int Z, int shell, double E);

		/// <summary>
		/// Calculates the fluorescent shell cross section without cascade effects.  
		/// </summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_FluorShell_Kissel_No_Cascade(int Z, int shell, double E);
		
		//Cross Section functions using the compound parser
		/// <summary>
		/// Calculates the total cross section of a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Total_CP(String^ compound, double E);
		
		/// <summary>
		/// Calculates the photoelectric absorption cross section of a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Photo_CP(String^ compound, double E);

		/// <summary>
		/// Calculates the Rayleigh scattering cross section of a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Rayl_CP(String^ compound, double E);
		
		/// <summary>
		/// Calculates the Compton scattering cross section of a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Compt_CP(String^ compound, double E);
		
		/// <summary>
		/// Calculates the total cross section of a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_Total_CP(String^ compound, double E);

		/// <summary>
		/// Calculates the photoelectric absorption cross section of a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_Photo_CP(String^ compound, double E);

		/// <summary>
		/// Calculates the Rayleigh scattering cross section of a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_Rayl_CP(String^ compound, double E);
		
		/// <summary>
		/// Calculates the Compton scattering cross section of a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_Compt_CP(String^ compound, double E);
		
		/// <summary>
		/// Calculates the Rayleigh differential scattering cross section of a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <returns>Cross section (cm2/g/sterad)</returns>
		static double DCS_Rayl_CP(String^ compound, double E, double theta);
		
		/// <summary>
		/// Calculates the Compton differential scattering cross section of a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <returns>Cross section (cm2/g/sterad)</returns>		
		static double DCS_Compt_CP(String^ compound, double E, double theta);
		
		/// <summary>
		/// Calculates the Rayleigh differential scattering cross section of a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <returns>Cross section (barn/sterad)</returns>
		static double DCSb_Rayl_CP(String^ compound, double E, double theta);
		
		/// <summary>
		/// Calculates the Compton differential scattering cross section of a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <returns>Cross section (barn/sterad)</returns>
		static double DCSb_Compt_CP(String^ compound, double E, double theta);
		
		/// <summary>
		/// Calculates the Rayleigh differential scattering cross section for polarized beam for a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <param name="phi">Scattering azimuthal angle (rad)</param>
		/// <returns>Cross section (cm2/g/sterad)</returns>
		static double DCSP_Rayl_CP(String^ compound, double E, double theta, double phi);

		/// <summary>
		/// Calculates the Compton differential scattering cross section for polarized beam for a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <param name="phi">Scattering azimuthal angle (rad)</param>
		/// <returns>Cross section (cm2/g/sterad)</returns>
		static double DCSP_Compt_CP(String^ compound, double E, double theta, double phi);

		/// <summary>
		/// Calculates the Rayleigh differential scattering cross section for polarized beam for a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <param name="phi">Scattering azimuthal angle (rad)</param>
		/// <returns>Cross section (barn/sterad)</returns>
		static double DCSPb_Rayl_CP(String^ compound, double E, double theta, double phi);

		/// <summary>
		/// Calculates the Compton differential scattering cross section for polarized beam for a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="theta">Scattering polar angle (rad)</param>
		/// <param name="phi">Scattering azimuthal angle (rad)</param>
		/// <returns>Cross section (barn/sterad)</returns>
		static double DCSPb_Compt_CP(String^ compound, double E, double theta, double phi);
	
		/// <summary>
		/// Calculates the total photoelectric absorption cross section for a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>		
		static double CS_Photo_Total_CP(String^ compound, double E);

		/// <summary>
		/// Calculates the total photoelectric absorption cross section for a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>	
		static double CSb_Photo_Total_CP(String^ compound, double E);

		/// <summary>
		/// Calculates the total photoelectric absorption cross section for a compound
		/// using Kissel partial photoelectric cross sections.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (cm2/g)</returns>
		static double CS_Total_Kissel_CP(String^ compound, double E);

		/// <summary>
		/// Calculates the total photoelectric absorption cross section for a compound
		/// using Kissel partial photoelectric cross sections.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CSb_Total_Kissel_CP(String^ compound, double E);

		/// <summary>
		/// Calculates the mass energy-absorption coefficient for a compound.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Cross section (barn)</returns>
		static double CS_Energy_CP(String^ compound, double E);

		//Refractive indices functions
		/// <summary>
		/// Calculates the real part of the refractive index.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Real part of refractive index (electrons)</returns>
		static double Refractive_Index_Re(String^ compound, double E, double density);
		
		/// <summary>
		/// Calculates the imaginary part of the refractive index.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Imaginary part of refractive index (electrons)</returns>
		static double Refractive_Index_Im(String^ compound, double E, double density);

		/// <summary>
		/// Calculates the refractive index.
		/// </summary>
		/// <param name="compound">Chemical formula of the compound</param>
		/// <param name="E">Energy (keV)</param>
		/// <returns>Refractive index (electrons)</returns>
		static Numerics::Complex Refractive_Index(String^ compound, double E, double density);

		/// <summary>
		/// Calculates the electron configuration according to Kissel.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <returns>Electron configuration</returns> 
		static double ElectronConfig(int Z, int shell);

		//ComptonProfiles
		/// <summary>
		/// Calculates the total Compton scattering profile.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="pz">Momentum</param>
		/// <returns>Compton profile</returns> 
		static double ComptonProfile(int Z, double pz);

		/// <summary>
		/// Calculates the sub-shell Compton scattering profile.
		/// </summary>
		/// <param name="Z">Atomic number of the element</param>
		/// <param name="shell">Atomic shell ID</param>
		/// <param name="pz">Momentum</param>
		/// <returns>Compton profile</returns> 
		static double ComptonProfile_Partial(int Z, int shell, double pz);

		/// <summary>Calculates the atomic level width. </summary>
		/// <param name="Z">Atomic number of the element. </param>
		/// <param name="shell">Atomic shell ID. </param>
		/// <returns>Level width (keV)</returns>
		static double AtomicLevelWidth(int Z, int shell);

		/// <summary>Calculates the Auger non-radiative rate. </summary>
		/// <param name="Z">Atomic number of the element. </param>
		/// <param name="auger_trans">Value identifying initial ionized shell and two resulting ejected electrons</param>
		/// <returns>Non-radiative rate</returns>
		static double AugerRate(int Z, int auger_trans);

		/// <summary>Calculates the Auger non-radiative yeild. </summary>
		/// <param name="Z">Atomic number of the element. </param>
		/// <param name="shell">Atomic shell ID. </param>
		/// <returns>Non-radiative yeild</returns>
		static double AugerYield(int Z, int shell);

		/// <summary>
		/// Returns the Siegbahn line name corresponding to the specified IUPAC name.
		/// </summary>
		/// <param name="name">IUPAC line name</param>
		/// <returns>Siegbahn line name </returns> 
		static System::String ^IUPACToSiegbahnLineName(System::String ^name);

		/// <summary>
		/// Returns IUPAC line name corresponding to the specified Siegbahn name.
		/// </summary>
		/// <param name="name">Siegbahn line name</param>
		/// <returns>IUPAC line name </returns> 
		static System::String ^SiegbahnToIUPACLineName(System::String ^name);

		/// <summary>
		/// Returns the energy of the specified fluorescent line string.
		/// </summary>
		/// <param name="lineName">String containing the element and emission line, e.g. Cu Ka1</param>
		/// <returns>Fluorescent line energy (keV)</returns> 
		static double LineEnergyFromName(System::String ^lineName);

		/// <summary>
		/// Returns the atomic number and line ID of the specified fluorescent line string.
		/// </summary>
		/// <param name="elementLine">String containing the element and emission line, e.g. Cu Ka1</param>
		/// <param name="Z">Atomic number</param>
		/// <param name="line">Emission line ID</param>
		static void ElementAndLineFromName(System::String ^elementLine, int %Z, int %line);

		/// <summary>
		/// Returns the line ID of the specified emission line name.
		/// </summary>
		/// <param name="name">String containing the emission line name, e.g. Ka1</param>
		/// <returns>ID of the emission line</returns> 
		static int SiegbahnLineIndex(System::String ^name);

		/// <summary>
		/// Returns the atomic number from the specified element name.
		/// </summary>
		/// <param name="name">Element name, e.g. Cu</param>
		/// <returns>Atomic number</returns> 
		static int AtomicNumber(System::String ^name);

		/// <summary>
		/// Calculates the energy of the escape peak for a Si detector.
		/// </summary>
		/// <param name="energy">Energy of the incident X-ray peak (keV)</param>
		/// <returns>Energy of the escape peak (keV)</returns> 
		static double SiEscapeEnergy(double energy);

		/// <summary>
		/// Calculates the fraction of photons in the escape peak for a Si detector.
		/// </summary>
		/// <param name="energy">Energy of the incident X-ray peak (keV)</param>
		/// <returns>Fraction of incident photons in the escape peak</returns> 
		static double SiEscapeFraction(double energy); 
	};

}
