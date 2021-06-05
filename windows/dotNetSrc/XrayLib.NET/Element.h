/*
	XrayLib.NET copyright (c) 2010-2019 Matthew Wormington. All rights reserved.
	
	File: Element.h
	Author: Matthew Wormington
	Language: C++/CLI   
	Compiler: Microsoft Visual Studio 2017
	Created: September 4, 2010
	$Version:$
	$Revision:$
	$RevDate:$

	Description:
	Contains a managed class containing element data for use with XrayLib.NET.

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

using namespace System;
using namespace System::Collections::Generic;
using namespace System::Text;

/// <summary>
/// A namespace that contains scientific classes. 
/// </summary>
namespace Science {

	/// <summary>A structure that contains data for an element in the Periodic Table. </summary>
	public value struct ElementData
	{		
		/// <summary>Atomic number</summary>
		int Number;  
		/// <summary>Name </summary>
		String^ Name;    
		/// <summary>Atomic symbol </summary>
		String^ Symbol;
		/// <summary>Atomic weight </summary>
		double Weight;  
		/// <summary>Group in Periodic Table</summary>
		int Group;   
		/// <summary>Period in Periodic Table</summary>
		int Period;      
		/// <summary>Density (g/cm^3)</summary>
		double Density;  
	public:
		virtual String^ ToString() override
		{
			return String::Format("{0}, {1}, {2}, {3}, {4}, {5}, {6}", 
				Number, Name, Symbol, Weight, Group, Period, Density);
		}
	};

	private ref class Elements
	{
	private:
		static array<ElementData> ^_elementData = {
			{0, "", "", 0.0, 0, 0, 0.0},
			{1, "Hydrogen", "H", 1.00794, 1, 1, 0.00008988},
			{2, "Helium", "He", 4.002602, 1, 18, 0.0001785},
			{3, "Lithium", "Li", 6.941, 2, 1, 0.534},
			{4, "Beryllium", "Be", 9.012182, 2, 2, 1.8477},
			{5, "Boron", "B", 10.811, 2, 13, 2.34},
			{6, "Carbon", "C", 12.0107, 2, 14, 2.26},
			{7, "Nitrogen", "N", 14.0067, 2, 15, 0.0012506},
			{8, "Oxygen", "O", 15.9994, 2, 16, 0.001429},
			{9, "Fluorine", "F", 18.9984032, 2, 17, 0.001696},
			{10, "Neon", "Ne", 20.1797, 2, 18, 0.00089994},
			{11, "Sodium", "Na", 22.98977, 3, 1, 0.971},
			{12, "Magnesium", "Mg", 24.305, 3, 2, 1.738},
			{13, "Aluminum", "Al", 26.981538, 3, 13, 2.698},
			{14, "Silicon", "Si", 28.0855, 3, 14, 2.329},
			{15, "Phosphorus", "P", 30.973761, 3, 15, 1.82},
			{16, "Sulfur", "S", 32.065, 3, 16, 1.957},
			{17, "Chlorine", "Cl", 35.453, 3, 17, 0.003214},
			{18, "Argon", "Ar", 39.948, 3, 18, 0.001784},
			{19, "Potassium", "K", 39.0983, 4, 1, 0.862},
			{20, "Calcium", "Ca", 40.078, 4, 2, 1.55},
			{21, "Scandium", "Sc", 44.95591, 4, 3, 2.989},
			{22, "Titanium", "Ti", 47.867, 4, 4, 4.54},
			{23, "Vanadium", "V", 50.9415, 4, 5, 6.11},
			{24, "Chromium", "Cr", 51.9961, 4, 6, 7.19},
			{25, "Manganese", "Mn", 54.938049, 4, 7, 7.44},
			{26, "Iron", "Fe", 55.845, 4, 8, 7.874},
			{27, "Cobalt", "Co", 58.9332, 4, 9, 8.9},
			{28, "Nickel", "Ni", 58.6934, 4, 10, 8.902},
			{29, "Copper", "Cu", 63.546, 4, 11, 8.96},
			{30, "Zinc", "Zn", 65.39, 4, 12, 7.133},
			{31, "Gallium", "Ga", 69.723, 4, 13, 5.907},
			{32, "Germanium", "Ge", 72.64, 4, 14, 5.323},
			{33, "Arsenic", "As", 74.9216, 4, 15, 5.78},
			{34, "Selenium", "Se", 78.96, 4, 16, 4.79},
			{35, "Bromine", "Br", 79.904, 4, 17, 3.1226},
			{36, "Krypton", "Kr", 83.8, 4, 18, 0.0037493},
			{37, "Rubidium", "Rb", 85.4678, 5, 1, 1.532},
			{38, "Strontium", "Sr", 87.62, 5, 2, 2.54},
			{39, "Yttrium", "Y", 88.90585, 5, 3, 4.469},
			{40, "Zirconium", "Zr", 91.224, 5, 4, 6.506},
			{41, "Niobium", "Nb", 92.90638, 5, 5, 8.57},
			{42, "Molybdenum", "Mo", 95.94, 5, 6, 10.22},
			{43, "Technetium", "Tc", 98, 5, 7, 11.5},
			{44, "Ruuthenium", "Ru", 101.07, 5, 8, 12.37},
			{45, "Rhodium", "Rh", 102.9055, 5, 9, 12.41},
			{46, "Palladium", "Pd", 106.42, 5, 10, 12.02},
			{47, "Silver", "Ag", 107.8682, 5, 11, 10.5},
			{48, "Cadmium", "Cd", 112.411, 5, 12, 8.65},
			{49, "Indium", "In", 114.818, 5, 13, 7.31},
			{50, "Tin", "Sn", 118.71, 5, 14, 7.31},
			{51, "Antimony", "Sb", 121.76, 5, 15, 6.691},
			{52, "Tellurium", "Te", 127.6, 5, 16, 6.24},
			{53, "Iodine", "I", 126.90447, 5, 17, 4.93},
			{54, "Xenon", "Xe", 131.293, 5, 18, 0.0058971},
			{55, "Cesium", "Cs", 132.90545, 6, 1, 1.873},
			{56, "Barium", "Ba", 137.327, 6, 2, 3.594},
			{57, "Lanthanum", "La", 138.9055, 6, 3, 6.145},
			{58, "Cerium", "Ce", 140.116, 6, 3, 6.689},
			{59, "Praseodymium", "Pr", 140.90765, 6, 3, 6.773},
			{60, "Neodumium", "Nd", 144.24, 6, 3, 7.007},
			{61, "Promethium", "Pm", 145, 6, 3, 7.22},
			{62, "Samarium", "Sm", 150.36, 6, 3, 7.52},
			{63, "Europium", "Eu", 151.964, 6, 3, 5.243},
			{64, "Gadolinium", "Gd", 157.25, 6, 3, 7.9004},
			{65, "Terbium", "Tb", 158.92534, 6, 3, 8.229},
			{66, "Dysprosium", "Dy", 162.5, 6, 3, 8.55},
			{67, "Holmium", "Ho", 164.93032, 6, 3, 8.795},
			{68, "Erbium", "Er", 167.259, 6, 3, 9.066},
			{69, "Thulium", "Tm", 168.93421, 6, 3, 9.321},
			{70, "Ytterbium", "Yb", 173.04, 6, 3, 6.965},
			{71, "Lutetium", "Lu", 174.967, 6, 3, 9.84},
			{72, "Hafnium", "Hf", 178.49, 6, 4, 13.31},
			{73, "Tantalum", "Ta", 180.9479, 6, 5, 16.654},
			{74, "Tungsten", "W", 183.84, 6, 6, 19.3},
			{75, "Rhenium", "Re", 186.207, 6, 7, 21.02},
			{76, "Osmium", "Os", 190.23, 6, 8, 22.59},
			{77, "Iridium", "Ir", 192.217, 6, 9, 22.42},
			{78, "Platinum", "Pt", 195.078, 6, 10, 21.45},
			{79, "Gold", "Au", 196.96655, 6, 11, 19.32},
			{80, "Mercury", "Hg", 200.59, 6, 12, 13.546},
			{81, "Thallium", "Tl", 204.3833, 6, 13, 11.85},
			{82, "Lead", "Pb", 207.2, 6, 14, 11.35},
			{83, "Bismuth", "Bi", 208.98038, 6, 15, 9.747},
			{84, "Polonium", "Po", 209, 6, 16, 9.32},
			{85, "Astatine", "At", 210, 6, 17, -9999},
			{86, "Radon", "Rn", 222, 6, 18, 0.00973},
			{87, "Francium", "Fr", 223, 7, 1, -9999},
			{88, "Radium", "Ra", 226, 7, 2, 5},
			{89, "Actinium", "Ac", 227, 7, 3, 10.06},
			{90, "Thorium", "Th", 232.0381, 7, 3, 11.72},
			{91, "Protactnium", "Pa", 231.03588, 7, 3, 15.37},
			{92, "Uranium", "U", 238.02891, 7, 3, 18.95},
			{93, "Neptunium", "Np", 237, 7, 3, 20.25},
			{94, "Plutonium", "Pu", 244, 7, 3, 19.84},
			{95, "Americium", "Am", 243, 7, 3, 13.67},
			{96, "Curium", "Cm", 247, 7, 3, 13.3},
			{97, "Berkelium", "Bk", 247, 7, 3, 14.79},
			{98, "Californium", "Cf", 251, 7, 3, -9999},
			{99, "Einsteinium", "Es", 252, 7, 3, -9999},
			{100, "Fermium", "Fm", 257, 7, 3, -9999},
			{101, "Mendelevium", "Md", 258, 7, 3, -9999},
			{102, "Nobelium", "No", 259, 7, 3, -9999},
			{103, "Lawrencium", "Lr", 262, 7, 3, -9999},
			{104, "Rutherfordium", "Rf", 261, 7, 4, -9999},
			{105, "Dubnium", "Db", 262, 7, 5, -9999},
			{106, "Seaborgium", "Sg", 266, 7, 6, -9999},
			{107, "Bohrium", "Bh", 264, 7, 7, -9999},
			{108, "Hassium", "Hs", 277, 7, 8, -9999},
			{109, "Meitnerium", "Mt", 268, 7, 9, -9999},
			{110, "Darmstadtium", "Ds", 271, 7, 10, -9999},
			{111, "Roentgenium", "Rg", 272, 7, 11, -9999}}; 

	public:
		/// <summary>	Gets the element data for the specified element. </summary>
		/// <param name="Z">Atomic number. </param>
		/// <returns>Element data. </returns>
		static ElementData GetData(int Z)
		{
			if (Z < 1 || Z > 110) 
			{
				return _elementData[0];
			}

			return _elementData[Z];
		}
	};
}
