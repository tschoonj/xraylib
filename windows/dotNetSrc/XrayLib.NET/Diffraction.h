/*
	XrayLib.NET copyright (c) 2010-2019 Matthew Wormington. All rights reserved.

	File: Diffraction.h
	Author: Matthew Wormington
	Language: C++/CLI   
	Compiler: Microsoft Visual Studio 2017
	Created: July 17, 2012
	$Version:$
	$Revision:$
	$RevDate:$

	Description:
	Contains a managed class for basic X-ray diffraction (XRD) calculations 
	from crystals for use with XrayLib.NET.

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

extern "C" {
	#include "..\XrayLib\xraylib-crystal-diffraction.h"
}
#include <string.h>

using namespace System;
using namespace System::Collections::Generic;
using namespace System::Numerics;
using namespace System::Runtime::InteropServices;
using namespace System::Text;

/// <summary>
/// A namespace that contains scientific classes.
/// </summary>
namespace Science {

	/// <summary>A simple class for an atom in a crystal </summary>
	public value class CrystalAtom 
	{
	public:
		/// <summary>Atomic number of atom</summary>
		int Zatom;                  
		/// <summary>Fractional contribution. Normally 1.0</summary>
		double fraction;            
		/// <summary>Atom position (fraction of the unit cell length)</summary>
		double x;
		/// <summary>Atom position (fraction of the unit cell length)</summary>
		double y;
		/// <summary>Atom position (fraction of the unit cell length)</summary>
		double z;             
	};

	/// <summary>
	/// A managed wrapper for XrayLib's X-ray diffraction (XRD) functionality.
	/// </summary>
	public ref class Crystal
	{		
	private:	
		List<CrystalAtom>^ _atoms;   // List of atoms in unit cell.

		double CosD(Double x)
		{
			return Math::Cos(x * PI / 180.0);
		}

		double SinD(Double x)
		{
			return Math::Sin(x * PI / 180.0);
		}
		
		::Crystal_Struct* ToCrystal_Struct()
		{
			::Crystal_Struct* result = new ::Crystal_Struct;
			
			result->name = new char[Name->Length];
			
			IntPtr p = Marshal::StringToHGlobalAnsi(Name);
			try
			{					
				char* str = static_cast<char*>(p.ToPointer());
			
				::memcpy(result->name, str, Name->Length);
			}
			finally
			{
				Marshal::FreeHGlobal(p);
			}
			
			result->a = A;
			result->b = B;
			result->c = C;
			result->alpha = Alpha;
			result->beta = Beta;
			result->gamma = Gamma;
			result->volume = Volume;

			int atomCount = _atoms->Count;
			result->n_atom = atomCount;

			result->atom = new ::Crystal_Atom[atomCount];
			for (int i = 0; i < atomCount; i++)
			{
				::Crystal_Atom* ca = &result->atom[i];
				ca->Zatom = _atoms[i].Zatom;
				ca->fraction = _atoms[i].fraction;
				ca->x = _atoms[i].x;
				ca->y = _atoms[i].y;
				ca->z = _atoms[i].z;
			}

			return result;
		}

		void DeleteCrystal_Struct(::Crystal_Struct* cs)
		{
			if (cs != nullptr)
			{
				delete cs->name;
				delete cs->atom;
				delete cs;
				cs = nullptr;
			}
		}
		
	public:	

		/// <summary>Name of crystal</summary>
		property String^ Name;    
		/// <summary>Unit cell dimension (Angstrom)</summary>
		property double A;           
		/// <summary>Unit cell dimension (Angstrom)</summary>
		property double B;
		/// <summary>Unit cell dimension (Angstrom)</summary>
		property double C;       
		/// <summary>Unit cell angle (deg)</summary>
		property double Alpha;    
		/// <summary>Unit cell angle (deg)</summary>
		property double Beta;
		/// <summary>Unit cell angle (deg)</summary>
		property double Gamma; 
		/// <summary>Unit cell volume (Angstrom^3)</summary>
		property double Volume;     
		/// <summary>List of atoms in the unit cell</summary>
		property List<CrystalAtom>^ Atoms
		{
			List<CrystalAtom>^ get()
			{
				return _atoms;
			}
		}

		/// <summary> Default constructor. </summary>
		Crystal()
		{
			_atoms = gcnew List<CrystalAtom>();		
			Clear();
		}

		/// <summary> Clears this object to its blank/initial state. </summary>
		void Clear()
		{
			_atoms->Clear();
			
			Name = "";
			A = 0.0;
			B = 0.0;
			C = 0.0;
			Alpha = 0.0;
			Beta = 0.0;
			Gamma = 0.0;
			Volume = 0.0;
		}

		/// <summary>Calculates the Bragg angle for the specified reflection</summary>
		/// <param name="E">Energy (keV)</param>
		/// <param name="h">Miller index, h </param>
		/// <param name="k">Miller index, k</param>
		/// <param name="l">Miller index, l</param>
		/// <returns>Bragg angle (rad)</returns>
		double BraggAngle(double E, int h, int k, int l)
		{
			double result = 0.0;
			
			::Crystal_Struct* cs = ToCrystal_Struct();
			try
			{
				::xrl_error *error = nullptr;
				result = ::Bragg_angle(cs, E, h, k, l, &error);
				Errors::HandleError(error);
			}
			finally
			{
				DeleteCrystal_Struct(cs);
			}

			return result;
		}

		/// <summary>Calculates the components of the atomic scattering factor</summary>
		/// <param name="Z">Atomic number</param>
		/// <param name="E">Energy (keV)</param>
		/// <param name="q">Momentum transfer (1/Angstrom)</param>
		/// <param name="debyeFactor">Debye temperature facotor</param>
		/// <param name="f0">f0</param>
		/// <param name="fp">f'</param>
		/// <param name="fpp">f''</param>
		void AtomicScatteringFactors(int Z, double E, double q, double debyeFactor, double %f0, double %fp, double %fpp)
		{
			double f0_ = f0;
			double fp_ = fp;
			double fpp_ = fpp;

			::xrl_error *error = nullptr;
			::Atomic_Factors(Z, E, q, debyeFactor, &f0_, &fp_, &fpp_, &error);  
			Errors::HandleError(error);

			f0 = f0_;
			fp = fp_;
			fpp = fpp_;
		}

		/// <summary>Calculates the structure factor of the crystal for the specified reflection</summary>
		/// <remarks>See also PartialStructureFactor</remarks>
		/// <param name="E">Energy (keV)</param>
		/// <param name="h">Miller index, h </param>
		/// <param name="k">Miller index, k</param>
		/// <param name="l">Miller index, l</param>
		/// <param name="debyeFactor">Debye temperature factor</param>
		/// <param name="relativeAngle">Relative angle, i.e. fraction of Bragg angle</param>
		/// <returns>Complex structure factor</returns>
		Numerics::Complex StructureFactor(double E, int h, int k, int l, double debyeFactor, double relativeAngle)
		{
			Numerics::Complex result = 1.0;
			
			::Crystal_Struct* cs = ToCrystal_Struct();
			try
			{
				//::xrlComplex z = ::Crystal_F_H_StructureFactor(cs, E, h, k, l, debyeFactor, relativeAngle);
				
				::xrlComplex z;
				::xrl_error *error = nullptr;
				::Crystal_F_H_StructureFactor2(cs, E, h, k, l, debyeFactor, relativeAngle, &z, &error);  
				Errors::HandleError(error);
				result = Numerics::Complex(z.re, z.im);
			}
			finally
			{
				DeleteCrystal_Struct(cs);
			}  

			return result;		
		}

		/// <summary>Calculates the partial structure factor of the crystal for the specified reflection</summary>
		/// <remarks>
		/// See also StructureFactor
		/// The Atomic structure factor has three terms: F_H = f0 + f' + i*f''
		/// For each of these three terms, there is a corresponding *_flag argument which controls the numerical value used in computing F_H
		///      ...Flag = 0 --> Set this term to 0.
		///      ...Flag = 1 --> Set this term to 1. Only used for f0.
		///      ...Flag = 2 --> Set this term to the value given 
		/// </remarks>
		/// <param name="E">Energy (keV)</param>
		/// <param name="h">Miller index, h </param>
		/// <param name="k">Miller index, k</param>
		/// <param name="l">Miller index, l</param>
		/// <param name="debyeFactor">Debye temperature factor</param>
		/// <param name="relativeAngle">Relative angle, i.e. fraction of Bragg angle</param>		
		/// <param name="f0Flag">f0 flag. </param>
		/// <param name="fpFlag">f' flag. </param>
		/// <param name="fppFlag">f'' flag. </param>
		/// <returns>Complex structure factor</returns>
		Numerics::Complex PartialStructureFactor(double E, int h, int k, int l, double debyeFactor, double relativeAngle, int f0Flag, int fpFlag, int fppFlag)
		{
			Numerics::Complex result = -1.0;

			::Crystal_Struct* cs = ToCrystal_Struct();
			try
			{
				//::xrlComplex z = ::Crystal_F_H_StructureFactor_Partial(cs, E, h, k, l, debyeFactor, relativeAngle, f0Flag, fpFlag, fppFlag);
				
				::xrlComplex z;
				::xrl_error *error = nullptr;
				::Crystal_F_H_StructureFactor_Partial2(cs, E, h, k, l, debyeFactor, relativeAngle, f0Flag, fpFlag, fppFlag, &z, &error);
				Errors::HandleError(error);
				result = Numerics::Complex(z.re, z.im);
			}
			finally
			{
				DeleteCrystal_Struct(cs);
			}

			return result;
		}

		/// <summary>Calculates the unit cell volume of the crystal (using XrayLib)</summary>
		/// <returns>Unit cell volume (Angstrom^3)</returns>
		double UnitCellVolume()
		{
			double result = -1.0;

			::Crystal_Struct* cs = ToCrystal_Struct();
			try
			{
				::xrl_error *error = nullptr;
				result = ::Crystal_UnitCellVolume(cs, &error);
				Errors::HandleError(error);
			}
			finally
			{
				DeleteCrystal_Struct(cs);
			}		

			return result;
		}
	
		/// <summary>Calculates the inter-planar spacing (d-spacing) specified crystal planes</summary>
		/// <remarks>
		/// This routine assumes that if crystal.volume is nonzero then it holds a valid value.
		/// If (i, j, k) = (0, 0, 0) then zero is returned.
		/// </remarks>
		/// <param name="h">Miller index, h </param>
		/// <param name="k">Miller index, k</param>
		/// <param name="l">Miller index, l</param>
		/// <returns>Inter-planar distance (Angstrom)</returns>
		double DSpacing(int h, int k, int l)
		{
			double result = -1.0;

			::Crystal_Struct* cs = ToCrystal_Struct();
			try
			{
				::xrl_error *error = nullptr;
				result = ::Crystal_dSpacing(cs, h, k, l, &error);
				Errors::HandleError(error);
			}
			finally
			{
				DeleteCrystal_Struct(cs);
			}		

			return result;		
		}

		/// <summary>Calculates the unit cell volume of the crystal</summary>
		/// <remarks>
		/// Uses the triclinic formulas taken from Crystals And Crystal Structures, R.J.D. Tilley (Wiley, 2006) pp36-37
		/// </remarks>
		/// <returns>Unit cell volume (Angstrom^3)</returns>
		double UnitCellVolume2()
		{
			double V = 0.0;

			double cosAlpha = CosD(Alpha);
			double cosBeta = CosD(Beta);
			double cosGamma = CosD(Gamma);
			
			V = A*B*C*Math::Sqrt(1-cosAlpha*cosAlpha-cosBeta*cosBeta-cosGamma*cosGamma+2*cosAlpha*cosBeta*cosGamma);

			return V;
		}

		/// <summary>Calculates the inter-planar spacing (d-spacing) specified crystal planes</summary>
		/// <remarks>
		/// If (i, j, k) = (0, 0, 0) then zero is returned.
		/// Uses the triclinic formulas taken from Crystals And Crystal Structures, R.J.D. Tilley (Wiley, 2006) pp36-37
		/// </remarks>
		/// <param name="h">Miller index, h </param>
		/// <param name="k">Miller index, k</param>
		/// <param name="l">Miller index, l</param>
		/// <returns>Inter-planar distance (Angstrom)</returns>
		double InterplanarSpacing(int h, int k, int l)
		{
			double d = 0.0;

			if (A >= 0 && B > 0 && C > 0)
			{
				double cosAlpha = CosD(Alpha);
				double cosBeta = CosD(Beta);
				double cosGamma = CosD(Gamma);
				double sinAlpha = SinD(Alpha);
				double sinBeta = SinD(Beta);
				double sinGamma = SinD(Gamma);
				
				double S11 = B*B*C*C*sinAlpha*sinAlpha;
				double S22 = A*A*C*C*sinBeta*sinBeta;
				double S33 = A*A*B*B*sinGamma*sinGamma;
				double S12 = A*B*C*C*(cosAlpha*cosBeta-cosGamma);
				double S23 = A*A*B*C*(cosBeta*cosGamma-cosAlpha);
				double S13 = A*B*B*C*(cosGamma*cosAlpha-cosBeta);

				double V = UnitCellVolume2();

				d = Math::Sqrt((V*V)/(S11*h*h + S22*k*k + S33*l*l + 2*S12*k*l + 2*S23*k*l + 2*S13*h*l));
			}
			
			return d;
		}

		/// <summary>Calculates the inter-planar angle between the two specified crystal planes (h1,k1,l1) and (h2,k2,l2)</summary>
		/// <remarks>
		/// If (i, j, k) = (0, 0, 0) then zero is returned.
		/// Uses the triclinic formulas taken from Crystals And Crystal Structures, R.J.D. Tilley (Wiley, 2006) pp36-37
		/// </remarks>
		/// <param name="h1">Miller index, h1</param>
		/// <param name="k1">Miller index, k1</param>
		/// <param name="l1">Miller index, l1</param>
		/// <param name="h2">Miller index, h2</param>
		/// <param name="k2">Miller index, k2</param>
		/// <param name="l2">Miller index, l2</param> 
		/// <returns>Inter-planar distance (Angstrom)</returns>
		double InterplanarAngle(int h1, int k1, int l1, int h2, int k2, int l2)
		{
			double phi = 0.0;

			if (A >= 0 && B > 0 && C > 0)
			{
				double d1 = InterplanarSpacing(h1, k1, l1);
				double d2 = InterplanarSpacing(h2, k2, l2);
				
				double cosAlpha = CosD(Alpha);
				double cosBeta = CosD(Beta);
				double cosGamma = CosD(Gamma);
				double sinAlpha = SinD(Alpha);
				double sinBeta = SinD(Beta);
				double sinGamma = SinD(Gamma);

				double S11 = B*B*C*C*sinAlpha*sinAlpha;
				double S22 = A*A*C*C*sinBeta*sinBeta;
				double S33 = A*A*B*B*sinGamma*sinGamma;
				double S12 = A*B*C*C*(cosAlpha*cosBeta-cosGamma);
				double S23 = A*A*B*C*(cosBeta*cosGamma-cosAlpha);
				double S13 = A*B*B*C*(cosGamma*cosAlpha-cosBeta);

				double V = UnitCellVolume2();

				phi = Math::Acos(d1*d2/(V*V)*(S11*h1*h2 + S22*k1*k2 + S33*l1*l2 + 
					S23*(k1*l2+k2*l1) + S13*(l1*h2+l2*h1) + S12*(h1*k2+h2*k1)));
			}

			return phi;
		}

		/// <summary>Calculates the magnitude of the scattering vector Q relative to the Bragg angle of the specified reflection</summary>
		/// <remarks>Q = Sin(ThetaB*relativeAngle) / Wavelength</remarks>
		/// <param name="E">Energy (keV)</param>
		/// <param name="h">Miller index, h </param>
		/// <param name="k">Miller index, k</param>
		/// <param name="l">Miller index, l</param>
		/// <param name="relativeAngle">Relative angle, i.e. fraction of Bragg angle</param>
		/// <returns>The magnitude of the scattering vector (1/Angstrom)</returns>
		double ScatteringVectorMagnitide(double E, int h, int k, int l, double relativeAngle)
		{
			double result = -1.0;

			::Crystal_Struct* cs = ToCrystal_Struct();
			try
			{
				::xrl_error *error = nullptr;
				result = ::Q_scattering_amplitude(cs, E, h, k, l, relativeAngle, &error);
				Errors::HandleError(error);
			}
			finally
			{
				DeleteCrystal_Struct(cs);
			}		

			return result;		
		}

		/// <summary>Convert this object into a string representation. </summary>
		/// <returns>Null if it fails, else a string representation of this object. </returns>
		virtual String^ ToString() override 
		{
			StringBuilder^ sb = gcnew StringBuilder();
			sb->AppendLine(String::Format("Name: {0}", Name));
			sb->AppendLine("Unit cell lengths (Å):");
			sb->AppendLine(String::Format("  a = {0:F6}\n  b = {1:F6}\n  c = {2:F6}", A, B, C));
			sb->AppendLine("Unit cell angles (deg):");
			sb->AppendLine(String::Format("  alpha = {0:F6}\n  beta = {1:F6}\n  gamma = {2:F6}", Alpha, Beta, Gamma));
			sb->AppendLine("Atoms:");
			sb->AppendLine(String::Format("  Count: {0}", Atoms->Count));
			sb->AppendLine("  Z  Fraction X        Y        Z");
			for (int i = 0; i < Atoms->Count; i++)
			{
				CrystalAtom atom = Atoms[i];
				sb->AppendLine(String::Format("  {0} {1:F6} {2:F6} {3:F6} {4:F6}", atom.Zatom, atom.fraction, atom.x, atom.y, atom.z));
			}
			return sb->ToString();
		}
	};

	/// <summary>
	/// A class containing an array of Crystals.
	/// </summary>
	public ref class CrystalArray
	{		
	private:
		::Crystal_Array* ca;
	public:
		CrystalArray()
		{
			ca = nullptr;
		}
		
		~CrystalArray()
		{
			if (ca != nullptr)
				::Crystal_ArrayFree(ca);
		}

		/// <summary>Loads from the crystals from an appropriately formatted text file.</summary>
		/// <remarks>
		/// #UCOMMENT  comment 
		/// #UCELL a b c alpha beta gamma  
		///		The unit cell dimensions (A and deg) (*MANDATORY, IT MUST EXIST*)
		/// #UTEMP temperature in Kelvin at which UCELL is given
		/// #UREF reference
		/// #USYSTEM : 7 crystal system, i.e., triclinic monoclinic orthorhombic 
		/// 			tetragonal rhombohedral(trigonal) 
		/// 			hexagonal cubic
		/// #ULATTICE the lattice centering: 
		/// P: Primitive centering: lattice points on the cell corners only
		/// I: Body centered: one additional lattice point at the center of the cell
		/// F: Face centered: one additional lattice point at center of each 
		///    of the faces of the cell 
		///    A,B,C Centered on a single face (A, B or C centering): one
		///    additional lattice point at the center of one of the 
		///    cell faces.
		///   The 14 Bravais lattices are, then:
		///		1 triclinic (P)
		/// 			2 monoclinic (P,C)
		/// 			3 orthorhombic (P,C,I,F)
		/// 			2 tetragonal (P,I)
		/// 			1 rhombohedral (P)
		/// 			1 hexagonal (P)
		/// 			3 cubic (P,I,F)
		/// #USTRUCTURE Model for structure (e.g., diamond, fcc)
		///			Data columns: 
		///  		4 or 5: 
		///  		AtomicNumber  Fraction  X  Y  Z Biso
		/// 			The Biso one is optional		
		/// </remarks>
		/// <param name="fileName">	[in,out] If non-null, filename of the file. </param>
		/// <returns>A value that indicates whether the file was loaded</returns>
		int LoadFromFile(String^ fileName, int capacity)
		{
			int result = -1;
			
			if (ca != nullptr)
				::Crystal_ArrayFree(ca);
			
			::xrl_error *error = nullptr;
			ca = ::Crystal_ArrayInit(capacity, &error);
			Errors::HandleError(error);
			
			IntPtr p = Marshal::StringToHGlobalAnsi(fileName);
			try
			{					
				char* pFileName = static_cast<char*>(p.ToPointer());
				::xrl_error *error = nullptr;
				result = ::Crystal_ReadFile(pFileName, ca, &error);
				Errors::HandleError(error);
			}
			finally
			{
				Marshal::FreeHGlobal(p);
			}	

			return result;
		}

		/// <summary>Gets a Crystal object from the array using the specified name</summary>
		/// <param name="name">	[in,out] If non-null, the crystal name. </param>
		/// <returns>Null if it fails, else the crystal. </returns>
		Crystal^ GetCrystal(String^ name)
		{
			Crystal^ result = gcnew Crystal();
			Crystal_Struct* cs;

			IntPtr p = Marshal::StringToHGlobalAnsi(name);
			try
			{					
				char* name = static_cast<char*>(p.ToPointer());
				::xrl_error *error = nullptr;
				cs = ::Crystal_GetCrystal(name, ca, &error);
				Errors::HandleError(error);
			}
			finally
			{
				Marshal::FreeHGlobal(p);
			}	

			if (cs != nullptr)
			{
				result->Name = gcnew String(cs->name);

				result->A = cs->a;
				result->B = cs->b;
				result->C = cs->c;
				result->Alpha = cs->alpha;
				result->Beta = cs->beta;
				result->Gamma = cs->gamma;
				result->Volume = cs->volume;

				int atomCount = cs->n_atom;
				for (int i = 0; i < atomCount; i++)
				{
					::Crystal_Atom* ca = &cs->atom[i];
					CrystalAtom item;

					item.Zatom = ca->Zatom;
					item.fraction = ca->fraction;
					item.x = ca->x;
					item.y = ca->y;
					item.z = ca->z;

					result->Atoms->Add(item);
				}
			}

			return result;
		}

		/// <summary>Gets the names of crystals within within the CrystalArray object.</summary>
		/// <returns>A list of strings containing the names.</returns>
		List<String^>^ GetNames()
		{
			int i;
			List<String^>^ result;

			result = gcnew List<String^>;

			if (ca != NULL)
			{
				for (i = 0; i < ca->n_crystal; i++)
				{
					result->Add(gcnew String(ca->crystal[i].name));
				}
			}

			return result;
		}

		/// <summary>Gets the names of default crystals defined within the XrayLib library.</summary>
		/// <returns>A list of strings containing the names.</returns>
		static List<String^>^ GetDefaultNames()
		{
			char **names;
			int i;
			List<String^>^ result;

			result = gcnew List<String^>;

			::xrl_error *error = nullptr;
			names = ::Crystal_GetCrystalsList(NULL, 0, &error);
			Errors::HandleError(error);
			for (i = 0; names[i] != NULL; i++) 
			{
				result->Add(gcnew String(names[i]));
				::xrlFree(names[i]);
			}
			::xrlFree(names);

			return result;
		}
	};
}
