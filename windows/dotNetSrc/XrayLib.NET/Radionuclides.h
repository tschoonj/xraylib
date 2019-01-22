/*
	XrayLib.NET copyright (c) 2010-2019 Matthew Wormington. All rights reserved.

	File: Radionuclides.h
	Author: Matthew Wormington
	Language: C++/CLI
	Compiler: Microsoft Visual Studio 2017
	Created: October 8, 2014
	$Version:$
	$Revision:$
	$RevDate:$

	Description:
	Contains a managed class for radionuclide data for use with XrayLib.NET.

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
#include "..\XrayLib\xraylib-radionuclides.h"
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

	/// <summary>
	/// A managed wrapper for XrayLib's radionuclide data.
	/// </summary>
	public ref class RadionuclideData
	{
	private:
		List<int>^ _xrayLines;
		List<double>^ _xrayIntensities;
		List<double>^ _gammaEnergies;
		List<double>^ _gammaIntensities;
		List<String^>^ _names;

		void GetNames()
		{
			char **names;
			int i;
			String^ name;

			_names->Clear();

			::xrl_error *error = nullptr;
			names = GetRadioNuclideDataList(NULL, &error);
			Errors::HandleError(error);

			for (i = 0; names[i] != NULL; i++)
			{
				name = gcnew String(names[i]);
				_names->Add(name);
				xrlFree(names[i]);
			}
			xrlFree(names);
		}

		void Initialize(::radioNuclideData* rd)
		{
			_xrayLines = gcnew List<int>();
			_xrayIntensities = gcnew List<double>();
			_gammaEnergies = gcnew List<double>();
			_gammaIntensities = gcnew List<double>();
			_names = gcnew List<String^>();
			Clear();

			if (rd != nullptr)
			{
				Name = gcnew String(rd->name);
				Z = rd->Z;
				A = rd->A;
				N = rd->N;
				XrayCount = rd->nXrays;
				for (int i = 0; i < XrayCount; i++)
				{
					_xrayLines->Add(rd->XrayLines[i]);
					_xrayIntensities->Add(rd->XrayIntensities[i]);
				}
				GammaCount = rd->nGammas;
				for (int i = 0; i < GammaCount; i++)
				{
					_gammaEnergies->Add(rd->GammaEnergies[i]);
					_gammaIntensities->Add(rd->GammaIntensities[i]);
				}

				GammaCount = rd->nGammas;
			}
		}

	public:
		/// <summary>A string containing the mass number (A), followed by the chemical element (e.g. 55Fe)</summary>
		property String^ Name;
		/// <summary>Atomic number of the radionuclide</summary>
		property int Z;
		/// <summary>Mass number of the radionuclide</summary>
		property int A;
		/// <summary>Number of neutrons of the radionuclide</summary>
		property int N;
		/// <summary>Atomic number of the nuclide after decay, and which should be used in calculating the energy of the emitted X-ray lines</summary>
		property int ZXray;
		/// <summary>Number of emitted characteristic X-rays</summary>
		property int XrayCount;
		/// <summary>Number of emitted gamma-rays</summary>
		property int GammaCount;

		/// <summary>List of *_LINE macros, identifying the emitted X-rays</summary>
		property List<int>^ XrayLines
		{
			List<int>^ get()
			{
				return _xrayLines;
			}
		}

		/// <summary>List of photons per disintegration, one value per emitted X-ray</summary>
		property List<double>^ XrayIntensities
		{
			List<double>^ get()
			{
				return _xrayIntensities;
			}
		}

		/// <summary>List of all available radionuclide names</summary>
		property List<String^>^ Names
		{
			List<String^>^ get()
			{
				return _names;
			}
		}

		/// <summary>List of emitted gamma-ray energies</summary>
		property List<double>^ GammaEnergies
		{
			List<double>^ get()
			{
				return _gammaEnergies;
			}
		}

		/// <summary>List of emitted gamma-ray photons per disintegration</summary>
		property List<double>^ GammaIntensities
		{
			List<double>^ get()
			{
				return _gammaIntensities;
			}
		}

		/// <summary> Default constructor. </summary>
		RadionuclideData()
		{
			Initialize(nullptr);
		}

		/// <summary>Constructor. </summary>
		/// <param name="index">Index of the radionuclide in the internal table</param>
		RadionuclideData(int index)
		{
			::radioNuclideData* rd;
			::xrl_error *error = nullptr;
			rd = GetRadioNuclideDataByIndex(index, &error);
			Errors::HandleError(error);
			Initialize(rd);
			if (rd != nullptr)
				FreeRadioNuclideData(rd);
		}

		/// <summary>Constructor. </summary>
		/// <param name="name">Name of the radionuclide</param>
		RadionuclideData(String^ name)
		{
			::radioNuclideData* rd;

			char* cName = new char[name->Length];
			IntPtr p = Marshal::StringToHGlobalAnsi(name);
			try
			{
				cName = static_cast<char*>(p.ToPointer());
				::xrl_error *error = nullptr;
				rd = GetRadioNuclideDataByName(cName, &error);
				Errors::HandleError(error);
				Initialize(rd);
				if (rd != nullptr)
					FreeRadioNuclideData(rd);
			}
			finally
			{
				Marshal::FreeHGlobal(p);
			}
		}

		/// <summary> Clears this object to its blank/initial state. </summary>
		void Clear()
		{
			Name = "";
			Z = 0;
			A = 0;
			N = 0;
			ZXray = 0;
			XrayCount = 0;
			GammaCount = 0;
			_xrayLines->Clear();
			_xrayIntensities->Clear();
			_gammaEnergies->Clear();
			_gammaIntensities->Clear();

			GetNames();
		}

		/// <summary>Convert this object into a string representation. </summary>
		/// <returns>Null if it fails, else a string representation of this object. </returns>
		virtual String^ ToString() override
		{
			StringBuilder^ sb = gcnew StringBuilder();
			sb->AppendLine(String::Format("Name: {0}", Name));
			sb->AppendLine(String::Format("Z: {0}", Z));
			sb->AppendLine(String::Format("A: {0}", A));
			sb->AppendLine(String::Format("N: {0}", N));
			sb->AppendLine(String::Format("ZXray: {0}", ZXray));
			sb->AppendLine(String::Format("XrayCount: {0}", XrayCount));
			sb->AppendLine(String::Format("XrayLine, XrayIntensity"));
			for (int i = 0; i < XrayCount; i++)
				sb->AppendLine(String::Format("{0}, {1}", _xrayLines[i], _xrayIntensities[i]));
			sb->AppendLine(String::Format("GammaCount: {0}", GammaCount));
			sb->AppendLine(String::Format("GammaEnergy, GammaIntensity"));
			for (int i = 0; i < GammaCount; i++)
				sb->AppendLine(String::Format("{0}, {1}", _gammaEnergies[i], _gammaIntensities[i]));

			return sb->ToString();
		}
	};
}