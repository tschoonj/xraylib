/*
	XrayLib.NET copyright (c) 2010-2011 Matthew Wormington. All rights reserved.
	
	File: XrfLibNET.cpp
	Author: Matthew Wormington
	Language: C++/CLI   
	Compiler: Microsoft Visual Studio 2010
	Created: September 4, 2010
	$Version:$
	$Revision:$
	$RevDate:$

	Description:
	Contains the implementation of a managed wrapper class around the native 
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

#include "stdafx.h"

#include "Compound.h"
#include "Diffraction.h"
#include "Element.h"
#include "XrayLib.NET.h"
#include "..\XrayLib\xraylib.h"

using namespace System::Runtime::InteropServices;

Science::XrayLibException::XrayLibException( System::String ^message )
{

}

Science::XrayLib::XrayLib()
{
	// Initialize the library
	::XRayInit();
}

// Error Handling
void Science::XrayLib::SetHardExit( int hard_exit )
{
	::SetHardExit(hard_exit);
}

void Science::XrayLib::SetExitStatus( int exit_status )
{
	::SetExitStatus(exit_status);
}

int Science::XrayLib::GetExitStatus()
{
	return ::GetExitStatus();
}

void Science::XrayLib::SetErrorMessages( int status )
{
	::SetErrorMessages(status);
}

int Science::XrayLib::GetErrorMessages( void )
{
	return ::GetErrorMessages();
}

double Science::XrayLib::AtomicWeight( int Z )
{
	return ::AtomicWeight(Z);
}

Science::ElementData Science::XrayLib::GetElementData( int Z )
{
	return Elements::GetData(Z);
}

// Cross sections (cm2/g)
double Science::XrayLib::CS_Total(int Z, double E)
{
	return ::CS_Total(Z, float(E));
}

double Science::XrayLib::CS_Photo(int Z, double E)
{
	return ::CS_Photo(Z, float(E));
}

double Science::XrayLib::CS_Rayl(int Z, double E)
{
	return ::CS_Rayl(Z, float(E));
}

double Science::XrayLib::CS_Compt(int Z, double E)
{
	return ::CS_Compt(Z, float(E));
}

// barn/atom
double Science::XrayLib::CSb_Total(int Z, double E)
{
	return ::CSb_Total(Z, float(E));
}

double Science::XrayLib::CSb_Photo(int Z, double E)
{
	return ::CSb_Photo(Z, float(E));
}

double Science::XrayLib::CSb_Rayl(int Z, double E)
{
	return ::CSb_Rayl(Z, float(E));
}

double Science::XrayLib::CSb_Compt(int Z, double E)
{
	return ::CSb_Compt(Z, float(E));
}

double Science::XrayLib::CS_KN(double E)
{
	return ::CS_KN(float(E));
}

// Unpolarized differential scattering cross sections
double Science::XrayLib::DCS_Thoms(double theta)
{
	return ::DCS_Thoms(float(theta));
}

double Science::XrayLib::DCS_KN(double E, double theta)
{
	return ::DCS_KN(float(E), (float)theta);
}

double Science::XrayLib::DCS_Rayl(int Z, double E, double theta)
{
	return ::DCS_Rayl(Z, float(E), float(theta));
}

double Science::XrayLib::DCS_Compt(int Z, double E, double theta)
{
	return ::DCS_Compt(Z, float(E), float(theta));
}

double Science::XrayLib::DCSb_Rayl(int Z, double E, double theta)
{
	return ::DCSb_Rayl(Z, float(E), float(theta));
}

double Science::XrayLib::DCSb_Compt(int Z, double E, double theta)
{
	return ::DCSb_Compt(Z, float(E), float(theta));
}

// Polarized differential scattering cross sections
double Science::XrayLib::DCSP_Thoms(double theta, double phi)
{
	return ::DCSP_Thoms(float(theta), float(phi));
}

double Science::XrayLib::DCSP_KN(double E, double theta, double phi)
{
	return ::DCSP_KN(float(E), float(theta), float(phi));
}

double Science::XrayLib::DCSP_Rayl(int Z, double E, double theta, double phi)
{
	return ::DCSP_Rayl(Z, float(E), float(theta), float(phi));
}

double Science::XrayLib::DCSP_Compt(int Z, double E, double theta, double phi)
{
	return ::DCSP_Compt(Z, float(E), float(theta), float(phi));
}

double Science::XrayLib::DCSPb_Rayl(int Z, double E, double theta, double phi)
{
	return ::DCSPb_Rayl(Z, float(E), float(theta), float(phi));
}

double Science::XrayLib::DCSPb_Compt(int Z, double E, double theta, double phi)
{
	return ::DCSPb_Compt(Z, float(E), float(theta), float(phi));
}

// Scattering factors
double Science::XrayLib::FF_Rayl(int Z, double q)
{
	return ::FF_Rayl(Z, float(q));
}

double  Science::XrayLib::SF_Compt(int Z, double q)
{
	return:: SF_Compt(Z, float(q));
}

double  Science::XrayLib::MomentTransf(double E, double theta)
{
	return:: MomentTransf(float(E), float(theta));
}

// X-ray fluorescent line energy
double Science::XrayLib::LineEnergy(int Z, int line)
{
	return:: LineEnergy(Z, line);
}

// Fluorescence yield 
double Science::XrayLib::FluorYield(int Z, int shell)
{
	return:: FluorYield(Z, shell); 
}

// Coster-Kronig transition Probability
double Science::XrayLib::CosKronTransProb(int Z, int trans)
{
	return:: CosKronTransProb(Z, trans);
}

// Absorption-edge energies     
double Science::XrayLib::EdgeEnergy(int Z, int shell)
{
	return:: EdgeEnergy(Z, shell);
}

// Jump ratio
double Science::XrayLib::JumpFactor(int Z, int shell)
{
	return:: JumpFactor(Z, shell);
}

// Fluorescent-lines cross sections
double Science::XrayLib::CS_FluorLine(int Z, int line, double E)
{
	return ::CS_FluorLine(Z, line, float(E)); 
}

double Science::XrayLib::CSb_FluorLine(int Z, int line, double E)
{
	return ::CSb_FluorLine(Z, line, float(E)); 
}

// Fractional radiative rate
double Science::XrayLib::RadRate(int Z, int line)
{
	return ::RadRate(Z, line); 
}

// Photon energy after Compton scattering
double Science::XrayLib::ComptonEnergy(double E0, double theta)
{
	return ::ComptonEnergy(float(E0), float(theta));
}

// Anomalous scattering factors
double Science::XrayLib::Fi( int Z, double E )
{
	return ::Fi(Z, float(E)); 
}

double Science::XrayLib::Fii( int Z, double E )
{
	return ::Fii(Z, float(E)); 
}

double Science::XrayLib::CS_Photo_Total( int Z, double E )
{
	return ::CS_Photo_Total(Z, float(E));
}

double Science::XrayLib::CSb_Photo_Total( int Z, double E )
{
	return ::CSb_Photo_Total(Z, float(E));
}

double Science::XrayLib::CS_Photo_Partial( int Z, int shell, double E )
{
	return ::CS_Photo_Partial(Z, shell, float(E));
}

double Science::XrayLib::CSb_Photo_Partial( int Z, int shell, double E )
{
	return ::CSb_Photo_Partial(Z, shell, float(E));
}

double Science::XrayLib::CS_FluorLine_Kissel( int Z, int line, double E )
{
	return ::CS_FluorLine_Kissel(Z, line, float(E));
}

double Science::XrayLib::CSb_FluorLine_Kissel( int Z, int line, double E )
{
	return ::CSb_FluorLine_Kissel(Z, line, float(E));
}

double Science::XrayLib::CS_Total_Kissel( int Z, double E )
{
	return ::CS_Total_Kissel(Z, float(E));
}

double Science::XrayLib::CSb_Total_Kissel( int Z, double E )
{
	return ::CSb_Total_Kissel(Z, float(E));
}

double Science::XrayLib::CS_FluorLine_Kissel_Cascade( int Z, int line, double E )
{
	return ::CS_FluorLine_Kissel_Cascade(Z, line, float(E));
}

double Science::XrayLib::CSb_FluorLine_Kissel_Cascade( int Z, int line, double E )
{
	return ::CSb_FluorLine_Kissel_Cascade(Z, line, float(E));
}

double Science::XrayLib::CS_FluorLine_Kissel_Nonradiative_Cascade( int Z, int line, double E )
{
	return ::CS_FluorLine_Kissel_Nonradiative_Cascade(Z, line, float(E));
}

double Science::XrayLib::CSb_FluorLine_Kissel_Nonradiative_Cascade( int Z, int line, double E )
{
	return ::CSb_FluorLine_Kissel_Nonradiative_Cascade(Z, line, float(E));
}

double Science::XrayLib::CS_FluorLine_Kissel_Radiative_Cascade( int Z, int line, double E )
{
	return ::CS_FluorLine_Kissel_Radiative_Cascade(Z, line, float(E));
}

double Science::XrayLib::CSb_FluorLine_Kissel_Radiative_Cascade( int Z, int line, double E )
{
	return ::CSb_FluorLine_Kissel_Radiative_Cascade(Z, line, float(E));
}

double Science::XrayLib::CS_FluorLine_Kissel_No_Cascade( int Z, int line, double E )
{
	return ::CS_FluorLine_Kissel_no_Cascade(Z, line, float(E));
}

double Science::XrayLib::CSb_FluorLine_Kissel_No_Cascade( int Z, int line, double E )
{
	return ::CSb_FluorLine_Kissel_no_Cascade(Z, line, float(E));
}

double Science::XrayLib::CS_Total_CP( String^ compound, double E )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::CS_Total_CP(pCompound, (float)E);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::CS_Photo_CP( String^ compound, double E )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::CS_Photo_CP(pCompound, (float)E);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::CS_Rayl_CP( String^ compound, double E )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::CS_Rayl_CP(pCompound, (float)E);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::CS_Compt_CP( String^ compound, double E )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::CS_Compt_CP(pCompound, (float)E);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::CSb_Total_CP( String^ compound, double E )
{
	double result = -1.0;

	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::CSb_Total_CP(pCompound, (float)E);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::CSb_Photo_CP( String^ compound, double E )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::CSb_Photo_CP(pCompound, (float)E);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::CSb_Rayl_CP( String^ compound, double E )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::CSb_Rayl_CP(pCompound, (float)E);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::CSb_Compt_CP( String^ compound, double E )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::CSb_Compt_CP(pCompound, (float)E);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::DCS_Rayl_CP( String^ compound, double E, double theta )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::DCS_Rayl_CP(pCompound, (float)E, (float)theta);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::DCS_Compt_CP( String^ compound, double E, double theta )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::DCS_Compt_CP(pCompound, (float)E, (float)theta);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::DCSb_Rayl_CP( String^ compound, double E, double theta )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::DCSb_Rayl_CP(pCompound, (float)E, (float)theta);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::DCSb_Compt_CP( String^ compound, double E, double theta )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::DCSb_Compt_CP(pCompound, (float)E, (float)theta);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::DCSP_Rayl_CP( String^ compound, double E, double theta, double phi )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::DCSP_Rayl_CP(pCompound, (float)E, (float)theta, (float)phi);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::DCSP_Compt_CP( String^ compound, double E, double theta, double phi )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::DCSP_Compt_CP(pCompound, (float)E, (float)theta, (float)phi);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::DCSPb_Rayl_CP( String^ compound, double E, double theta, double phi )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::DCSPb_Rayl_CP(pCompound, (float)E, (float)theta, (float)phi);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::DCSPb_Compt_CP( String^ compound, double E, double theta, double phi )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::DCSPb_Compt_CP(pCompound, (float)E, (float)theta, (float)phi);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::CS_Photo_Total_CP( String^ compound, double E )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::CS_Photo_Total_CP(pCompound, (float)E);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::CSb_Photo_Total_CP( String^ compound, double E )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::CSb_Photo_Total_CP(pCompound, (float)E);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::CS_Total_Kissel_CP( String^ compound, double E )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::CS_Total_Kissel_CP(pCompound, (float)E);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::CSb_Total_Kissel_CP( String^ compound, double E )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		result = ::CSb_Total_Kissel_CP(pCompound, (float)E);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::ElectronConfig( int Z, int shell )
{
	return ::ElectronConfig(Z, shell);
}

double Science::XrayLib::ComptonProfile( int Z, double pz )
{
	return ::ComptonProfile(Z, (float)pz);
}

double Science::XrayLib::ComptonProfile_Partial( int Z, int shell, double pz )
{
	return ::ComptonProfile_Partial(Z, shell, (float)pz);
}

double Science::XrayLib::Refractive_Index_Re( String^ compound, double E, double density )
{
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		return ::Refractive_Index_Re(pCompound, (float)E, (float)density);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}
}

double Science::XrayLib::Refractive_Index_Im( String^ compound, double E, double density )
{
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		return ::Refractive_Index_Im(pCompound, (float)E, (float)density);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}
}

double Science::XrayLib::AtomicLevelWidth( int Z, int shell )
{
	return ::AtomicLevelWidth(Z, shell);
}

double Science::XrayLib::AugerRate( int Z, int auger_trans )
{
	return ::AugerRate(Z, auger_trans);
}

System::String ^ Science::XrayLib::IUPACToSiegbahnLineName( System::String ^name )
{
	System::String ^result = "";

	for (int i = 0; i < IUPACLineNames->Length; i++)
	{
		if (String::Compare(name, IUPACLineNames[i], StringComparison::InvariantCultureIgnoreCase) == 0)
		{
			result = SiegbahnLineNames[i];
			break;
		}
	}
	return result;
}

System::String ^ Science::XrayLib::SiegbahnToIUPACLineName( System::String ^name )
{
	System::String ^result = "";

	for(int i = 0; i < SiegbahnLineNames->Length; i++)
	{
		if (String::Compare(name, SiegbahnLineNames[i], StringComparison::InvariantCultureIgnoreCase) == 0)
		{
			result = IUPACLineNames[i];
			break;
		}
	}
	return result;
}


void Science::XrayLib::ElementAndLineFromName( System::String ^elementLine, int %Z, int %line )
{
	System::String ^delimitersStr = " ";
	array<wchar_t> ^delimiters = delimitersStr->ToCharArray();
	System::String ^symbol;
	System::String ^lineName;
	ElementData^ ed;

	double result = -1.0;

	//  Split the name into a element symbol and line name
	array<String^> ^tokens = elementLine->Split(delimiters, StringSplitOptions::RemoveEmptyEntries);
	if (tokens->Length != 2)
		throw gcnew XrayLibException(System::String::Format("Invalid X-ray fluorescence line {0}", elementLine));

	symbol = tokens[0]->ToLower();
	lineName = tokens[1]->ToLower();

	//  Find the atomic number of the specified symbol or throw an
	//  exception if the symbol is unknown
	Z = -99;
	for(int i = 1; i <= 100; i++)
	{
		ed = GetElementData(i);
		if (String::Compare(symbol, ed->Symbol, StringComparison::InvariantCultureIgnoreCase) == 0)
		{
			Z = i;
			break;
		}
	}
	if (Z == -99)
		throw gcnew XrayLibException(System::String::Format("Unknown atomic symbol {0}", tokens[0]));

	//  Find the line number of the specified symbol or throw an
	//  exception if the line is unknown
	line = -99;
	if (lineName == "ka")
		line = KA_LINE;
	else if (lineName == "kb")
		line = KB_LINE;
	else if (lineName == "la")
		line = LA_LINE;
	else if (lineName == "lb")
		line = LB_LINE;
	else if (lineName == "lg")
		line = LG1_LINE;
	else
	{
		for(int i = 0; i < 17; i++)
		{
			if (String::Compare(lineName, SiegbahnLineNames[i], StringComparison::InvariantCultureIgnoreCase) == 0)
			{
				line = SiegbahnLines[i];
				break;
			}
		}
	}
	if (line == -99)
		throw gcnew XrayLibException(System::String::Format("Unknown emission line {0}", tokens[1]));
}

double Science::XrayLib::LineEnergyFromName( System::String ^lineName )
{	
	int Z;
	int line;

	ElementAndLineFromName( lineName, Z, line );
	
	//  Return the X-ray fluorescence energy (keV)
	return LineEnergy(Z, line);
}

int Science::XrayLib::SiegbahnLineIndex( System::String ^name )
{
	System::String ^lineName = name->Trim()->ToLower();

	//  Find the identifier of the specified line
	int line = -99;
	if (lineName == "ka")
		line = KA_LINE;
	else if (lineName == "kb")
		line = KB_LINE;
	else if (lineName == "la")
		line = LA_LINE;
	else if (lineName == "lb")
		line = LB_LINE;
	else if (lineName == "lg")
		line = LG1_LINE;
	else
	{
		for(int i = 0; i < 17; i++)
		{
			if (String::Compare(lineName, SiegbahnLineNames[i], StringComparison::InvariantCultureIgnoreCase) == 0)
			{
				line = SiegbahnLines[i];
				break;
			}
		}
	}

	return line;
}

int Science::XrayLib::AtomicNumber( System::String ^name )
{
	int Z;
	ElementData^ ed;

	System::String ^symbol = name->Trim()->ToLower();

	//  Find the atomic number of the specified symbol
	Z = -99;
	for (int i = 1; i <= 100; i++)
	{
		ed = GetElementData(i);
		if (String::Compare(symbol, ed->Symbol, StringComparison::InvariantCultureIgnoreCase) == 0)
		{
			Z = i;
			break;
		}
	}

	return Z;
}

double Science::XrayLib::SiEscapeEnergy( double energy )
{
	double result = energy - 1.750;
	return result;
}

double Science::XrayLib::SiEscapeFraction( double energy )
{
	const double rho = 2.329;
	const double omegaK = 0.047;
	const double r = 10.8;
	int Z;
	double muI;
	double muK;

	Z = 14; // Si
	muI = CS_Total(Z, energy)*rho;
	muK = CS_Total(Z, 1.750)*rho;

	double result = 0.5*omegaK*(1-1/r)*(1-muK/muI*Math::Log(1+muI/muK));
	return result;
}

