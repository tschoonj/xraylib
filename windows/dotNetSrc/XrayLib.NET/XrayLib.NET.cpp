/*
	XrayLib.NET copyright (c) 2010-2021 Matthew Wormington. All rights reserved.
	
	File: XrfLibNET.cpp
	Author: Matthew Wormington
	Language: C++/CLI   
	Compiler: Microsoft Visual Studio 2019
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
#include "Element.h"
#include "XrayLib.NET.h"
#include "..\XrayLib\xraylib.h"
#include "Errors.h"
#include "Compound.h"
#include "Diffraction.h"
#include "Radionuclides.h"

using namespace System::Runtime::InteropServices;

void Science::XrayLib::XrayInit()
{
	::XRayInit();
}

double Science::XrayLib::AtomicWeight( int Z )
{
	::xrl_error *error = nullptr;
	double result = ::AtomicWeight(Z, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::ElementDensity(int Z)
{
	::xrl_error *error = nullptr;
	double result = ::ElementDensity(Z, &error);
	Errors::HandleError(error);
	return result;
}

Science::ElementData Science::XrayLib::GetElementData( int Z )
{
	return Elements::GetData(Z);
}

// Cross sections (cm2/g)
double Science::XrayLib::CS_Total(int Z, double E)
{
	::xrl_error *error = nullptr;
	double result = ::CS_Total(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_Photo(int Z, double E)
{
	::xrl_error *error = nullptr;
	double result = ::CS_Photo(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_Rayl(int Z, double E)
{
	::xrl_error *error = nullptr;
	double result = ::CS_Rayl(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_Compt(int Z, double E)
{
	::xrl_error *error = nullptr;
	double result = ::CS_Compt(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_Energy(int Z, double E)
{
	::xrl_error *error = nullptr;
	double result = ::CS_Energy(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

// barn/atom
double Science::XrayLib::CSb_Total(int Z, double E)
{
	::xrl_error *error = nullptr;
	double result = ::CSb_Total(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_Photo(int Z, double E)
{
	::xrl_error *error = nullptr;
	double result = ::CSb_Photo(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_Rayl(int Z, double E)
{
	::xrl_error *error = nullptr;
	double result = ::CSb_Rayl(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_Compt(int Z, double E)
{
	::xrl_error *error = nullptr;
	double result = ::CSb_Compt(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_KN(double E)
{
	::xrl_error *error = nullptr;
	double result = ::CS_KN(E, &error);
	Errors::HandleError(error);
	return result;
}

// Unpolarized differential scattering cross sections
double Science::XrayLib::DCS_Thoms(double theta)
{
	::xrl_error *error = nullptr;
	double result = ::DCS_Thoms(theta, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::DCS_KN(double E, double theta)
{
	::xrl_error *error = nullptr;
	double result = ::DCS_KN(E, theta, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::DCS_Rayl(int Z, double E, double theta)
{
	::xrl_error *error = nullptr;
	double result = ::DCS_Rayl(Z, E, theta, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::DCS_Compt(int Z, double E, double theta)
{
	::xrl_error *error = nullptr;
	double result = ::DCS_Compt(Z, E, theta, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::DCSb_Rayl(int Z, double E, double theta)
{
	::xrl_error *error = nullptr;
	double result = ::DCSb_Rayl(Z, E, theta, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::DCSb_Compt(int Z, double E, double theta)
{
	::xrl_error *error = nullptr;
	double result = ::DCSb_Compt(Z, E, theta, &error);
	Errors::HandleError(error);
	return result;
}

// Polarized differential scattering cross sections
double Science::XrayLib::DCSP_Thoms(double theta, double phi)
{
	::xrl_error *error = nullptr;
	double result = ::DCSP_Thoms(theta, phi, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::DCSP_KN(double E, double theta, double phi)
{
	::xrl_error *error = nullptr;
	double result = ::DCSP_KN(E, theta, phi, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::DCSP_Rayl(int Z, double E, double theta, double phi)
{
	::xrl_error *error = nullptr;
	double result = ::DCSP_Rayl(Z, E, theta, phi, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::DCSP_Compt(int Z, double E, double theta, double phi)
{
	::xrl_error *error = nullptr;
	double result = ::DCSP_Compt(Z, E, theta, phi, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::DCSPb_Rayl(int Z, double E, double theta, double phi)
{
	::xrl_error *error = nullptr;
	double result = ::DCSPb_Rayl(Z, E, theta, phi, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::DCSPb_Compt(int Z, double E, double theta, double phi)
{
	::xrl_error *error = nullptr;
	double result = ::DCSPb_Compt(Z, E, theta, phi, &error);
	Errors::HandleError(error);
	return result;
}

// Scattering factors
double Science::XrayLib::FF_Rayl(int Z, double q)
{
	::xrl_error *error = nullptr;
	double result = ::FF_Rayl(Z, q, &error);
	Errors::HandleError(error);
	return result;
}

double  Science::XrayLib::SF_Compt(int Z, double q)
{
	::xrl_error *error = nullptr;
	double result = ::SF_Compt(Z, q, &error);
	Errors::HandleError(error);
	return result;
}

double  Science::XrayLib::MomentTransf(double E, double theta)
{
	::xrl_error *error = nullptr;
	double result = ::MomentTransf(E, theta, &error);
	Errors::HandleError(error);
	return result;
}

// X-ray fluorescent line energy
double Science::XrayLib::LineEnergy(int Z, int line)
{
	::xrl_error *error = nullptr;
	double result = ::LineEnergy(Z, line, &error);
	Errors::HandleError(error);
	return result;
}

// Fluorescence yield 
double Science::XrayLib::FluorYield(int Z, int shell)
{
	::xrl_error *error = nullptr;
	double result = ::FluorYield(Z, shell, &error);
	Errors::HandleError(error);
	return result;
}

// Coster-Kronig transition Probability
double Science::XrayLib::CosKronTransProb(int Z, int trans)
{
	::xrl_error *error = nullptr;
	double result = ::CosKronTransProb(Z, trans, &error);
	Errors::HandleError(error);
	return result;
}

// Absorption-edge energies     
double Science::XrayLib::EdgeEnergy(int Z, int shell)
{
	::xrl_error *error = nullptr;
	double result = ::EdgeEnergy(Z, shell, &error);
	Errors::HandleError(error);
	return result;
}

// Jump ratio
double Science::XrayLib::JumpFactor(int Z, int shell)
{
	::xrl_error *error = nullptr;
	double result = ::JumpFactor(Z, shell, &error);
	Errors::HandleError(error);
	return result;
}

// Fluorescent-lines cross sections
double Science::XrayLib::CS_FluorLine(int Z, int line, double E)
{
	::xrl_error *error = nullptr;
	double result = ::CS_FluorLine(Z, line, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_FluorLine(int Z, int line, double E)
{
	::xrl_error *error = nullptr;
	double result = ::CSb_FluorLine(Z, line, E, &error);
	Errors::HandleError(error);
	return result;
}

// Fluorescence cross sections for an entire shell 
double Science::XrayLib::CS_FluorShell(int Z, int shell, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CS_FluorShell(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_FluorShell(int Z, int shell, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CSb_FluorShell(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

// Fractional radiative rate
double Science::XrayLib::RadRate(int Z, int line)
{
	::xrl_error *error = nullptr;
	double result = ::RadRate(Z, line, &error);
	Errors::HandleError(error);
	return result;
}

// Photon energy after Compton scattering
double Science::XrayLib::ComptonEnergy(double E0, double theta)
{
	::xrl_error *error = nullptr;
	double result = ::ComptonEnergy(E0, theta, &error);
	Errors::HandleError(error);
	return result;
}

// Anomalous scattering factors
double Science::XrayLib::Fi( int Z, double E )
{
	::xrl_error *error = nullptr;
	double result = ::Fi(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::Fii( int Z, double E )
{
	::xrl_error *error = nullptr;
	double result = ::Fii(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_Photo_Total( int Z, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CS_Photo_Total(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_Photo_Total( int Z, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CSb_Photo_Total(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_Photo_Partial( int Z, int shell, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CS_Photo_Partial(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_Photo_Partial( int Z, int shell, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CSb_Photo_Partial(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_Total_Kissel( int Z, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CS_Total_Kissel(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_Total_Kissel( int Z, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CSb_Total_Kissel(Z, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_FluorLine_Kissel(int Z, int line, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CS_FluorLine_Kissel(Z, line, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_FluorLine_Kissel(int Z, int line, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CSb_FluorLine_Kissel(Z, line, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_FluorLine_Kissel_Cascade( int Z, int line, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CS_FluorLine_Kissel_Cascade(Z, line, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_FluorLine_Kissel_Cascade( int Z, int line, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CSb_FluorLine_Kissel_Cascade(Z, line, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_FluorLine_Kissel_Nonradiative_Cascade( int Z, int line, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CS_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_FluorLine_Kissel_Nonradiative_Cascade( int Z, int line, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CSb_FluorLine_Kissel_Nonradiative_Cascade(Z, line, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_FluorLine_Kissel_Radiative_Cascade( int Z, int line, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CS_FluorLine_Kissel_Radiative_Cascade(Z, line, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_FluorLine_Kissel_Radiative_Cascade( int Z, int line, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CSb_FluorLine_Kissel_Radiative_Cascade(Z, line, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_FluorLine_Kissel_No_Cascade( int Z, int line, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CS_FluorLine_Kissel_no_Cascade(Z, line, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_FluorLine_Kissel_No_Cascade( int Z, int line, double E )
{
	::xrl_error *error = nullptr;
	double result = ::CSb_FluorLine_Kissel_no_Cascade(Z, line, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_FluorShell_Kissel(int Z, int shell, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CS_FluorShell_Kissel(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_FluorShell_Kissel(int Z, int shell, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CSb_FluorShell_Kissel(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_FluorShell_Kissel_Cascade(int Z, int shell, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CS_FluorShell_Kissel_Cascade(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_FluorShell_Kissel_Cascade(int Z, int shell, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CSb_FluorShell_Kissel_Cascade(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_FluorShell_Kissel_Nonradiative_Cascade(int Z, int shell, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CS_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_FluorShell_Kissel_Nonradiative_Cascade(int Z, int shell, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CSb_FluorShell_Kissel_Nonradiative_Cascade(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_FluorShell_Kissel_Radiative_Cascade(int Z, int shell, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CS_FluorShell_Kissel_Radiative_Cascade(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_FluorShell_Kissel_Radiative_Cascade(int Z, int shell, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CSb_FluorShell_Kissel_Radiative_Cascade(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_FluorShell_Kissel_No_Cascade(int Z, int shell, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CS_FluorShell_Kissel_no_Cascade(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CSb_FluorShell_Kissel_No_Cascade(int Z, int shell, double E)
{
	::xrl_error* error = nullptr;
	double result = ::CSb_FluorShell_Kissel_no_Cascade(Z, shell, E, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::CS_Total_CP( String^ compound, double E )
{
	double result = -1.0;
	
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		::xrl_error *error = nullptr;
		result = ::CS_Total_CP(pCompound, E, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::CS_Photo_CP(pCompound, E, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::CS_Rayl_CP(pCompound, E, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::CS_Compt_CP(pCompound, E, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::CSb_Total_CP(pCompound, E, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::CSb_Photo_CP(pCompound, E, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::CSb_Rayl_CP(pCompound, E, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::CSb_Compt_CP(pCompound, E, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::DCS_Rayl_CP(pCompound, E, theta, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::DCS_Compt_CP(pCompound, E, theta, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::DCSb_Rayl_CP(pCompound, E, theta, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::DCSb_Compt_CP(pCompound, E, theta, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::DCSP_Rayl_CP(pCompound, E, theta, phi, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::DCSP_Compt_CP(pCompound, E, theta, phi, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::DCSPb_Rayl_CP(pCompound, E, theta, phi, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::DCSPb_Compt_CP(pCompound, E, theta, phi, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::CS_Photo_Total_CP(pCompound, E, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::CSb_Photo_Total_CP(pCompound, E, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::CS_Total_Kissel_CP(pCompound, E, &error);
		Errors::HandleError(error);
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
		::xrl_error *error = nullptr;
		result = ::CSb_Total_Kissel_CP(pCompound, E, &error);
		Errors::HandleError(error);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}

double Science::XrayLib::CS_Energy_CP(String^ compound, double E)
{
	double result = -1.0;

	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{
		char* pCompound = static_cast<char*>(p.ToPointer());
		::xrl_error *error = nullptr;
		result = ::CS_Energy_CP(pCompound, E, &error);
		Errors::HandleError(error);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}

	return result;
}


double Science::XrayLib::ElectronConfig( int Z, int shell )
{
	::xrl_error *error = nullptr;
	double result = ::ElectronConfig(Z, shell, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::ComptonProfile( int Z, double pz )
{
	::xrl_error *error = nullptr;
	double result = ::ComptonProfile(Z, pz, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::ComptonProfile_Partial( int Z, int shell, double pz )
{
	::xrl_error *error = nullptr;
	double result = ::ComptonProfile_Partial(Z, shell, pz, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::Refractive_Index_Re( String^ compound, double E, double density )
{
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		::xrl_error *error = nullptr;
		double result = ::Refractive_Index_Re(pCompound, E, density, &error);
		Errors::HandleError(error);
		return result;
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
		::xrl_error *error = nullptr;
		double result = ::Refractive_Index_Im(pCompound, E, density, &error);
		Errors::HandleError(error);
		return result;
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}
}


Numerics::Complex Science::XrayLib::Refractive_Index( String^ compound, double E, double density )
{
	IntPtr p = Marshal::StringToHGlobalAnsi(compound);
	try
	{					
		char* pCompound = static_cast<char*>(p.ToPointer());
		::xrlComplex z;
		::xrl_error *error = nullptr;
		::Refractive_Index2(pCompound, E, density, &z, &error);
		Errors::HandleError(error);

		return Numerics::Complex(z.re, z.im);
	}
	finally
	{
		Marshal::FreeHGlobal(p);
	}
}

double Science::XrayLib::AtomicLevelWidth( int Z, int shell )
{
	::xrl_error *error = nullptr;
	double result = ::AtomicLevelWidth(Z, shell, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::AugerRate( int Z, int auger_trans )
{
	::xrl_error *error = nullptr;
	double result = ::AugerRate(Z, auger_trans, &error);
	Errors::HandleError(error);
	return result;
}

double Science::XrayLib::AugerYield(int Z, int shell)
{
	::xrl_error *error = nullptr;
	double result = ::AugerYield(Z, shell, &error);
	Errors::HandleError(error);
	return result;
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