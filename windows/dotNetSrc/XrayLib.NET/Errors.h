/*
	XrayLib.NET copyright (c) 2010-2019 Matthew Wormington. All rights reserved.

	File: Errors.h
	Author: Matthew Wormington
	Language: C++/CLI
	Compiler: Microsoft Visual Studio 2017
	Created: January 15, 2019
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


#pragma once

using namespace System::IO;

#include "..\XrayLib\xraylib.h"

/// <summary>
/// A namespace that contains scientific classes. 
/// </summary>
namespace Science {

	/// <summary>
	/// A managed wrapper for XrayLib's error functions.
	/// </summary>
	public ref class Errors
	{
	public:

		/// <summary>
		/// Handles library errors by raising a .NET exception.
		/// </summary>
		/// <param name="error">The error.</param>
		static void HandleError(::xrl_error *error)
		{
			if (error == nullptr)
			{
				return;
			}

			int code = error->code;
			String^ message = gcnew String(error->message);
			switch (code)
			{
				case ::XRL_ERROR_MEMORY : 
					throw gcnew OutOfMemoryException(message);
				case ::XRL_ERROR_INVALID_ARGUMENT: 
					throw gcnew ArgumentException(message);
				case ::XRL_ERROR_IO : 
					throw gcnew IOException(message);
				case ::XRL_ERROR_TYPE:
					throw gcnew InvalidCastException(message);
				case ::XRL_ERROR_UNSUPPORTED : 
					throw gcnew NotImplementedException(message);
				case ::XRL_ERROR_RUNTIME : 
					throw gcnew Exception(message);
				default:
					throw gcnew Exception("Unknown exception type raised!");
			}

			::xrl_clear_error(&error);

			String^ s = String::Format("Error {0}: {1}", code, message);
			throw gcnew Science::XrayLibException(s);
		}

		// Error Handling
		/// <summary>
		/// Sets the hard error exit code.
		/// </summary>
		/// <param name="hard_exit">Hard exit code</param>
		static void SetHardExit(int hard_exit)
		{
			::SetHardExit(hard_exit);
		}

		/// <summary>
		/// Sets the exit status code.
		/// </summary>
		/// <param name="exit_status">Exit status code</param>
		static void SetExitStatus(int exit_status)
		{
			::SetExitStatus(exit_status);
		}

		/// <summary>
		/// Gets the exit status code.
		/// </summary>
		/// <returns>Exit status code</returns>
		static int GetExitStatus()
		{
			return ::GetExitStatus();
		}

		/// <summary>	
		/// Sets whether, or not, error messages are displayed. 
		/// </summary>
		/// <param name="status">status is non-zero to display messages</param>
		static void SetErrorMessages(int status)
		{
			::SetErrorMessages(status);
		}

		/// <summary>Gets whether, or now, error messages are displayed. </summary>
		/// <returns>Returns a non-zero if messages are displayed</returns>
		static int GetErrorMessages(void)
		{
			return ::GetErrorMessages();
		}
	};
}