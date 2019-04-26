/* Copyright (C) 2019 Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans 'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef XRAYLIB_DEPRECATED_PRIVATE_H
#define XRAYLIB_DEPRECATED_PRIVATE_H

/* taken from glib */
#ifdef __ICC
#define XRL_GNUC_BEGIN_IGNORE_DEPRECATIONS                \
  _Pragma ("warning (push)")                            \
  _Pragma ("warning (disable:1478)")
#define XRL_GNUC_END_IGNORE_DEPRECATIONS                  \
  _Pragma ("warning (pop)")
#elif    __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#define XRL_GNUC_BEGIN_IGNORE_DEPRECATIONS                \
  _Pragma ("GCC diagnostic push")                       \
  _Pragma ("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#define XRL_GNUC_END_IGNORE_DEPRECATIONS                  \
  _Pragma ("GCC diagnostic pop")
#elif defined (_MSC_VER) && (_MSC_VER >= 1500)
#define XRL_GNUC_BEGIN_IGNORE_DEPRECATIONS                \
  __pragma (warning (push))  \
  __pragma (warning (disable : 4996))
#define XRL_GNUC_END_IGNORE_DEPRECATIONS                  \
  __pragma (warning (pop))
#elif defined (__clang__)
#define XRL_GNUC_BEGIN_IGNORE_DEPRECATIONS \
  _Pragma("clang diagnostic push") \
  _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")
#define XRL_GNUC_END_IGNORE_DEPRECATIONS \
  _Pragma("clang diagnostic pop")
#else
#define XRL_GNUC_BEGIN_IGNORE_DEPRECATIONS
#define XRL_GNUC_END_IGNORE_DEPRECATIONS
#endif
#endif
