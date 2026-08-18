/*
Copyright (c) 2010-2026, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef XRAYLIB_AUX_PRIVATE_H
#define XRAYLIB_AUX_PRIVATE_H

/*
 * The following types and methods are not visible outside the library!
 *
 * Minimal cross-platform mutex used to serialize access to process-global
 * mutable state (currently the built-in crystal array Crystal_arr, which the
 * SWIG-generated Python bindings mutate through the shared global). Now that the
 * Python modules declare themselves free-threading (no-GIL) compatible, the GIL
 * no longer serializes these accesses for us.
 *
 * On Windows (including MinGW/MSYS builds, which define _WIN32) we use an
 * SRWLOCK: it is statically initializable, requires no cleanup, and lives in
 * kernel32, so no extra library (e.g. libwinpthread) is pulled in. Everywhere
 * else (Linux, macOS, Cygwin) we use a POSIX mutex; on modern systems pthread is
 * part of libc, otherwise AX_PTHREAD / dependency('threads') links it in.
 */

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
typedef SRWLOCK xrl_lock;
#define XRL_LOCK_INITIALIZER SRWLOCK_INIT
#else
#include <pthread.h>
typedef pthread_mutex_t xrl_lock;
#define XRL_LOCK_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#endif

void xrl_lock_acquire(xrl_lock *lock);
void xrl_lock_release(xrl_lock *lock);

#endif
