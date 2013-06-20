/*
Copyright (c) 2009, 2010, 2011  Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans, Teemu Ikonen, and David Sagan
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xraylib-defs.h"
#include <xrayglob.h>

/*/////////////////////////////////////////////////////////////////////
/////            Variables                                       /////
/////////////////////////////////////////////////////////////////// */

static int HardExit = 0;
static int ExitStatus = 0;
static int ErrorMessages = 1;

char ShellName[][5] = {
"K",    "L1",   "L2",   "L3",   "M1",   "M2",   "M3",   "M4",   "M5",   "N1",
"N2",   "N3",   "N4",   "N5",   "N6",   "N7",   "O1",   "O2",   "O3",   "O4",
"O5",   "O6",   "O7",   "P1",   "P2",   "P3",   "P4",   "P5"
};

/* must match with contents of lines.h !!!! -> contains 383 linenames */
char LineName[][6] = {
"KL1", "KL2", "KL3", "KM1", "KM2", "KM3", "KM4", "KM5", "KN1", "KN2",
"KN3", "KN4", "KN5", "KN6", "KN7", "KO", "KO1", "KO2", "KO3", "KO4",
"KO5", "KO6", "KO7", "KP", "KP1", "KP2", "KP3", "KP4", "KP5", "L1L2",
"L1L3", "L1M1", "L1M2", "L1M3", "L1M4", "L1M5", "L1N1", "L1N2", "L1N3", "L1N4",
"L1N5", "L1N6", "L1N67", "L1N7", "L1O1", "L1O2", "L1O3", "L1O4", "L1O45", "L1O5",
"L1O6", "L1O7", "L1P1", "L1P2", "L1P23", "L1P3", "L1P4", "L1P5", "L2L3", "L2M1",
"L2M2", "L2M3", "L2M4", "L2M5", "L2N1", "L2N2", "L2N3", "L2N4", "L2N5", "L2N6",
"L2N7", "L2O1", "L2O2", "L2O3", "L2O4", "L2O5", "L2O6", "L2O7", "L2P1", "L2P2",
"L2P23", "L2P3", "L2P4", "L2P5", "L2Q1", "L3M1", "L3M2", "L3M3", "L3M4", "L3M5",
"L3N1", "L3N2", "L3N3", "L3N4", "L3N5", "L3N6", "L3N7", "L3O1", "L3O2", "L3O3",
"L3O4", "L3O45", "L3O5", "L3O6", "L3O7", "L3P1", "L3P2", "L3P23", "L3P3", "L3P4",
"L3P45", "L3P5", "L3Q1", "M1M2", "M1M3", "M1M4", "M1M5", "M1N1", "M1N2", "M1N3",
"M1N4", "M1N5", "M1N6", "M1N7", "M1O1", "M1O2", "M1O3", "M1O4", "M1O5", "M1O6",
"M1O7", "M1P1", "M1P2", "M1P3", "M1P4", "M1P5", "M2M3", "M2M4", "M2M5", "M2N1",
"M2N2", "M2N3", "M2N4", "M2N5", "M2N6", "M2N7", "M2O1", "M2O2", "M2O3", "M2O4",
"M2O5", "M2O6", "M2O7", "M2P1", "M2P2", "M2P3", "M2P4", "M2P5", "M3M4", "M3M5",
"M3N1", "M3N2", "M3N3", "M3N4", "M3N5", "M3N6", "M3N7", "M3O1", "M3O2", "M3O3",
"M3O4", "M3O5", "M3O6", "M3O7", "M3P1", "M3P2", "M3P3", "M3P4", "M3P5", "M3Q1",
"M4M5", "M4N1", "M4N2", "M4N3", "M4N4", "M4N5", "M4N6", "M4N7", "M4O1", "M4O2",
"M4O3", "M4O4", "M4O5", "M4O6", "M4O7", "M4P1", "M4P2", "M4P3", "M4P4", "M4P5",
"M5N1", "M5N2", "M5N3", "M5N4", "M5N5", "M5N6", "M5N7", "M5O1", "M5O2", "M5O3",
"M5O4", "M5O5", "M5O6", "M5O7", "M5P1", "M5P2", "M5P3", "M5P4", "M5P5", "N1N2",
"N1N3", "N1N4", "N1N5", "N1N6", "N1N7", "N1O1", "N1O2", "N1O3", "N1O4", "N1O5",
"N1O6", "N1O7", "N1P1", "N1P2", "N1P3", "N1P4", "N1P5", "N2N3", "N2N4", "N2N5",
"N2N6", "N2N7", "N2O1", "N2O2", "N2O3", "N2O4", "N2O5", "N2O6", "N2O7", "N2P1",
"N2P2", "N2P3", "N2P4", "N2P5", "N3N4", "N3N5", "N3N6", "N3N7", "N3O1", "N3O2",
"N3O3", "N3O4", "N3O5", "N3O6", "N3O7", "N3P1", "N3P2", "N3P3", "N3P4", "N3P5",
"N4N5", "N4N6", "N4N7", "N4O1", "N4O2", "N4O3", "N4O4", "N4O5", "N4O6", "N4O7",
"N4P1", "N4P2", "N4P3", "N4P4", "N4P5", "N5N6", "N5N7", "N5O1", "N5O2", "N5O3",
"N5O4", "N5O5", "N5O6", "N5O7", "N5P1", "N5P2", "N5P3", "N5P4", "N5P5", "N6N7",
"N6O1", "N6O2", "N6O3", "N6O4", "N6O5", "N6O6", "N6O7", "N6P1", "N6P2", "N6P3",
"N6P4", "N6P5", "N7O1", "N7O2", "N7O3", "N7O4", "N7O5", "N7O6", "N7O7", "N7P1",
"N7P2", "N7P3", "N7P4", "N7P5", "O1O2", "O1O3", "O1O4", "O1O5", "O1O6", "O1O7",
"O1P1", "O1P2", "O1P3", "O1P4", "O1P5", "O2O3", "O2O4", "O2O5", "O2O6", "O2O7",
"O2P1", "O2P2", "O2P3", "O2P4", "O2P5", "O3O4", "O3O5", "O3O6", "O3O7", "O3P1",
"O3P2", "O3P3", "O3P4", "O3P5", "O4O5", "O4O6", "O4O7", "O4P1", "O4P2", "O4P3",
"O4P4", "O4P5", "O5O6", "O5O7", "O5P1", "O5P2", "O5P3", "O5P4", "O5P5", "O6O7",
"O6P4", "O6P5", "O7P4", "O7P5", "P1P2", "P1P3", "P1P4", "P1P5", "P2P3", "P2P4", 
"P2P5", "P3P4", "P3P5"
};


char TransName[][6] = {"F1","F12","F13","FP13","F23","FM12","FM13","FM14","FM15",
	"FM23","FM24","FM25","FM34","FM35","FM45"};


char AugerName[][9] = {
"K-L1L1", "K-L1L2", "K-L1L3", "K-L1M1", "K-L1M2", "K-L1M3", "K-L1M4", "K-L1M5",
"K-L1N1", "K-L1N2", "K-L1N3", "K-L1N4", "K-L1N5", "K-L1N6", "K-L1N7", "K-L1O1",
"K-L1O2", "K-L1O3", "K-L1O4", "K-L1O5", "K-L1O6", "K-L1O7", "K-L1P1", "K-L1P2",
"K-L1P3", "K-L1P4", "K-L1P5", "K-L1Q1", "K-L1Q2", "K-L1Q3", "K-L2L1", "K-L2L2",
"K-L2L3", "K-L2M1", "K-L2M2", "K-L2M3", "K-L2M4", "K-L2M5", "K-L2N1", "K-L2N2",
"K-L2N3", "K-L2N4", "K-L2N5", "K-L2N6", "K-L2N7", "K-L2O1", "K-L2O2", "K-L2O3",
"K-L2O4", "K-L2O5", "K-L2O6", "K-L2O7", "K-L2P1", "K-L2P2", "K-L2P3", "K-L2P4",
"K-L2P5", "K-L2Q1", "K-L2Q2", "K-L2Q3", "K-L3L1", "K-L3L2", "K-L3L3", "K-L3M1",
"K-L3M2", "K-L3M3", "K-L3M4", "K-L3M5", "K-L3N1", "K-L3N2", "K-L3N3", "K-L3N4",
"K-L3N5", "K-L3N6", "K-L3N7", "K-L3O1", "K-L3O2", "K-L3O3", "K-L3O4", "K-L3O5",
"K-L3O6", "K-L3O7", "K-L3P1", "K-L3P2", "K-L3P3", "K-L3P4", "K-L3P5", "K-L3Q1",
"K-L3Q2", "K-L3Q3", "K-M1L1", "K-M1L2", "K-M1L3", "K-M1M1", "K-M1M2", "K-M1M3",
"K-M1M4", "K-M1M5", "K-M1N1", "K-M1N2", "K-M1N3", "K-M1N4", "K-M1N5", "K-M1N6",
"K-M1N7", "K-M1O1", "K-M1O2", "K-M1O3", "K-M1O4", "K-M1O5", "K-M1O6", "K-M1O7",
"K-M1P1", "K-M1P2", "K-M1P3", "K-M1P4", "K-M1P5", "K-M1Q1", "K-M1Q2", "K-M1Q3",
"K-M2L1", "K-M2L2", "K-M2L3", "K-M2M1", "K-M2M2", "K-M2M3", "K-M2M4", "K-M2M5",
"K-M2N1", "K-M2N2", "K-M2N3", "K-M2N4", "K-M2N5", "K-M2N6", "K-M2N7", "K-M2O1",
"K-M2O2", "K-M2O3", "K-M2O4", "K-M2O5", "K-M2O6", "K-M2O7", "K-M2P1", "K-M2P2",
"K-M2P3", "K-M2P4", "K-M2P5", "K-M2Q1", "K-M2Q2", "K-M2Q3", "K-M3L1", "K-M3L2",
"K-M3L3", "K-M3M1", "K-M3M2", "K-M3M3", "K-M3M4", "K-M3M5", "K-M3N1", "K-M3N2",
"K-M3N3", "K-M3N4", "K-M3N5", "K-M3N6", "K-M3N7", "K-M3O1", "K-M3O2", "K-M3O3",
"K-M3O4", "K-M3O5", "K-M3O6", "K-M3O7", "K-M3P1", "K-M3P2", "K-M3P3", "K-M3P4",
"K-M3P5", "K-M3Q1", "K-M3Q2", "K-M3Q3", "K-M4L1", "K-M4L2", "K-M4L3", "K-M4M1",
"K-M4M2", "K-M4M3", "K-M4M4", "K-M4M5", "K-M4N1", "K-M4N2", "K-M4N3", "K-M4N4",
"K-M4N5", "K-M4N6", "K-M4N7", "K-M4O1", "K-M4O2", "K-M4O3", "K-M4O4", "K-M4O5",
"K-M4O6", "K-M4O7", "K-M4P1", "K-M4P2", "K-M4P3", "K-M4P4", "K-M4P5", "K-M4Q1",
"K-M4Q2", "K-M4Q3", "K-M5L1", "K-M5L2", "K-M5L3", "K-M5M1", "K-M5M2", "K-M5M3",
"K-M5M4", "K-M5M5", "K-M5N1", "K-M5N2", "K-M5N3", "K-M5N4", "K-M5N5", "K-M5N6",
"K-M5N7", "K-M5O1", "K-M5O2", "K-M5O3", "K-M5O4", "K-M5O5", "K-M5O6", "K-M5O7",
"K-M5P1", "K-M5P2", "K-M5P3", "K-M5P4", "K-M5P5", "K-M5Q1", "K-M5Q2", "K-M5Q3",
"L1-L2L2", "L1-L2L3", "L1-L2M1", "L1-L2M2", "L1-L2M3", "L1-L2M4", "L1-L2M5", "L1-L2N1",
"L1-L2N2", "L1-L2N3", "L1-L2N4", "L1-L2N5", "L1-L2N6", "L1-L2N7", "L1-L2O1", "L1-L2O2",
"L1-L2O3", "L1-L2O4", "L1-L2O5", "L1-L2O6", "L1-L2O7", "L1-L2P1", "L1-L2P2", "L1-L2P3",
"L1-L2P4", "L1-L2P5", "L1-L2Q1", "L1-L2Q2", "L1-L2Q3", "L1-L3L2", "L1-L3L3", "L1-L3M1",
"L1-L3M2", "L1-L3M3", "L1-L3M4", "L1-L3M5", "L1-L3N1", "L1-L3N2", "L1-L3N3", "L1-L3N4",
"L1-L3N5", "L1-L3N6", "L1-L3N7", "L1-L3O1", "L1-L3O2", "L1-L3O3", "L1-L3O4", "L1-L3O5",
"L1-L3O6", "L1-L3O7", "L1-L3P1", "L1-L3P2", "L1-L3P3", "L1-L3P4", "L1-L3P5", "L1-L3Q1",
"L1-L3Q2", "L1-L3Q3", "L1-M1L2", "L1-M1L3", "L1-M1M1", "L1-M1M2", "L1-M1M3", "L1-M1M4",
"L1-M1M5", "L1-M1N1", "L1-M1N2", "L1-M1N3", "L1-M1N4", "L1-M1N5", "L1-M1N6", "L1-M1N7",
"L1-M1O1", "L1-M1O2", "L1-M1O3", "L1-M1O4", "L1-M1O5", "L1-M1O6", "L1-M1O7", "L1-M1P1",
"L1-M1P2", "L1-M1P3", "L1-M1P4", "L1-M1P5", "L1-M1Q1", "L1-M1Q2", "L1-M1Q3", "L1-M2L2",
"L1-M2L3", "L1-M2M1", "L1-M2M2", "L1-M2M3", "L1-M2M4", "L1-M2M5", "L1-M2N1", "L1-M2N2",
"L1-M2N3", "L1-M2N4", "L1-M2N5", "L1-M2N6", "L1-M2N7", "L1-M2O1", "L1-M2O2", "L1-M2O3",
"L1-M2O4", "L1-M2O5", "L1-M2O6", "L1-M2O7", "L1-M2P1", "L1-M2P2", "L1-M2P3", "L1-M2P4",
"L1-M2P5", "L1-M2Q1", "L1-M2Q2", "L1-M2Q3", "L1-M3L2", "L1-M3L3", "L1-M3M1", "L1-M3M2",
"L1-M3M3", "L1-M3M4", "L1-M3M5", "L1-M3N1", "L1-M3N2", "L1-M3N3", "L1-M3N4", "L1-M3N5",
"L1-M3N6", "L1-M3N7", "L1-M3O1", "L1-M3O2", "L1-M3O3", "L1-M3O4", "L1-M3O5", "L1-M3O6",
"L1-M3O7", "L1-M3P1", "L1-M3P2", "L1-M3P3", "L1-M3P4", "L1-M3P5", "L1-M3Q1", "L1-M3Q2",
"L1-M3Q3", "L1-M4L2", "L1-M4L3", "L1-M4M1", "L1-M4M2", "L1-M4M3", "L1-M4M4", "L1-M4M5",
"L1-M4N1", "L1-M4N2", "L1-M4N3", "L1-M4N4", "L1-M4N5", "L1-M4N6", "L1-M4N7", "L1-M4O1",
"L1-M4O2", "L1-M4O3", "L1-M4O4", "L1-M4O5", "L1-M4O6", "L1-M4O7", "L1-M4P1", "L1-M4P2",
"L1-M4P3", "L1-M4P4", "L1-M4P5", "L1-M4Q1", "L1-M4Q2", "L1-M4Q3", "L1-M5L2", "L1-M5L3",
"L1-M5M1", "L1-M5M2", "L1-M5M3", "L1-M5M4", "L1-M5M5", "L1-M5N1", "L1-M5N2", "L1-M5N3",
"L1-M5N4", "L1-M5N5", "L1-M5N6", "L1-M5N7", "L1-M5O1", "L1-M5O2", "L1-M5O3", "L1-M5O4",
"L1-M5O5", "L1-M5O6", "L1-M5O7", "L1-M5P1", "L1-M5P2", "L1-M5P3", "L1-M5P4", "L1-M5P5",
"L1-M5Q1", "L1-M5Q2", "L1-M5Q3", "L2-L3L3", "L2-L3M1", "L2-L3M2", "L2-L3M3", "L2-L3M4",
"L2-L3M5", "L2-L3N1", "L2-L3N2", "L2-L3N3", "L2-L3N4", "L2-L3N5", "L2-L3N6", "L2-L3N7",
"L2-L3O1", "L2-L3O2", "L2-L3O3", "L2-L3O4", "L2-L3O5", "L2-L3O6", "L2-L3O7", "L2-L3P1",
"L2-L3P2", "L2-L3P3", "L2-L3P4", "L2-L3P5", "L2-L3Q1", "L2-L3Q2", "L2-L3Q3", "L2-M1L3",
"L2-M1M1", "L2-M1M2", "L2-M1M3", "L2-M1M4", "L2-M1M5", "L2-M1N1", "L2-M1N2", "L2-M1N3",
"L2-M1N4", "L2-M1N5", "L2-M1N6", "L2-M1N7", "L2-M1O1", "L2-M1O2", "L2-M1O3", "L2-M1O4",
"L2-M1O5", "L2-M1O6", "L2-M1O7", "L2-M1P1", "L2-M1P2", "L2-M1P3", "L2-M1P4", "L2-M1P5",
"L2-M1Q1", "L2-M1Q2", "L2-M1Q3", "L2-M2L3", "L2-M2M1", "L2-M2M2", "L2-M2M3", "L2-M2M4",
"L2-M2M5", "L2-M2N1", "L2-M2N2", "L2-M2N3", "L2-M2N4", "L2-M2N5", "L2-M2N6", "L2-M2N7",
"L2-M2O1", "L2-M2O2", "L2-M2O3", "L2-M2O4", "L2-M2O5", "L2-M2O6", "L2-M2O7", "L2-M2P1",
"L2-M2P2", "L2-M2P3", "L2-M2P4", "L2-M2P5", "L2-M2Q1", "L2-M2Q2", "L2-M2Q3", "L2-M3L3",
"L2-M3M1", "L2-M3M2", "L2-M3M3", "L2-M3M4", "L2-M3M5", "L2-M3N1", "L2-M3N2", "L2-M3N3",
"L2-M3N4", "L2-M3N5", "L2-M3N6", "L2-M3N7", "L2-M3O1", "L2-M3O2", "L2-M3O3", "L2-M3O4",
"L2-M3O5", "L2-M3O6", "L2-M3O7", "L2-M3P1", "L2-M3P2", "L2-M3P3", "L2-M3P4", "L2-M3P5",
"L2-M3Q1", "L2-M3Q2", "L2-M3Q3", "L2-M4L3", "L2-M4M1", "L2-M4M2", "L2-M4M3", "L2-M4M4",
"L2-M4M5", "L2-M4N1", "L2-M4N2", "L2-M4N3", "L2-M4N4", "L2-M4N5", "L2-M4N6", "L2-M4N7",
"L2-M4O1", "L2-M4O2", "L2-M4O3", "L2-M4O4", "L2-M4O5", "L2-M4O6", "L2-M4O7", "L2-M4P1",
"L2-M4P2", "L2-M4P3", "L2-M4P4", "L2-M4P5", "L2-M4Q1", "L2-M4Q2", "L2-M4Q3", "L2-M5L3",
"L2-M5M1", "L2-M5M2", "L2-M5M3", "L2-M5M4", "L2-M5M5", "L2-M5N1", "L2-M5N2", "L2-M5N3",
"L2-M5N4", "L2-M5N5", "L2-M5N6", "L2-M5N7", "L2-M5O1", "L2-M5O2", "L2-M5O3", "L2-M5O4",
"L2-M5O5", "L2-M5O6", "L2-M5O7", "L2-M5P1", "L2-M5P2", "L2-M5P3", "L2-M5P4", "L2-M5P5",
"L2-M5Q1", "L2-M5Q2", "L2-M5Q3", "L3-M1M1", "L3-M1M2", "L3-M1M3", "L3-M1M4", "L3-M1M5",
"L3-M1N1", "L3-M1N2", "L3-M1N3", "L3-M1N4", "L3-M1N5", "L3-M1N6", "L3-M1N7", "L3-M1O1",
"L3-M1O2", "L3-M1O3", "L3-M1O4", "L3-M1O5", "L3-M1O6", "L3-M1O7", "L3-M1P1", "L3-M1P2",
"L3-M1P3", "L3-M1P4", "L3-M1P5", "L3-M1Q1", "L3-M1Q2", "L3-M1Q3", "L3-M2M1", "L3-M2M2",
"L3-M2M3", "L3-M2M4", "L3-M2M5", "L3-M2N1", "L3-M2N2", "L3-M2N3", "L3-M2N4", "L3-M2N5",
"L3-M2N6", "L3-M2N7", "L3-M2O1", "L3-M2O2", "L3-M2O3", "L3-M2O4", "L3-M2O5", "L3-M2O6",
"L3-M2O7", "L3-M2P1", "L3-M2P2", "L3-M2P3", "L3-M2P4", "L3-M2P5", "L3-M2Q1", "L3-M2Q2",
"L3-M2Q3", "L3-M3M1", "L3-M3M2", "L3-M3M3", "L3-M3M4", "L3-M3M5", "L3-M3N1", "L3-M3N2",
"L3-M3N3", "L3-M3N4", "L3-M3N5", "L3-M3N6", "L3-M3N7", "L3-M3O1", "L3-M3O2", "L3-M3O3",
"L3-M3O4", "L3-M3O5", "L3-M3O6", "L3-M3O7", "L3-M3P1", "L3-M3P2", "L3-M3P3", "L3-M3P4",
"L3-M3P5", "L3-M3Q1", "L3-M3Q2", "L3-M3Q3", "L3-M4M1", "L3-M4M2", "L3-M4M3", "L3-M4M4",
"L3-M4M5", "L3-M4N1", "L3-M4N2", "L3-M4N3", "L3-M4N4", "L3-M4N5", "L3-M4N6", "L3-M4N7",
"L3-M4O1", "L3-M4O2", "L3-M4O3", "L3-M4O4", "L3-M4O5", "L3-M4O6", "L3-M4O7", "L3-M4P1",
"L3-M4P2", "L3-M4P3", "L3-M4P4", "L3-M4P5", "L3-M4Q1", "L3-M4Q2", "L3-M4Q3", "L3-M5M1",
"L3-M5M2", "L3-M5M3", "L3-M5M4", "L3-M5M5", "L3-M5N1", "L3-M5N2", "L3-M5N3", "L3-M5N4",
"L3-M5N5", "L3-M5N6", "L3-M5N7", "L3-M5O1", "L3-M5O2", "L3-M5O3", "L3-M5O4", "L3-M5O5",
"L3-M5O6", "L3-M5O7", "L3-M5P1", "L3-M5P2", "L3-M5P3", "L3-M5P4", "L3-M5P5", "L3-M5Q1",
"L3-M5Q2", "L3-M5Q3", "M1-M2M2", "M1-M2M3", "M1-M2M4", "M1-M2M5", "M1-M2N1", "M1-M2N2",
"M1-M2N3", "M1-M2N4", "M1-M2N5", "M1-M2N6", "M1-M2N7", "M1-M2O1", "M1-M2O2", "M1-M2O3",
"M1-M2O4", "M1-M2O5", "M1-M2O6", "M1-M2O7", "M1-M2P1", "M1-M2P2", "M1-M2P3", "M1-M2P4",
"M1-M2P5", "M1-M2Q1", "M1-M2Q2", "M1-M2Q3", "M1-M3M2", "M1-M3M3", "M1-M3M4", "M1-M3M5",
"M1-M3N1", "M1-M3N2", "M1-M3N3", "M1-M3N4", "M1-M3N5", "M1-M3N6", "M1-M3N7", "M1-M3O1",
"M1-M3O2", "M1-M3O3", "M1-M3O4", "M1-M3O5", "M1-M3O6", "M1-M3O7", "M1-M3P1", "M1-M3P2",
"M1-M3P3", "M1-M3P4", "M1-M3P5", "M1-M3Q1", "M1-M3Q2", "M1-M3Q3", "M1-M4M2", "M1-M4M3",
"M1-M4M4", "M1-M4M5", "M1-M4N1", "M1-M4N2", "M1-M4N3", "M1-M4N4", "M1-M4N5", "M1-M4N6",
"M1-M4N7", "M1-M4O1", "M1-M4O2", "M1-M4O3", "M1-M4O4", "M1-M4O5", "M1-M4O6", "M1-M4O7",
"M1-M4P1", "M1-M4P2", "M1-M4P3", "M1-M4P4", "M1-M4P5", "M1-M4Q1", "M1-M4Q2", "M1-M4Q3",
"M1-M5M2", "M1-M5M3", "M1-M5M4", "M1-M5M5", "M1-M5N1", "M1-M5N2", "M1-M5N3", "M1-M5N4",
"M1-M5N5", "M1-M5N6", "M1-M5N7", "M1-M5O1", "M1-M5O2", "M1-M5O3", "M1-M5O4", "M1-M5O5",
"M1-M5O6", "M1-M5O7", "M1-M5P1", "M1-M5P2", "M1-M5P3", "M1-M5P4", "M1-M5P5", "M1-M5Q1",
"M1-M5Q2", "M1-M5Q3", "M2-M3M3", "M2-M3M4", "M2-M3M5", "M2-M3N1", "M2-M3N2", "M2-M3N3",
"M2-M3N4", "M2-M3N5", "M2-M3N6", "M2-M3N7", "M2-M3O1", "M2-M3O2", "M2-M3O3", "M2-M3O4",
"M2-M3O5", "M2-M3O6", "M2-M3O7", "M2-M3P1", "M2-M3P2", "M2-M3P3", "M2-M3P4", "M2-M3P5",
"M2-M3Q1", "M2-M3Q2", "M2-M3Q3", "M2-M4M3", "M2-M4M4", "M2-M4M5", "M2-M4N1", "M2-M4N2",
"M2-M4N3", "M2-M4N4", "M2-M4N5", "M2-M4N6", "M2-M4N7", "M2-M4O1", "M2-M4O2", "M2-M4O3",
"M2-M4O4", "M2-M4O5", "M2-M4O6", "M2-M4O7", "M2-M4P1", "M2-M4P2", "M2-M4P3", "M2-M4P4",
"M2-M4P5", "M2-M4Q1", "M2-M4Q2", "M2-M4Q3", "M2-M5M3", "M2-M5M4", "M2-M5M5", "M2-M5N1",
"M2-M5N2", "M2-M5N3", "M2-M5N4", "M2-M5N5", "M2-M5N6", "M2-M5N7", "M2-M5O1", "M2-M5O2",
"M2-M5O3", "M2-M5O4", "M2-M5O5", "M2-M5O6", "M2-M5O7", "M2-M5P1", "M2-M5P2", "M2-M5P3",
"M2-M5P4", "M2-M5P5", "M2-M5Q1", "M2-M5Q2", "M2-M5Q3", "M3-M4M4", "M3-M4M5", "M3-M4N1",
"M3-M4N2", "M3-M4N3", "M3-M4N4", "M3-M4N5", "M3-M4N6", "M3-M4N7", "M3-M4O1", "M3-M4O2",
"M3-M4O3", "M3-M4O4", "M3-M4O5", "M3-M4O6", "M3-M4O7", "M3-M4P1", "M3-M4P2", "M3-M4P3",
"M3-M4P4", "M3-M4P5", "M3-M4Q1", "M3-M4Q2", "M3-M4Q3", "M3-M5M4", "M3-M5M5", "M3-M5N1",
"M3-M5N2", "M3-M5N3", "M3-M5N4", "M3-M5N5", "M3-M5N6", "M3-M5N7", "M3-M5O1", "M3-M5O2",
"M3-M5O3", "M3-M5O4", "M3-M5O5", "M3-M5O6", "M3-M5O7", "M3-M5P1", "M3-M5P2", "M3-M5P3",
"M3-M5P4", "M3-M5P5", "M3-M5Q1", "M3-M5Q2", "M3-M5Q3", "M4-M5M5", "M4-M5N1", "M4-M5N2",
"M4-M5N3", "M4-M5N4", "M4-M5N5", "M4-M5N6", "M4-M5N7", "M4-M5O1", "M4-M5O2", "M4-M5O3",
"M4-M5O4", "M4-M5O5", "M4-M5O6", "M4-M5O7", "M4-M5P1", "M4-M5P2", "M4-M5P3", "M4-M5P4",
"M4-M5P5", "M4-M5Q1", "M4-M5Q2", "M4-M5Q3"};

char AugerNameTotal[][9] = {
"K-TOTAL",
"L1-TOTAL",
"L2-TOTAL",
"L3-TOTAL",
"M1-TOTAL",
"M2-TOTAL",
"M3-TOTAL",
"M4-TOTAL",
"M5-TOTAL",
};


/*////////////////////////////////////////////////////////////////////
/////            Functions                                       /////
//////////////////////////////////////////////////////////////////// */


int compareMendelElements(const void *i1, const void *i2) {
	struct MendelElement *ca1 = (struct MendelElement *) i1;
	struct MendelElement *ca2 = (struct MendelElement *) i2;

	return strcmp(ca1->name, ca2->name);
}

int compareCrystalStructs(const void *i1, const void *i2) {
	Crystal_Struct *ca1 = (Crystal_Struct *) i1;
	Crystal_Struct *ca2 = (Crystal_Struct *) i2;

	return strcmp(ca1->name, ca2->name);
}

int matchMendelElement(const void *i1, const void *i2) {
	char *ca1 = (char *) i1;
	struct MendelElement *ca2 = (struct MendelElement *) i2;

	return strcmp(ca1, ca2->name);
}

int matchCrystalStruct(const void *i1, const void *i2) {
	char *ca1 = (char *) i1;
	Crystal_Struct *ca2 = (Crystal_Struct *) i2;

	return strcmp(ca1, ca2->name);
}

void ErrorExit(char *error_message)
{
  if (ErrorMessages) {
  	printf("%s\n", error_message);
  }
  ExitStatus = 1;
  if (HardExit != 0) exit(EXIT_FAILURE);
}

void SetHardExit(int hard_exit)
{
  HardExit = hard_exit;
}

void SetExitStatus(int exit_status)
{
  ExitStatus = exit_status;
}

int GetExitStatus()
{
  return ExitStatus;
}

void SetErrorMessages(int status)
{
  ErrorMessages = status;
}

int GetErrorMessages(void)
{
  return ErrorMessages;
}


