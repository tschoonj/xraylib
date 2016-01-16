/*
Copyright (c) 2011, 2012  David Sagan and Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY David Sagan, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef XRAYLIB_DEFS_H
#define XRAYLIB_DEFS_H



#define ZMAX 120
#define MENDEL_MAX 107
#define CRYSTALARRAY_MAX 100
#define MAXFILENAMESIZE 1000
#define SHELLNUM 28
#define SHELLNUM_K 31
#define SHELLNUM_C 29
#define LINENUM 383
#define TRANSNUM 15
#define AUGERNUM 996
#define SHELLNUM_A 9
/* Delta for size increase of Crystal_Array.crystal array */
#define N_NEW_CRYSTAL 10


/* Structs */

/* Complex number */

typedef struct {
  double re;               /* Real part */
  double im;               /* Imaginary part */
} xrlComplex;
#ifndef c_abs
/* this is giving a lot of trouble with python */
double c_abs(xrlComplex x);
#endif
xrlComplex c_mul(xrlComplex x, xrlComplex y);

/* Struct for an atom in a crystal. */

typedef struct {
  int Zatom;              /* Atomic number of atom. */
  double fraction;         /* Fractional contribution. Normally 1.0. */
  double x, y, z;          /* Atom position in fractions of the unit cell lengths. */
} Crystal_Atom;

/* Struct for a crystal. */

typedef struct {
  char* name;                 /* Name of crystal. */
  double a, b, c;              /* Unit cell size in Angstroms. */
  double alpha, beta, gamma;   /* Unit cell angles in degrees. */
  double volume;               /* Unit cell volume in Angstroms^3. */
  int n_atom;                 /* Number of atoms. */
  Crystal_Atom* atom;   /* Array of atoms in unit cell. */
} Crystal_Struct;

/* Container struct to hold an array of CrystalStructs */

typedef struct {
  int n_crystal;          /* Number of defined crystals. */
  int n_alloc;            /* Size of .crystal array malloc'd */
  Crystal_Struct* crystal;
} Crystal_Array;

#endif
