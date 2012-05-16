/*
Copyright (c) 2011  David Sagan
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ANYONE BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef XRAY_DEFS
#define XRAY_DEFS

#include <stdlib.h>

#define FALSE 0
#define TRUE 1

#define SUCCESS 0
#define FAILURE 1

#define ZMAX 120
#define MENDEL_MAX 107
#define CRYSTALARRAY_MAX 100
#define MAXFILENAMESIZE 1000
#define SHELLNUM 28
#define SHELLNUM_K 31
#define SHELLNUM_C 29
#define LINENUM 383
#define TRANSNUM 15
#define AUGERNUM 204
#define SHELLNUM_A 9
#define N_NEW_CRYSTAL 10   // Delta for size increase of Crystal_Array.crystal array

#define Crystal_Struct struct CrystalStruct 
#define Crystal_Atom   struct CrystalAtom 
#define Crystal_Array  struct CrystalArray

// Structs

// Complex number

typedef struct {
  float re;               // Real part
  float im;               // Imaginary part.
} Complex;

float c_abs(Complex x);
Complex c_mul(Complex x, Complex y);

// Struct to hold info on a particular type of atom.

struct MendelElement {
  int Zatom;              // Atomic number of atom.
  char *name;             // Name of atom.
};

// Struct for an atom in a crystal.

struct CrystalAtom {
  int Zatom;              // Atomic number of atom.
  float fraction;         // Fractional contribution. Normally 1.0.
  float x, y, z;          // Atom position in fractions of the unit cell lengths.
};

// Struct for a crystal.

struct CrystalStruct {
  char* name;                 // Name of crystal.
  float a, b, c;              // Unit cell size in Angstroms.
  float alpha, beta, gamma;   // Unit cell angles in degrees.
  float volume;               // Unit cell volume in Angstroms^3.
  int n_atom;                 // Number of atoms.
  struct CrystalAtom* atom;   // Array of atoms in unit cell.
};

// Container struct to hold an array of CrystalStructs

struct CrystalArray {
  int n_crystal;          // Number of defined crystals.
  int n_alloc;            // Size of .crystal array malloc'd
  Crystal_Struct* crystal;
};

#endif
