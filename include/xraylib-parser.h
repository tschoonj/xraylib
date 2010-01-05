/*
Copyright (c) 2009, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/



#ifndef _XRAYLIB_PARSER_H
#define _XRAYLIB_PARSER_H

/*
 *
 * this header includes the prototype of a function designed to parse
 * chemical formulas and a structure definition which is used to store the information.
 *
 */


/*
 * A compoundData structure will be used to store the results after parsing:
 * 	nElements: the number of different atoms present
 * 	nAtomsAll: the total number of atoms present in the compound
 * 	Elements: an array with length nElements that will contain the atomic
 * 	  numbers of the different elements present in the compound, 
 * 	  in ascending order. The array memory will be allocated using malloc 
 * 	  and should be freed by the user when appropriate.
 *	massFractions: an array with length nElements that will contain
 *	  the atomic mass fractions of the different elements present in the compound,
 *	  in an order corresponding with the Elements array. The sum of the values
 *	  in this array is equal to 1. The array memory will be allocated 
 *	  using malloc and should be freed by the user when appropriate.
 *
 * For SiO2 this would yield a structure with contents:
 *  nElements: 2
 *  nAtomsAll: 3
 *  Elements: 8 14
 *  massFractions: 0.467465  0.532535
 *
 *
 *
 */

struct compoundData {
	int nElements;
	int nAtomsAll;
	int *Elements;
	double *massFractions;
};

/* 
 * Consider using the following macro to free the allocated memory 
 * in a compoundData structure
 */

#define FREE_COMPOUND_DATA(cd) free(cd.Elements);\
				free(cd.massFractions);

/*
 * The CompoundParser function will parse a string and will put the results in
 * a compoundData structure pointed to by cd. If successful, the function
 * returns 1, otherwise 0. The cd structure must point to a valid location in memory.
 */


int CompoundParser(char compoundString[], struct compoundData *cd);


#endif
