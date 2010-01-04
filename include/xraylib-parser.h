


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
