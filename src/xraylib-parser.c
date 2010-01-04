#include "xraylib.h"
#include "xraylib-aux.h"
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>



struct MendeljevElement {
	int number;
	char *name;
};

struct MendeljevElement MendeljevArray[] = {
	{1,"H"},{2,"He"},{3,"Li"},{4,"Be"},{5,"B"},{6,"C"},{7,"N"},{8,"O"},{9,"F"},{10,"Ne"},
	{11,"Na"},{12,"Mg"},{13,"Al"},{14,"Si"},{15,"P"},{16,"S"},{17,"Cl"},{18,"Ar"},{19,"K"},{20,"Ca"},
	{21,"Sc"},{22,"Ti"},{23,"V"},{24,"Cr"},{25,"Mn"},{26,"Fe"},{27,"Co"},{28,"Ni"},{29,"Cu"},{30,"Zn"},
	{31,"Ga"},{32,"Ge"},{33,"As"},{34,"Se"},{35,"Br"},{36,"Kr"},{37,"Rb"},{38,"Sr"},{39,"Y"},{40,"Zr"},
	{41,"Nb"},{42,"Mo"},{43,"Tc"},{44,"Ru"},{45,"Rh"},{46,"Pd"},{47,"Ag"},{48,"Cd"},{49,"In"},{50,"Sn"},
	{51,"Sb"},{52,"Te"},{53,"I"},{54,"Xe"},{55,"Cs"},{56,"Ba"},{57,"La"},{58,"Ce"},{59,"Pr"},{60,"Nd"},
	{61,"Pm"},{62,"Sm"},{63,"Eu"},{64,"Gd"},{65,"Tb"},{66,"Dy"},{67,"Ho"},{68,"Er"},{69,"Tm"},{70,"Yb"},
	{71,"Lu"},{72,"Hf"},{73,"Ta"},{74,"W"},{75,"Re"},{76,"Os"},{77,"Ir"},{78,"Pt"},{79,"Au"},{80,"Hg"},
	{81,"Tl"},{82,"Pb"},{83,"Bi"},{84,"Po"},{85,"At"},{86,"Rn"},{87,"Fr"},{88,"Ra"},{89,"Ac"},{90,"Th"},
	{91,"Pa"},{92,"U"},{93,"Np"},{94,"Pu"},{95,"Am"},{96,"Cm"},{97,"Bk"},{98,"Cf"},{99,"Es"},{100,"Fm"},
	{101,"Md"},{102,"No"},{103,"Lr"},{104,"Rf"},{105,"Db"},{106,"Sg"},{107,"Bh"}
	};


struct compoundAtom {
	int Element;
	int nAtoms;
};

struct compoundAtoms {
	int nElements;
	struct compoundAtom *singleElements;
};

static int compareCompoundAtoms(const void *i1, const void *i2) {
	struct compoundAtom *ca1 = (struct compoundAtom *) i1;
	struct compoundAtom *ca2 = (struct compoundAtom *) i2;

	return (ca1->Element - ca2->Element);
}

static int compareMendeljevElements(const void *i1, const void *i2) {
	struct MendeljevElement *ca1 = (struct MendeljevElement *) i1;
	struct MendeljevElement *ca2 = (struct MendeljevElement *) i2;

	return strcmp(ca1->name,ca2->name);
}

int CompoundParserSimple(char compoundString[], struct compoundAtoms *ca) {

	int nbrackets=0;
	int nuppers=0;
	int i,j;
	char **upper_locs = NULL;
	char buffer[1024];
	char **brackets_begin_locs=NULL;
	char **brackets_end_locs=NULL;
	int nbracket_pairs=0;

	for (i = 0 ; compoundString[i] != '\0' ; i++) {
		fprintf(stdout,"%c",compoundString[i]);
		if (compoundString[i] == '(') {
			fprintf(stdout,"joske\n");
			nbrackets++;
			if (nbrackets == 1) {
				brackets_begin_locs = (char **) realloc((char **) brackets_begin_locs,sizeof(char *)*++nbracket_pairs);
				brackets_begin_locs[nbracket_pairs-1] = compoundString+i;
			}
		}
		else if (compoundString[i] == ')') {
			fprintf(stdout,"joske2\n");
			nbrackets--;
			if (nbrackets == 0) {
				brackets_end_locs = (char **) realloc((char **) brackets_end_locs,sizeof(char *)*nbracket_pairs);
				brackets_end_locs[nbracket_pairs-1] = compoundString+i;
			}
		}
		else if (nbrackets > 0) {}
		else if (nbrackets == 0 && isupper(compoundString[i])) {
			upper_locs =(char **) realloc((char **) upper_locs,sizeof(char *)*++nuppers);	
			upper_locs[nuppers-1] = compoundString+i;
		}
		else if (compoundString[i] == ' '){
			sprintf(buffer,"xraylib-parser: spaces are not allowed in compound formula\n");
		//	ErrorExit(buffer);
			fprintf(stdout,buffer);
			return 0;
		}
		else if (islower(compoundString[i]) || isdigit(compoundString[i])) {}
		else {
			sprintf(buffer,"xraylib-parser: invalid character detected %c\n",compoundString[i]);
//			ErrorExit(buffer);
			fprintf(stdout,buffer);
			return 0;
		}

		if (nbrackets < 0) {
			sprintf(buffer,"xraylib-parser: brackets not matching\n");
		//	ErrorExit(buffer);
			fprintf(stdout,buffer);
			return 0;
		}

	}
	if (nuppers == 0 && nbracket_pairs == 0) {
		sprintf(buffer,"xraylib-parser: Chemical formula contains no elements\n");
	//	ErrorExit(buffer);
		fprintf(stdout,buffer);
		return 0;
	}
	if (nbrackets > 0) {
		sprintf(buffer,"xraylib-parser: brackets not matching\n");
		//ErrorExit(buffer);
		fprintf(stdout,buffer);
		return 0;
	}

	char *tempElement;
	char *tempSubstring;
	int tempnAtoms;
	struct MendeljevElement *res,key;
	struct compoundAtom *res2,key2;
	//parse locally
	for (i = 0 ; i < nuppers ; i++) {
		if (islower(upper_locs[i][1]) && !islower(upper_locs[i][2])) {
			fprintf(stdout,"case1\n");
			//second letter is lowercase and third one isn't -> valid
			tempElement = strndup(upper_locs[i],2);
			//get corresponding atomic number
			key.name = tempElement;	
			res = bsearch(&key,MendeljevArray,107,sizeof(struct MendeljevElement),compareMendeljevElements);
			if (res == NULL) {
				sprintf(buffer,"xraylib-parser: invalid element %s in chemical formula\n",tempElement);
			//	ErrorExit(buffer);
			fprintf(stdout,buffer);
				return 0;	
			}
			//determine element subscript
			j=2;
			while (isdigit(upper_locs[i][j])) {
				j++;
			}
			if (j==2) {
				tempnAtoms = 1;				
			}
			else {
				tempSubstring = strndup(upper_locs[i]+2,j-2);
				tempnAtoms = (int) strtol(tempSubstring,NULL,10);
				free(tempSubstring);
			}
			free(tempElement);
		}	
		else if (!islower(upper_locs[i][1])) {
			fprintf(stdout,"case2\n");
			//second letter is not lowercase -> valid
			tempElement = strndup(upper_locs[i],1);
			fprintf(stdout,"tempElement: %s\n",tempElement);
			//get corresponding atomic number
			key.name = tempElement;	
			res = bsearch(&key,MendeljevArray,107,sizeof(struct MendeljevElement),compareMendeljevElements);
			if (res == NULL) {
				sprintf(buffer,"xraylib-parser: invalid element %s in chemical formula\n",tempElement);
			//	ErrorExit(buffer);
			fprintf(stdout,buffer);
				return 0;	
			}
			//determine element subscript
			j=1;
			while (isdigit(upper_locs[i][j])) {
				j++;
			}
			if (j==1) {
				tempnAtoms = 1;				
			}
			else {
				tempSubstring = strndup(upper_locs[i]+1,j-1);
				tempnAtoms = (int) strtol(tempSubstring,NULL,10);
				free(tempSubstring);
			}
			free(tempElement);
		}
		else {
			//error
			sprintf(buffer,"xraylib-parser: invalid chemical formula\n");
		//	ErrorExit(buffer);
			fprintf(stdout,buffer);
			return 0;	
		}
		//atomic number identification ok -> add it to the array if necessary
		if (ca->nElements == 0) {
			//array is empty
			ca->singleElements = (struct compoundAtom *) malloc(sizeof(struct compoundAtom));
			ca->singleElements[0].Element = res->number;
			ca->singleElements[0].nAtoms = tempnAtoms;
			ca->nElements++;
		}
		else {
			//array is not empty
			//check if current element is already present in the array
			key2.Element = res->number;
			res2 = bsearch(&key2,ca->singleElements,ca->nElements,sizeof(struct compoundAtom),compareCompoundAtoms);
			if (res2 == NULL) {
				//element not in array -> add it
				ca->singleElements = (struct compoundAtom *) realloc((struct compoundAtom *) ca->singleElements,(++ca->nElements)*sizeof(struct compoundAtom));
				ca->singleElements[ca->nElements-1].Element = res->number; 
				ca->singleElements[ca->nElements-1].nAtoms = tempnAtoms; 
				//sort array
				qsort(ca->singleElements,ca->nElements,sizeof(struct compoundAtom), compareCompoundAtoms);
			}
			else {
				//element is in array -> update it
				res2->nAtoms += tempnAtoms;
			}
		}
	} 
	if (nuppers > 0)
		free(upper_locs);

	//handle the brackets...
	struct compoundAtoms *tempBracketAtoms;
	char *tempBracketString;

	for (i = 0 ; i < nbracket_pairs ; i++) {
		tempBracketAtoms = (struct compoundAtoms *) malloc(sizeof(struct compoundAtoms));
		tempBracketString = strndup(brackets_begin_locs[i]+1,(size_t) (brackets_end_locs[i]-brackets_begin_locs[i]-1));
		fprintf(stdout,"tempBracketString: %s\n",tempBracketString);
		tempBracketAtoms->nElements = 0;
		tempBracketAtoms->singleElements = NULL;
		//recursive call
		if (CompoundParserSimple(tempBracketString,tempBracketAtoms) == 0) {
			return 0;
		}
		free(tempBracketString);
		//check if the brackets pair is followed by a subscript
		j=1;
		while (isdigit(brackets_end_locs[i][j])) {
			j++;
		}
		if (j==1) {
			tempnAtoms = 1;				
		}
		else {
			tempSubstring = strndup(brackets_end_locs[i]+1,j-1);
			tempnAtoms = (int) strtol(tempSubstring,NULL,10);
			free(tempSubstring);
		}

		//add them to the array...
		if (ca->nElements == 0) {
			//array is empty
			ca->nElements = tempBracketAtoms->nElements;
			ca->singleElements = tempBracketAtoms->singleElements;
			if (tempnAtoms > 1)
				for (j = 0 ; j < ca->nElements ; j++) 
					ca->singleElements[j].nAtoms *= tempnAtoms;
		}
		else {
			for (j = 0 ; j < tempBracketAtoms->nElements ; j++) {
				key2.Element = tempBracketAtoms->singleElements[j].Element;
				res2 = bsearch(&key2,ca->singleElements,ca->nElements,sizeof(struct compoundAtom),compareCompoundAtoms);
				if (res2 == NULL) {
					//element not in array -> add it
					ca->singleElements = (struct compoundAtom *) realloc((struct compoundAtom *) ca->singleElements,(++ca->nElements)*sizeof(struct compoundAtom));
					ca->singleElements[ca->nElements-1].Element = key2.Element; 
					ca->singleElements[ca->nElements-1].nAtoms = tempBracketAtoms->singleElements[j].nAtoms*tempnAtoms; 
					//sort array
					qsort(ca->singleElements,ca->nElements,sizeof(struct compoundAtom), compareCompoundAtoms);
				}
				else {
					//element is in array -> update it
					res2->nAtoms +=tempBracketAtoms->singleElements[j].nAtoms*tempnAtoms;
				}
			}
			free(tempBracketAtoms->singleElements);
			free(tempBracketAtoms);
		}
	} 	
	if (nbracket_pairs > 0) {
		free(brackets_begin_locs);
		free(brackets_end_locs);
	}



	return 1;
}




int CompoundParser(char compoundString[], struct compoundData *cd) {
	struct compoundAtoms ca = {0,NULL};
	int rvCPS,i;
	double sum = 0.0;

	//sort MendeljevArray
	qsort(MendeljevArray,107,sizeof(struct MendeljevElement),compareMendeljevElements);

	rvCPS=CompoundParserSimple(compoundString,&ca);

	fprintf(stdout,"rvCPS: %i\n",rvCPS);
	if (rvCPS) {
		cd->nElements = ca.nElements;
		cd->nAtomsAll = 0;
		cd->Elements = (int *) malloc(sizeof(int)*ca.nElements);
		cd->massFractions = (double *) malloc(sizeof(double)*ca.nElements);
		for (i = 0 ; i < ca.nElements ; i++) {
			sum += AtomicWeight(ca.singleElements[i].Element)*ca.singleElements[i].nAtoms;	
			cd->nAtomsAll += ca.singleElements[i].nAtoms;
		}
		for (i = 0 ; i < ca.nElements ; i++) {
			cd->Elements[i] = ca.singleElements[i].Element;
			cd->massFractions[i] = AtomicWeight(ca.singleElements[i].Element)*ca.singleElements[i].nAtoms/sum;
		}
		fprintf(stdout,"nElements: %i\n",ca.nElements);
		for (i = 0 ; i < ca.nElements ; i++) 
			fprintf(stdout,"Element: %i   nAtoms: %i  massFraction: %lf\n",cd->Elements[i],cd->singleElements[i].nAtoms,cd->massFractions[i]);
			
		return 1;
	}
	else
		return 0;
}

int main (int argc, char *argv[]) {
	char testCompound[] = "SiO23(Fe2S(H2O)4)2Fe3";
	struct compoundData cd;

	if (argc > 1)	
		CompoundParser(argv[1],&cd);

	return 0;
}


