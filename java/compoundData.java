/*
Copyright (c) 2015, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Teemu Ikonen and Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Teemu Ikonen and Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

import java.util.ArrayList;
import java.util.Locale;
import java.util.Locale.Category;
import java.lang.Double;

public class compoundData {
  public int nElements;
  public double nAtomsAll;
  public int[] Elements;
  public double[] massFractions;  
  
  public compoundData(String compoundString) {
    ArrayList<compoundAtom> ca = new ArrayList<>();
   
    Locale old_locale = Locale.getDefault(Locale.Category.FORMAT); 
    Locale.setDefault(Locale.Category.FORMAT, Locale.US);

    try {
      CompoundParserSimple(compoundString, ca);
    }
    catch (XraylibException e) {
      throw new XraylibException(e.getMessage());
    }

    Locale.setDefault(Locale.Category.FORMAT, old_locale);
  }

  private static void CompoundParserSimple(String compoundString, ArrayList<compoundAtom> ca) {
    char[] csa = compoundString.toCharArray();
    int nbrackets=0;
    int nuppers=0;
    int i,j;
    ArrayList<Integer> upper_locs = new ArrayList<>();
    //char buffer[1024];
    ArrayList<Integer> brackets_begin_locs = new ArrayList<>();
    ArrayList<Integer> brackets_end_locs = new ArrayList<>();
    int nbracket_pairs=0;
    String tempElement;
    String tempSubstring;
    double tempnAtoms;
    int res;
    compoundAtom res2;
    ArrayList<compoundAtom> tempBracketAtoms;
    String tempBracketString;
    int ndots;
    //char *endPtr;
  
    if (Character.isLowerCase(csa[0]) || Character.isDigit(csa[0])) {
      throw new XraylibException("invalid chemical formula. Found a lowercase character or digit where not allowed");
    }

    for (i = 0 ; i < csa.length ; i++) {
      if (csa[i] == '(') {
        nbrackets++;
        if (nbrackets == 1) {
          nbracket_pairs++;
          brackets_begin_locs.add(i);
        }
      }
      else if (csa[i] == ')') {
        nbrackets--;
        if (nbrackets == 0) {
          brackets_end_locs.add(i);
        }
      }
      else if (nbrackets > 0) {}
      else if (nbrackets == 0 && Character.isUpperCase(csa[i])) {
        nuppers++;
        upper_locs.add(i);
      }
      else if (csa[i] == ' '){
        throw new XraylibException("spaces are not allowed in compound formula");
      }
      else if (Character.isLowerCase(csa[i]) || 
               Character.isDigit(csa[i]) || csa[i] == '.') {}
      else {
        throw new XraylibException("invalid character detected " + csa[i]);
      }

      if (nbrackets < 0) {
        throw new XraylibException("brackets not matching");
      }
    }

    if (nuppers == 0 && nbracket_pairs == 0) {
      throw new XraylibException("chemical formula contains no elements");
    }

    if (nbrackets > 0) {
      throw new XraylibException("brackets not matching");
    }

    /*parse locally*/
    for (i = 0 ; i < nuppers ; i++) {
      if (Character.isLowerCase(csa[upper_locs.get(i)+1]) &&
          !Character.isLowerCase(csa[upper_locs.get(i)+2])) {
        /*second letter is lowercase and third one isn't -> valid */
        tempElement = compoundString.substring(upper_locs.get(i), upper_locs.get(i)+2);
        /*get corresponding atomic number */
        try {
          res = Xraylib.SymbolToAtomicNumber(tempElement);
        }
        catch (XraylibException e) {
          throw new XraylibException("invalid element " + tempElement + " in chemical formula");
        }
        /*determine element subscript */
        j=2;
        ndots=0;

        while (Character.isDigit(csa[upper_locs.get(i)+j]) ||
               csa[upper_locs.get(i)+j] == '.') {
          j++;
          if (csa[upper_locs.get(i)+j] == '.') {
            ndots++;
          }
        }
        if (ndots > 1) {
          throw new XraylibException("only one dot allowed in subscripts of the chemical formula");
        }
        if (j==2) {
          tempnAtoms = 1.0;				
        }
        else {
          tempSubstring = compoundString.substring(upper_locs.get(i)+2, upper_locs.get(i)+2+j-2);
          try {
            tempnAtoms = Double.parseDouble(tempSubstring);
          }
          catch (NumberFormatException e) {
            throw new XraylibException("error converting subscript "+ tempSubstring +" of the chemical formula to a real number");
          }

          /*zero subscript is not allowed */
          if (tempnAtoms == 0.0) {
            throw new XraylibException("zero subscript detected in chemical formula");
          }
        }
      }	
      else if (!Character.isLowerCase(csa[upper_locs.get(i)+1])) {
        /*second letter is not lowercase -> valid */
        tempElement = compoundString.substring(upper_locs.get(i), upper_locs.get(i)+1);
        /*get corresponding atomic number */
        try {
          res = Xraylib.SymbolToAtomicNumber(tempElement);
        }
        catch (XraylibException e) {
          throw new XraylibException("invalid element " + tempElement + " in chemical formula");
        }
        /*determine element subscript */
        j=1;
        ndots=0;
        while (Character.isDigit(csa[upper_locs.get(i)+j]) ||
               csa[upper_locs.get(i)+j] == '.') {
          j++;
          if (csa[upper_locs.get(i)+j] == '.') {
            ndots++;
          }
        }
        if (ndots > 1) {
	  throw new XraylibException("only one dot allowed in subscripts of the chemical formula");
        }
        if (j==1) {
          tempnAtoms = 1.0;	
        }
        else {
          tempSubstring = compoundString.substring(upper_locs.get(i)+1, upper_locs.get(i)+1+j-1);
          try {
            tempnAtoms = Double.parseDouble(tempSubstring);
          }
          catch (NumberFormatException e) {
            throw new XraylibException("error converting subscript "+ tempSubstring +" of the chemical formula to a real number");
          }

          /*zero subscript is not allowed */
          if (tempnAtoms == 0.0) {
            throw new XraylibException("zero subscript detected in chemical formula");
          }
        }
      }
      else {
        throw new XraylibException("invalid chemical formula");
      }

      /*atomic number identification ok -> add it to the array if necessary */
      /*check if current element is already present in the array */
      try {
        res2 = ca.get(ca.indexOf(new compoundAtom(res, 0.0)));
        /*element is in array -> update it */
        res2.nAtoms += tempnAtoms;
      }
      catch (IndexOutOfBoundsException e) {
        /*element not in array -> add it */
        ca.add(new compoundAtom(res, tempnAtoms));
      }
    } 

    /*handle the brackets... */
    for (i = 0 ; i < nbracket_pairs ; i++) {
      tempBracketAtoms = new ArrayList<>();
      tempBracketString = compoundString.substring(brackets_begin_locs.get(i)+1, brackets_end_locs.get(i));
      /*recursive call */
      CompoundParserSimple(tempBracketString,tempBracketAtoms);
		
      /*check if the brackets pair is followed by a subscript */
      j=1;
      ndots=0;
      while (isDigit(csa[brackets_end_locs.get(i)+j]) || csa[brackets_end_locs.get(i)+j] == '.') {
        j++;
        if (csa[brackets_end_locs.get(i)+j] == '.') {
          ndots++;
        }
      }
      if (ndots > 1) {
        throw new XraylibException("only one dot allowed in subscripts of the chemical formula");
      }
      if (j==1) {
        tempnAtoms = 1.0;				
      }
      else {
        tempSubstring = compoundString.substring(upper_locs.get(i)+1, upper_locs.get(i)+1+j-1);
        try {
          tempnAtoms = Double.parseDouble(tempSubstring);
        }
        catch (NumberFormatException e) {
          throw new XraylibException("error converting subscript "+ tempSubstring +" of the chemical formula to a real number");
        }

        /*zero subscript is not allowed */
        if (tempnAtoms == 0.0) {
          throw new XraylibException("zero subscript detected in chemical formula");
        }
      }

		/*add them to the array... */
		if (ca->nElements == 0) {
			/*array is empty */
			ca->nElements = tempBracketAtoms->nElements;
			ca->singleElements = tempBracketAtoms->singleElements;
			for (j = 0 ; j < ca->nElements ; j++) 
				ca->singleElements[j].nAtoms *= tempnAtoms;
		}
		else {
			for (j = 0 ; j < tempBracketAtoms->nElements ; j++) {
				key2.Element = tempBracketAtoms->singleElements[j].Element;
				res2 = bsearch(&key2,ca->singleElements,ca->nElements,sizeof(struct compoundAtom),compareCompoundAtoms);
				if (res2 == NULL) {
					/*element not in array -> add it */
					ca->singleElements = (struct compoundAtom *) realloc((struct compoundAtom *) ca->singleElements,(++ca->nElements)*sizeof(struct compoundAtom));
					ca->singleElements[ca->nElements-1].Element = key2.Element; 
					ca->singleElements[ca->nElements-1].nAtoms = tempBracketAtoms->singleElements[j].nAtoms*tempnAtoms; 
					/*sort array */
					qsort(ca->singleElements,ca->nElements,sizeof(struct compoundAtom), compareCompoundAtoms);
				}
				else {
					/*element is in array -> update it */
					res2->nAtoms +=tempBracketAtoms->singleElements[j].nAtoms*tempnAtoms;
				}
			}
			free(tempBracketAtoms->singleElements);
			free(tempBracketAtoms);
		}
	} 	


  }

  static class compoundAtom {
    int Element;
    double nAtoms;
    public compoundAtom(int Element, double nAtoms) {
      this.Element = Element;
      this.nAtoms = nAtoms;
    } 
    @Override
    public boolean equals(Object object2) {
      if (object2 == null) {
        return false;
      }
      else if (object2 instanceof compoundAtom &&
          this.Element == ((compoundAtom) object2).Element) {
        return true;
      }
      else if (object2 instanceof Integer &&
          this.Element == (Integer) object2) {
        return true;
      }

      return false; 
    }
  } 

  private static final String[] MendelArray = { "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh"
  };

}
