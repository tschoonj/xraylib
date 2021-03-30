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

package com.github.tschoonj.xraylib;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Locale;

/** 
 *  
 * 
 * @author Tom Schoonjans (Tom.Schoonjans@diamond.ac.uk)
 * @since 3.2.0
 */
public class compoundData implements compoundDataBase {
  public final int nElements;
  public final int[] Elements;
  public final double[] massFractions;
  public final double nAtomsAll;
  public final double[] nAtoms;
  public final double molarMass;
  private final String formattedCompoundString;
 
  @Override
  public String toString() {
    return formattedCompoundString;
  }
 
  /** 
   * Constructor for compoundData instances
   *
   * Use @see Xraylib#CompoundParser instead, if you prefer to stay closer to the C implementation.
   * 
   * @param  compoundString the chemical formula that will be parsed
   */
  public compoundData(String compoundString) {
    ArrayList<compoundAtom> ca = new ArrayList<>();
   
    Locale old_locale = Locale.getDefault(Locale.Category.FORMAT); 
    Locale.setDefault(Locale.Category.FORMAT, Locale.US);

    try {
      CompoundParserSimple(compoundString, ca);
      Collections.sort(ca, new compoundAtomComparator());
      nElements = ca.size();
      double nAtomsAll= 0.0;
      Elements = new int[nElements];
      massFractions = new double[nElements];
      nAtoms = new double[nElements];

      double sum = 0.0;

      for (compoundAtom cas : ca) {
        sum += Xraylib.AtomicWeight(cas.Element)*cas.nAtoms;	
        nAtomsAll+= cas.nAtoms; 
      }

      String formattedCompoundString = String.format("%s contains %g atoms, %d elements and has a molar mass of %g", compoundString, nAtomsAll, nElements, sum);

      for (int i = 0 ; i < nElements ; i++) {
        Elements[i] = ca.get(i).Element;
        massFractions[i] = Xraylib.AtomicWeight(ca.get(i).Element)*ca.get(i).nAtoms/sum;
	nAtoms[i] = ca.get(i).nAtoms;
        formattedCompoundString += String.format("\nElement %d: %f %% and %g atoms", Elements[i], massFractions[i]*100.0, nAtoms[i]);
      }
      this.nAtomsAll = nAtomsAll;
      this.molarMass = sum;
      this.formattedCompoundString = formattedCompoundString;
    }
    catch (Exception e) {
      throw e;
    }
    finally {
      Locale.setDefault(Locale.Category.FORMAT, old_locale);
    }
  }

  private static void CompoundParserSimple(String compoundString, ArrayList<compoundAtom> ca) {
    //the %% is added to allow checking characters after the end of the compoundString
    //the C-version relies on the null char for this
    char[] csa = (compoundString+"%%").toCharArray();
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
      throw new IllegalArgumentException("Invalid chemical formula: Found a lowercase character or digit where not allowed");
    }

    // the -2 is necessary to avoid evaluating the %% in the compoundString
    for (i = 0 ; i < csa.length-2 ; i++) {
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
      else if (nbrackets > 0) {

      }
      else if (Character.isUpperCase(csa[i])) {
        nuppers++;
        upper_locs.add(i);
      }
      else if (csa[i] == ' '){
        throw new IllegalArgumentException("Invalid chemical formula: Spaces are not allowed in compound formula");
      }
      else if (i > 0 && Character.isLowerCase(csa[i]) && Character.isDigit(csa[i-1])) {
	throw new IllegalArgumentException("Invalid chemical formula: Found a lowercase character where not allowed");
      }
      else if (Character.isLowerCase(csa[i]) || Character.isDigit(csa[i]) || csa[i] == '.') {

      }
      else {
        throw new IllegalArgumentException(String.format("Invalid chemical formula: Invalid character %c detected", csa[i]));
      }

      if (nbrackets < 0) {
        throw new IllegalArgumentException("Invalid chemical formula: Brackets not matching");
      }
    }

    if (nuppers == 0 && nbracket_pairs == 0) {
      throw new IllegalArgumentException("Invalid chemical formula: No elements found");
    }

    if (nbrackets > 0) {
      throw new IllegalArgumentException("Invalid chemical formula: Backets not matching");
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
        catch (IllegalArgumentException e) {
          throw new IllegalArgumentException(String.format("Invalid chemical formula: unknown symbol %s detected", tempElement));
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
          throw new IllegalArgumentException("Invalid chemical formula: only one dot allowed in subscripts of the chemical formula");
        }
        if (j == 2) {
          tempnAtoms = 1.0;				
        }
        else {
          tempSubstring = compoundString.substring(upper_locs.get(i)+2, upper_locs.get(i)+2+j-2);
          try {
            tempnAtoms = Double.parseDouble(tempSubstring);
          }
          catch (NumberFormatException e) {
            throw new IllegalArgumentException(String.format("Invalid chemical formula: could not convert subscript %s to a real number", tempSubstring));
          }

          /*zero subscript is not allowed */
          if (tempnAtoms == 0.0) {
            throw new IllegalArgumentException("Invalid chemical formula: zero subscript detected");
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
        catch (IllegalArgumentException e) {
          throw new IllegalArgumentException(String.format("Invalid chemical formula: unknown symbol %s detected", tempElement));
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
          throw new IllegalArgumentException("Invalid chemical formula: only one dot allowed in subscripts of the chemical formula");
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
            throw new IllegalArgumentException(String.format("Invalid chemical formula: could not convert subscript %s to a real number", tempSubstring));
          }

          /*zero subscript is not allowed */
          if (tempnAtoms == 0.0) {
            throw new IllegalArgumentException("Invalid chemical formula: zero subscript detected");
          }
        }
      }
      else {
        throw new IllegalArgumentException("Invalid chemical formula");
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
      while (Character.isDigit(csa[brackets_end_locs.get(i)+j]) ||
             csa[brackets_end_locs.get(i)+j] == '.') {
        j++;
        if (csa[brackets_end_locs.get(i)+j] == '.') {
          ndots++;
        }
      }
      if (ndots > 1) {
        throw new IllegalArgumentException("Invalid chemical formula: only one dot allowed in subscripts of the chemical formula");
      }
      if (j==1) {
        tempnAtoms = 1.0;				
      }
      else {
        tempSubstring = compoundString.substring(brackets_end_locs.get(i)+1, brackets_end_locs.get(i)+1+j-1);
        try {
          tempnAtoms = Double.parseDouble(tempSubstring);
        }
        catch (NumberFormatException e) {
          throw new IllegalArgumentException(String.format("Invalid chemical formula: could not convert subscript %s to a real number", tempSubstring));
        }

        /*zero subscript is not allowed */
        if (tempnAtoms == 0.0) {
          throw new IllegalArgumentException("Invalid chemical formula: zero subscript detected");
        }
      }

      /*add them to the array... */
      for (j = 0 ; j < tempBracketAtoms.size() ; j++) {
        try {
          res2 = ca.get(ca.indexOf(new compoundAtom(tempBracketAtoms.get(j).Element, 0.0)));
          res2.nAtoms += tempBracketAtoms.get(j).nAtoms*tempnAtoms;
        }
        catch (IndexOutOfBoundsException e) {
          /*element not in array -> add it */
          ca.add(new compoundAtom(tempBracketAtoms.get(j).Element, tempBracketAtoms.get(j).nAtoms*tempnAtoms));
        }
      }
    }
  }

  static class compoundAtomComparator implements Comparator<compoundAtom> {
    @Override
    public int compare(compoundAtom ca1, compoundAtom ca2) {
      return ca1.Element - ca2.Element;
    }
  }

  static class compoundAtom  {
    int Element;
    double nAtoms;
    public compoundAtom(int Element, double nAtoms) {
      this.Element = Element;
      this.nAtoms = nAtoms;
    }
  }

  @Override
  public int getNElements() {
    return nElements;
  }

  @Override
  public int[] getElements() {
    return Elements;
  }

  @Override
  public double[] getMassFractions() {
    return massFractions;
  }
}
