/*
Copyright (c) 2016, Tom Schoonjans
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Teemu Ikonen and Tom Schoonjans ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Teemu Ikonen and Tom Schoonjans BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

package com.github.tschoonj.xraylib;

import java.nio.ByteBuffer;


/** 
 *  
 * 
 * @author Tom Schoonjans (Tom.Schoonjans@diamond.ac.uk)
 * @since 3.2.0
 */
public class compoundDataNIST implements compoundDataBase {
  public final String name;
  public final int nElements;
  public final int[] Elements;
  public final double[] massFractions;
  public final double density;
  private final String formattedNISTCompoundString;

  protected compoundDataNIST(ByteBuffer byte_buffer) {
    name = Xraylib.readString(byte_buffer);
    nElements = byte_buffer.getInt();
    Elements = Xraylib.readIntArray(nElements, byte_buffer);
    massFractions = Xraylib.readDoubleArray(nElements, byte_buffer);
    density = byte_buffer.getDouble();

    String formattedNISTCompoundString = String.format("%s contains %d elements", name, nElements);
    for (int i = 0 ; i < nElements ; i++) {
      formattedNISTCompoundString += String.format("\nElement %d: %f %%", Elements[i], massFractions[i]*100.0);
    }
    formattedNISTCompoundString += String.format("\nDensity: %f g/cm3", density);
    this.formattedNISTCompoundString = formattedNISTCompoundString;
  }

  /** 
   *  Copy constructor
   * 
   * @param cd instance that will be copied
   */
  public compoundDataNIST(compoundDataNIST cd) {
    this.name = new String(cd.name);
    this.nElements = cd.nElements;
    this.Elements = new int[cd.nElements];
    System.arraycopy(cd.Elements, 0, this.Elements, 0, cd.nElements);
    this.massFractions = new double[cd.nElements];
    System.arraycopy(cd.massFractions, 0, this.massFractions, 0, cd.nElements);
    this.density = cd.density;
    this.formattedNISTCompoundString = new String(cd.formattedNISTCompoundString);
  }

  @Override
  public String toString() {
    return formattedNISTCompoundString;
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
