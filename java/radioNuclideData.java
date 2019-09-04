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

import java.nio.ByteBuffer;

/** 
 *  
 * 
 * @author Tom Schoonjans (Tom.Schoonjans@diamond.ac.uk)
 * @since 3.2.0
 */
public class radioNuclideData {
  public final String name;
  public final int Z;
  public final int A;
  public final int N;
  public final int Z_xray;
  public final int nXrays;
  public final int[] XrayLines;
  public final double[] XrayIntensities;
  public final int nGammas;
  public final double[] GammaEnergies;
  public final double[] GammaIntensities;
  private final String formattedRadioNuclideString;

  protected radioNuclideData(ByteBuffer byte_buffer) {
    name = Xraylib.readString(byte_buffer);
    Z = byte_buffer.getInt();
    A = byte_buffer.getInt();
    N = byte_buffer.getInt();
    Z_xray = byte_buffer.getInt();
    nXrays = byte_buffer.getInt();
    XrayLines = Xraylib.readIntArray(nXrays, byte_buffer);
    XrayIntensities = Xraylib.readDoubleArray(nXrays, byte_buffer);
    nGammas = byte_buffer.getInt();
    GammaEnergies = Xraylib.readDoubleArray(nGammas, byte_buffer);
    GammaIntensities = Xraylib.readDoubleArray(nGammas, byte_buffer);

    String formattedRadioNuclideString = String.format("Name : %s\nZ: %d\nA: %d\nN: %d\nZ_xray: %d\nnXrays: %d", name, Z, A, N, Z_xray, nXrays);
    for (int i = 0 ; i < nXrays ; i++) {
      formattedRadioNuclideString += String.format("\n%f keV -> %f", Xraylib.LineEnergy(Z_xray, XrayLines[i]), XrayIntensities[i]);
    }
    formattedRadioNuclideString += String.format("\nnGammas: %d", nGammas);
    for (int i = 0 ; i < nGammas ; i++) {
      formattedRadioNuclideString += String.format("\n%f keV -> %f", GammaEnergies[i], GammaIntensities[i]);
    }
    this.formattedRadioNuclideString = formattedRadioNuclideString;
  }

  /** 
   * Copy constructor 
   * 
   * @param rnd The instance that will be copied
   */
  public radioNuclideData(radioNuclideData rnd) {
    name = new String(rnd.name);
    Z = rnd.Z;
    A = rnd.A;
    N = rnd.N;
    Z_xray = rnd.Z_xray;
    nXrays = rnd.nXrays;
    XrayLines = new int[rnd.nXrays];
    XrayIntensities = new double[rnd.nXrays];
    System.arraycopy(rnd.XrayLines, 0, XrayLines, 0, rnd.nXrays);
    System.arraycopy(rnd.XrayIntensities, 0, XrayIntensities, 0, rnd.nXrays);
    nGammas = rnd.nGammas;
    GammaEnergies = new double[rnd.nGammas];
    GammaIntensities = new double[rnd.nGammas];
    System.arraycopy(rnd.GammaEnergies, 0, GammaEnergies, 0, rnd.nGammas);
    System.arraycopy(rnd.GammaIntensities, 0, GammaIntensities, 0, rnd.nGammas);
    formattedRadioNuclideString = new String(rnd.formattedRadioNuclideString);
  }

  @Override
  public String toString() {
    return formattedRadioNuclideString;
  }
}
