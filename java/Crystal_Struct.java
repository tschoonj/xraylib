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
import org.apache.commons.numbers.complex.Complex;

/** 
 * This class allows for dealing with crystal structures.
 *
 * All methods in this class can also be accessed using their static method counterparts
 * in the @see Xraylib class, if you prefer to stay closer to the C-API.
 * 
 * If an invalid argument has been passed to any of these methods,
 * an @see IllegalArgumentException will be thrown.
 * 
 * @author Tom Schoonjans (Tom.Schoonjans@diamond.ac.uk)
 * @since 3.2.0
 */
public class Crystal_Struct {
  public final String name;
  public final double a;
  public final double b;
  public final double c;
  public final double alpha;
  public final double beta;
  public final double gamma;
  public final double volume;
  public final int n_atom;
  public final Crystal_Atom[] atom;

  protected Crystal_Struct(ByteBuffer byte_buffer) {
    name = Xraylib.readString(byte_buffer);
    a = byte_buffer.getDouble();
    b = byte_buffer.getDouble();
    c = byte_buffer.getDouble();
    alpha = byte_buffer.getDouble();
    beta = byte_buffer.getDouble();
    gamma = byte_buffer.getDouble();
    volume = byte_buffer.getDouble();
    n_atom = byte_buffer.getInt();
    atom = new Crystal_Atom[n_atom];
    for (int i = 0 ; i < n_atom ; i++) {
      atom[i] = new Crystal_Atom(byte_buffer);
    }
  }

  /** Copy constructor
   * 
   * @param cs The instance that will be copied
   */
  public Crystal_Struct(Crystal_Struct cs) {
    name = new String(cs.name);
    a = cs.a;
    b = cs.b;
    c = cs.c;
    alpha = cs.alpha;
    beta = cs.beta;
    gamma = cs. gamma;
    volume = cs.volume;
    n_atom = cs.n_atom;
    atom = new Crystal_Atom[cs.n_atom];
    for (int i = 0 ; i < cs.n_atom ; i++) {
      atom[i] = new Crystal_Atom(cs.atom[i]);
    }
  }

  /** Calculates the Bragg angle, given an energy and set of Miller indices
   *  
   * 
   * @param energy expressed in keV
   * @param i_miller Miller index i
   * @param j_miller Miller index j
   * @param k_miller Miller index k
   * @return the Bragg angle
   */
  public double Bragg_angle(double energy, int i_miller, int j_miller, int k_miller) {
    if (energy <= 0.0)
      throw new IllegalArgumentException(Xraylib.NEGATIVE_ENERGY);

    double d_spacing = Crystal_dSpacing(i_miller, j_miller, k_miller);

    double wavelength = Xraylib.KEV2ANGST / energy;
    return Math.asin(wavelength / (2.0 * d_spacing));
  }

  /** Calculates the Q scattering amplitude, given an energy, Miller indices and relative angle
   *  
   * 
   * @param energy expressed in keV
   * @param i_miller Miller index i
   * @param j_miller Miller index j
   * @param k_miller Miller index k
   * @param rel_angle expressed in radians
   * @return The Q scattering amplitude
   */
  public double Q_scattering_amplitude(double energy, int i_miller, int j_miller, int k_miller, double rel_angle) {
    if (i_miller == 0 && j_miller == 0 && k_miller == 0)
      return 0.0;
    double wavelength = Xraylib.KEV2ANGST / energy;
    return Math.sin(rel_angle * Bragg_angle(energy, i_miller, j_miller, k_miller)) / wavelength;
  }

  /** Calculates the crystal structure factor
   *  
   * 
   * @param energy expressed in keV
   * @param i_miller Miller index i
   * @param j_miller Miller index j
   * @param k_miller Miller index k
   * @param debye_factor The Debye factor
   * @param rel_angle expressed in radians
   * @return the crystal structure factor, as a complex number
   */
  public Complex Crystal_F_H_StructureFactor (double energy, int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle) {
    return Crystal_F_H_StructureFactor_Partial (energy, i_miller, j_miller, k_miller, debye_factor, rel_angle, 2, 2, 2);
  }

  /** Calculates the partial crystal structure factor
   *  
   * 
   * @param energy expressed in keV
   * @param i_miller Miller index i
   * @param j_miller Miller index j
   * @param k_miller Miller index k
   * @param debye_factor The Debye factor
   * @param rel_angle expressed in radians
   * @param f0_flag 
   * @param f_prime_flag
   * @param f_prime2_flag
   * @return the crystal structure factor, as a complex number
   */
  public Complex Crystal_F_H_StructureFactor_Partial(double energy,
                      int i_miller, int j_miller, int k_miller, double debye_factor, double rel_angle,
                      int f0_flag, int f_prime_flag, int f_prime2_flag) {

    double[] f_re = new double[120];
    double[] f_im = new double[120];
    double H_dot_r;
    boolean[] f_is_computed = new boolean[120];
    int i, Z;

    /* Loop over all atoms and compute the f values */

    double q = Q_scattering_amplitude(energy, i_miller, j_miller, k_miller, rel_angle);

    for (i = 0; i < n_atom; i++) {
      Z = atom[i].Zatom;
      if (f_is_computed[Z])
        continue;

      double[] factors = Xraylib.Atomic_Factors(Z, energy, q, debye_factor);
      double f0 = factors[0];
      double f_prime = factors[1];
      double f_prime2 = factors[2];

      switch (f0_flag) {
        case 0:
          f_re[Z] = 0;
          break;
        case 1:
          f_re[Z] = 1;
          break;
        case 2:
          f_re[Z] = f0;
          break;
        default:
          throw new IllegalArgumentException(String.format("Invalid f0_flag argument: %d", f0_flag));
      }

      switch (f_prime_flag) {
        case 0:
          break;
        case 2:
          f_re[Z] = f_re[Z] + f_prime;
          break;
        default:
          throw new IllegalArgumentException(String.format("Invalid f_prime_flag argument: %d", f_prime_flag));
      }

      switch (f_prime2_flag) {
        case 0:
          f_im[Z] = 0;
          break;
        case 2:
          f_im[Z] = f_prime2;
          break;
        default:
          throw new IllegalArgumentException(String.format("Invalid f_prime2_flag argument: %d", f_prime_flag));
      }

      f_is_computed[Z] = true;

    }

    /* Now compute F_H */
    double F_H_re = 0.0;
    double F_H_im = 0.0;

    for (i = 0; i < n_atom; i++) {
      Z = atom[i].Zatom;
      H_dot_r = 2.0 * Math.PI * (i_miller * atom[i].x + j_miller * atom[i].y + k_miller * atom[i].z);
      F_H_re += atom[i].fraction * (f_re[Z] * Math.cos(H_dot_r) - f_im[Z] * Math.sin(H_dot_r));
      F_H_im += atom[i].fraction * (f_re[Z] * Math.sin(H_dot_r) + f_im[Z] * Math.cos(H_dot_r));
    }

    return Complex.ofCartesian(F_H_re, F_H_im);
  }

  private double pow2(double arg) {
    return Math.pow(arg, 2.0);
  }

  private double sind(double arg) {
    return Math.sin(arg * Math.PI / 180.0);
  }

  private double cosd(double arg) {
    return Math.cos(arg * Math.PI / 180.0);
  }


  /** Calculate the unit cell volume 
   *  
   * 
   * @return The unit cell volume
   */
  public double Crystal_UnitCellVolume() {
    return a * b * c * Math.sqrt((1.0 - pow2(cosd(alpha)) - pow2(cosd(beta)) - pow2(cosd(gamma))) + 2.0 * cosd(alpha) * cosd(beta) * cosd(gamma));
  }

  /** Calculates the d-spacing for the crystal and Miller indices.
   *  
   * 
   * @param i_miller Miller index i
   * @param j_miller Miller index j
   * @param k_miller Miller index k
   * @return The crystal D-spacing
   */
  public double Crystal_dSpacing(int i_miller, int j_miller, int k_miller) {

    if (i_miller == 0 && j_miller == 0 && k_miller == 0)
      throw new IllegalArgumentException(Xraylib.INVALID_MILLER);

    return (volume / (a * b * c)) * Math.sqrt(1.0 / (
     pow2(i_miller * sind(alpha) / a) + pow2(j_miller * sind(beta) / b) +
          pow2(k_miller * sind(gamma) / c) +
          2 * i_miller * j_miller * (cosd(alpha) * cosd(beta)  - cosd(gamma)) / (a * b) +
          2 * i_miller * k_miller * (cosd(alpha) * cosd(gamma) - cosd(beta))  / (a * c) +
          2 * j_miller * k_miller * (cosd(beta) * cosd(gamma)  - cosd(alpha)) / (b * c)));
  }
}
