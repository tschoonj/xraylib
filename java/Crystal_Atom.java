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
public class Crystal_Atom {
  public final int Zatom;
  public final double fraction;
  public final double x;
  public final double y;
  public final double z;

  protected Crystal_Atom(ByteBuffer byte_buffer) {
    Zatom = byte_buffer.getInt();
    fraction = byte_buffer.getDouble();
    x = byte_buffer.getDouble();
    y = byte_buffer.getDouble();
    z = byte_buffer.getDouble();
  }

  /** 
   *  
   * 
   * @param ca an instance of this class that will be copied
   */
  public Crystal_Atom(Crystal_Atom ca) {
    Zatom = ca.Zatom;
    fraction = ca.fraction;
    x = ca.x;
    y = ca.y;
    z = ca.z;
  }
 
}
