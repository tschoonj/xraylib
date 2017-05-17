/*
Copyright (c) 2009, Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * The names of the contributors may not be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Bruno Golosio, Antonio Brunetti, Manuel Sanchez del Rio, Tom Schoonjans and Teemu Ikonen BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

%module(docstring="xraylib: a library for X-ray--matter interactions") xraylib
%feature("autodoc", "3");
%include "typemaps.i"
%include "exception.i"

%apply double *OUTPUT { double* f0, double* f_primep, double* f_prime2 }

%begin %{
#ifdef SWIGRUBY_TRICK
#include <config.h>
#undef PACKAGE_NAME
#undef PACKAGE_TARNAME
#undef PACKAGE_VERSION
#undef PACKAGE_STRING
#undef PACKAGE_BUGREPORT
#endif
#ifdef _WIN64
  #define MS_WIN64
  #define Py_InitModule4 Py_InitModule4_64
#endif
%}

#ifdef SWIGPYTHON
%pythoncode {
__version__ = VERSION 
}
#endif


%{
#ifdef SWIGPYTHON

#undef c_abs
/* include numpy headers */
  #ifndef WITHOUT_NUMPY
    #include <numpy/ndarraytypes.h>
    #include <numpy/ndarrayobject.h>
  #endif
#endif

#ifdef SWIGLUA
  #if LUA_VERSION_NUM < 502
    #ifndef lua_rawlen
      #define lua_rawlen lua_objlen
    #endif
  #endif
#endif

#include "xraylib.h"


%}

#ifdef SWIGPYTHON
  #ifndef WITHOUT_NUMPY
%init %{
    import_array();
%}
  #endif
#endif

#if defined(SWIGPHP) && !defined(SWIGPHP5) && !defined(SWIGPHP7)
/* hack to ensure PHP bindings will still get built with old SWIGs */
#define SWIGPHP5
#endif

#if !defined(SWIGLUA) && !defined(SWIGPHP)
%ignore c_abs;
%ignore c_mul;
#endif

%ignore Crystal_AddCrystal;
%ignore Crystal_ReadFile;
%ignore Crystal_Array;
%ignore xrlFree;
%ignore FreeCompoundData;
%ignore FreeCompoundDataNIST;
%ignore compoundData;
%ignore compoundDataNIST;
%ignore xrlComplex;
%ignore radioNuclideData;
%ignore FreeRadioNuclideData;

%typemap(newfree) char * {
        if ($1)
                xrlFree($1);
}

%newobject AtomicNumberToSymbol;

#ifndef SWIGJAVA
%typemap(in, numinputs=0) Crystal_Array* c_array {
   /* do not use crystal_array argument for now... */
   $1 = NULL;
}
%typemap(in, numinputs=0) int* nRadioNuclides {
   $1 = NULL;
}
%typemap(in, numinputs=0) int* nCompounds {
   $1 = NULL;
}
%typemap(in, numinputs=0) int* nCrystals {
   $1 = NULL;
}
#endif

#ifdef SWIGLUA
%typemap(out) char ** {
        int i=0;
        char ** list = $1;

        lua_newtable(L);
        for (i = 0 ; list[i] != NULL ; i++) {
                lua_pushinteger(L,i+1);
                lua_pushstring(L, list[i]);
                lua_settable(L, -3);
                xrlFree(list[i]);
        }
        xrlFree(list);
        lua_pushvalue(L,-1);

        SWIG_arg++;
}

%typemap(out) struct radioNuclideData * {
        int i;
        struct radioNuclideData *rnd = $1;

        if (rnd == NULL) {
                fprintf(stderr,"Error: requested radionuclide not found in database\n");
                lua_pushnil(L);
                SWIG_arg++;
        }
        else {
                lua_newtable(L);

                lua_pushstring(L, "name");
                lua_pushstring(L, rnd->name);
                lua_settable(L,-3);

                lua_pushstring(L, "Z");
                lua_pushinteger(L, rnd->Z);
                lua_settable(L,-3);

                lua_pushstring(L, "A");
                lua_pushinteger(L, rnd->A);
                lua_settable(L,-3);

                lua_pushstring(L, "N");
                lua_pushinteger(L, rnd->N);
                lua_settable(L,-3);

                lua_pushstring(L, "Z_xray");
                lua_pushinteger(L, rnd->Z_xray);
                lua_settable(L,-3);

                lua_pushstring(L, "nXrays");
                lua_pushinteger(L, rnd->nXrays);
                lua_settable(L,-3);

                lua_pushstring(L, "nGammas");
                lua_pushinteger(L, rnd->nGammas);
                lua_settable(L,-3);

                lua_pushstring(L, "XrayLines");
                lua_createtable(L, rnd->nXrays, 0);
                for (i = 0 ; i < rnd->nXrays; i++) {
                        lua_pushinteger(L, i+1);
                        lua_pushinteger(L, rnd->XrayLines[i]);
                        lua_settable(L,-3);
                }
                lua_settable(L, -3);

                lua_pushstring(L, "XrayIntensities");
                lua_createtable(L, rnd->nXrays, 0);
                for (i = 0 ; i < rnd->nXrays; i++) {
                        lua_pushinteger(L,i+1);
                        lua_pushnumber(L, rnd->XrayIntensities[i]);
                        lua_settable(L,-3);
                }
                lua_settable(L, -3);

                lua_pushstring(L, "GammaEnergies");
                lua_createtable(L, rnd->nGammas, 0);
                for (i = 0 ; i < rnd->nGammas; i++) {
                        lua_pushinteger(L, i+1);
                        lua_pushnumber(L, rnd->GammaEnergies[i]);
                        lua_settable(L,-3);
                }
                lua_settable(L, -3);

                lua_pushstring(L, "GammaIntensities");
                lua_createtable(L, rnd->nGammas, 0);
                for (i = 0 ; i < rnd->nGammas; i++) {
                        lua_pushinteger(L,i+1);
                        lua_pushnumber(L, rnd->GammaIntensities[i]);
                        lua_settable(L,-3);
                }
                lua_settable(L, -3);


                lua_pushvalue(L, -1);

                FreeRadioNuclideData(rnd);

                SWIG_arg++;
        }
}

%typemap(out) struct compoundDataNIST * {
        int i;
        struct compoundDataNIST *cdn = $1;

        if (cdn == NULL) {
                fprintf(stderr,"Error: requested NIST compound not found in database\n");
                lua_pushnil(L);
                SWIG_arg++;
        }
        else {
                lua_newtable(L);

                lua_pushstring(L, "name");
                lua_pushstring(L, cdn->name);
                lua_settable(L,-3);

                lua_pushstring(L, "nElements");
                lua_pushinteger(L, cdn->nElements);
                lua_settable(L,-3);

                lua_pushstring(L, "Elements");
                lua_createtable(L, cdn->nElements, 0);
                for (i = 0 ; i < cdn->nElements ; i++) {
                        lua_pushinteger(L,i+1);
                        lua_pushinteger(L,cdn->Elements[i]);
                        lua_settable(L,-3);
                }
                lua_settable(L, -3);

                lua_pushstring(L, "massFractions");
                lua_createtable(L, cdn->nElements, 0);
                for (i = 0 ; i < cdn->nElements ; i++) {
                        lua_pushinteger(L,i+1);
                        lua_pushnumber(L,cdn->massFractions[i]);
                        lua_settable(L,-3);
                }
                lua_settable(L, -3);

                lua_pushstring(L, "density");
                lua_pushnumber(L, cdn->density);
                lua_settable(L,-3);

                lua_pushvalue(L, -1);

                FreeCompoundDataNIST(cdn);

                SWIG_arg++;
        }
}


%typemap(out) struct compoundData * {
        int i;
        struct compoundData *cd = $1;

        if (cd == NULL) {
                lua_pushnil(L);
                SWIG_arg++;
        }
        else {
                lua_newtable(L);

                lua_pushstring(L, "nElements");
                lua_pushinteger(L, cd->nElements);
                lua_settable(L,-3);

                lua_pushstring(L, "nAtomsAll");
                lua_pushnumber(L, cd->nAtomsAll);
                lua_settable(L,-3);

                lua_pushstring(L, "Elements");
                lua_createtable(L, cd->nElements, 0);
                for (i = 0 ; i < cd->nElements ; i++) {
                        lua_pushinteger(L,i+1);
                        lua_pushinteger(L,cd->Elements[i]);
                        lua_settable(L,-3);
                }
                lua_settable(L, -3);

                lua_pushstring(L, "massFractions");
                lua_createtable(L, cd->nElements, 0);
                for (i = 0 ; i < cd->nElements ; i++) {
                        lua_pushinteger(L,i+1);
                        lua_pushnumber(L,cd->massFractions[i]);
                        lua_settable(L,-3);
                }
                lua_settable(L, -3);

                lua_pushstring(L, "nAtoms");
                lua_createtable(L, cd->nElements, 0);
                for (i = 0 ; i < cd->nElements ; i++) {
                        lua_pushinteger(L,i+1);
                        lua_pushnumber(L,cd->nAtoms[i]);
                        lua_settable(L,-3);
                }
                lua_settable(L, -3);

                lua_pushstring(L, "molarMass");
                lua_pushnumber(L, cd->molarMass);
                lua_settable(L,-3);

                lua_pushvalue(L, -1);
                FreeCompoundData(cd);
                SWIG_arg++;
        }
}

%typemap(out) xrlComplex {
        xrlComplex c = $1;

        lua_newtable(L);

        lua_pushstring(L, "re");
        lua_pushnumber(L, c.re);
        lua_settable(L,-3);

        lua_pushstring(L, "im");
        lua_pushnumber(L, c.im);
        lua_settable(L,-3);
        lua_pushvalue(L,-1);

        SWIG_arg++;
}


%typemap(out) Crystal_Struct * {
        Crystal_Struct *cs = $1;
        int i;

        if (cs == NULL) {
                fprintf(stderr,"Crystal_GetCrystal Error: crystal not found");
                lua_pushnil(L);
                SWIG_arg++;
        }
        else {
                lua_newtable(L);

                lua_pushstring(L, "name");
                lua_pushstring(L, cs->name);
                lua_settable(L,-3);

                lua_pushstring(L, "a");
                lua_pushnumber(L, cs->a);
                lua_settable(L,-3);

                lua_pushstring(L, "b");
                lua_pushnumber(L, cs->b);
                lua_settable(L,-3);

                lua_pushstring(L, "c");
                lua_pushnumber(L, cs->c);
                lua_settable(L,-3);

                lua_pushstring(L, "alpha");
                lua_pushnumber(L, cs->alpha);
                lua_settable(L,-3);

                lua_pushstring(L, "beta");
                lua_pushnumber(L, cs->beta);
                lua_settable(L,-3);

                lua_pushstring(L, "gamma");
                lua_pushnumber(L, cs->gamma);
                lua_settable(L,-3);

                lua_pushstring(L, "volume");
                lua_pushnumber(L, cs->volume);
                lua_settable(L,-3);

                lua_pushstring(L, "n_atom");
                lua_pushinteger(L, cs->n_atom);
                lua_settable(L,-3);

                lua_pushstring(L, "atom");
                lua_createtable(L, cs->n_atom, 0);
                for (i = 0 ; i < cs->n_atom ; i++) {
                        lua_pushinteger(L, i+1);
                        lua_createtable(L,0,5);
                        lua_pushstring(L,"Zatom");
                        lua_pushinteger(L, cs->atom[i].Zatom);
                        lua_settable(L,-3);

                        lua_pushstring(L,"fraction");
                        lua_pushnumber(L, cs->atom[i].fraction);
                        lua_settable(L,-3);

                        lua_pushstring(L,"x");
                        lua_pushnumber(L, cs->atom[i].x);
                        lua_settable(L,-3);

                        lua_pushstring(L,"y");
                        lua_pushnumber(L, cs->atom[i].y);
                        lua_settable(L,-3);

                        lua_pushstring(L,"z");
                        lua_pushnumber(L, cs->atom[i].z);
                        lua_settable(L,-3);

                        lua_settable(L,-3);
                }
                lua_settable(L, -3);

                lua_pushstring(L, "cpointer");
                lua_pushlightuserdata(L, (void*) cs);
                lua_settable(L, -3);

                lua_pushvalue(L, -1);

                SWIG_arg++;

        }



}

%typemap(in) xrlComplex {
        xrlComplex c;
       if (!lua_istable(L, $input)) {
                SWIG_exception(SWIG_TypeError,"Argument must be a table");
       }

       lua_pushstring(L, "re");
       lua_gettable(L, $input);
       if (lua_isnumber(L,-1)) {
                c.re = lua_tonumber(L,-1);
                lua_pop(L,1);
       }
       else {
                SWIG_exception(SWIG_RuntimeError,"re hash key not present or value not a float");
       }

       lua_pushstring(L, "im");
       lua_gettable(L, $input);
       if (lua_isnumber(L,-1)) {
                c.im = lua_tonumber(L,-1);
                lua_pop(L,1);
       }
       else {
                SWIG_exception(SWIG_RuntimeError,"im hash key not present or value not a float");
       }

       $1 = c;
}


%typemap(in) Crystal_Struct * {
       Crystal_Struct *cs;
       int i;

       if (!lua_istable(L, $input)) {
                SWIG_exception(SWIG_TypeError,"Argument must be a table");
       }

       lua_pushstring(L, "cpointer");
       lua_gettable(L, $input);
       if (lua_islightuserdata(L,-1)) {
                cs = (Crystal_Struct*) lua_topointer(L,-1);
                lua_pop(L,1);
       }
       else {
                /* name */
                cs = (Crystal_Struct *) malloc(sizeof(Crystal_Struct));
                lua_pushstring(L, "name");
                lua_gettable(L, $input);
                if (lua_isstring(L,-1)) {
                        cs->name = strdup(lua_tostring(L,-1));
                        lua_pop(L,1);
                }
                else {
                        SWIG_exception(SWIG_RuntimeError,"name hash key not present or value not a string");
                }
                /* a */
                lua_pushstring(L, "a");
                lua_gettable(L, $input);
                if (lua_isnumber(L,-1)) {
                        cs->a = lua_tonumber(L,-1);
                        lua_pop(L,1);
                }
                else {
                        SWIG_exception(SWIG_RuntimeError,"a hash key not present or value not a float");
                }
                /* b */
                lua_pushstring(L, "b");
                lua_gettable(L, $input);
                if (lua_isnumber(L,-1)) {
                        cs->b = lua_tonumber(L,-1);
                        lua_pop(L,1);
                }
                else {
                        SWIG_exception(SWIG_RuntimeError,"b hash key not present or value not a float");
                }
                /* c */
                lua_pushstring(L, "c");
                lua_gettable(L, $input);
                if (lua_isnumber(L,-1)) {
                        cs->c = lua_tonumber(L,-1);
                        lua_pop(L,1);
                }
                else {
                        SWIG_exception(SWIG_RuntimeError,"c hash key not present or value not a float");
                }
                /* alpha */
                lua_pushstring(L, "alpha");
                lua_gettable(L, $input);
                if (lua_isnumber(L,-1)) {
                        cs->alpha = lua_tonumber(L,-1);
                        lua_pop(L,1);
                }
                else {
                        SWIG_exception(SWIG_RuntimeError,"alpha hash key not present or value not a float");
                }
                /* beta */
                lua_pushstring(L, "beta");
                lua_gettable(L, $input);
                if (lua_isnumber(L,-1)) {
                        cs->beta= lua_tonumber(L,-1);
                        lua_pop(L,1);
                }
                else {
                        SWIG_exception(SWIG_RuntimeError,"beta hash key not present or value not a float");
                }
                /* gamma */
                lua_pushstring(L, "gamma");
                lua_gettable(L, $input);
                if (lua_isnumber(L,-1)) {
                        cs->gamma= lua_tonumber(L,-1);
                        lua_pop(L,1);
                }
                else {
                        SWIG_exception(SWIG_RuntimeError,"gamma hash key not present or value not a float");
                }
                /* volume */
                lua_pushstring(L, "volume");
                lua_gettable(L, $input);
                if (lua_isnumber(L,-1)) {
                        cs->volume = lua_tonumber(L,-1);
                        lua_pop(L,1);
                }
                else {
                        SWIG_exception(SWIG_RuntimeError,"volume hash key not present or value not a float");
                }
                /* n_atom */
                lua_pushstring(L, "n_atom");
                lua_gettable(L, $input);
                if (lua_isnumber(L,-1)) {
                        cs->n_atom = lua_tointeger(L,-1);
                        lua_pop(L,1);
                }
                else {
                        SWIG_exception(SWIG_RuntimeError,"n_atom hash key not present or value not a float");
                }
                if (cs->n_atom < 1) {
                        SWIG_exception(SWIG_RuntimeError,"n_atom hash value must be greater than zero");
                }
                /* atom */
                lua_getfield(L, $input,"atom");
                if (lua_istable(L,-1)) {
                        /* count number of elements */
                        size_t n_atom = lua_rawlen(L, -1);
                        if (n_atom != cs->n_atom) {
                                SWIG_exception(SWIG_RuntimeError,"n_atom hash value differs from number of elements");
                        }
                        cs->atom = (Crystal_Atom *) malloc(sizeof(Crystal_Atom)*cs->n_atom);
                        for (i = 0 ; i < cs->n_atom ; i++) {
                                lua_rawgeti(L, -1, i+1);
                                if (!lua_istable(L, -1)) {
                                        SWIG_exception(SWIG_RuntimeError,"atom hash value element must be an array (table)");
                                }
                                /* Zatom */
                                lua_getfield(L, -1, "Zatom");
                                if (lua_isnumber(L,-1)) {
                                        cs->atom[i].Zatom = lua_tointeger(L,-1);
                                        lua_pop(L,1);
                                }
                                else {
                                        SWIG_exception(SWIG_RuntimeError,"Zatom hash value element must be an integer");
                                }

                                /* fraction */
                                lua_getfield(L, -1, "fraction");
                                if (lua_isnumber(L,-1)) {
                                        cs->atom[i].fraction = lua_tonumber(L,-1);
                                        lua_pop(L,1);
                                }
                                else {
                                        SWIG_exception(SWIG_RuntimeError,"fraction hash value element must be a float");
                                }

                                /* x */
                                lua_getfield(L, -1, "x");
                                if (lua_isnumber(L,-1)) {
                                        cs->atom[i].x = lua_tonumber(L,-1);
                                        lua_pop(L,1);
                                }
                                else {
                                        SWIG_exception(SWIG_RuntimeError,"x hash value element must be a float");
                                }

                                /* y */
                                lua_getfield(L, -1, "y");
                                if (lua_isnumber(L,-1)) {
                                        cs->atom[i].y = lua_tonumber(L,-1);
                                        lua_pop(L,1);
                                }
                                else {
                                        SWIG_exception(SWIG_RuntimeError,"y hash value element must be a float");
                                }

                                /* z */
                                lua_getfield(L, -1, "z");
                                if (lua_isnumber(L,-1)) {
                                        cs->atom[i].z = lua_tonumber(L,-1);
                                        lua_pop(L,1);
                                }
                                else {
                                        SWIG_exception(SWIG_RuntimeError,"z hash value element must be a float");
                                }

                                lua_pop(L, 1);
                        }
                }
                else {
                        SWIG_exception(SWIG_RuntimeError,"atom hash value must be an array (table)");
                }
       }




       $1 = cs;


}
#endif



#ifdef SWIGPYTHON
%typemap(out) char ** {
        int i;
        char **list = $1;

        PyObject *res = PyList_New(0);
        for (i = 0 ; list[i] != NULL ; i++) {
                PyList_Append(res,PyString_FromString(list[i]));
                xrlFree(list[i]);
        }
        xrlFree(list);

        $result = res;
}

%typemap(out) struct radioNuclideData * {
        int i;
        struct radioNuclideData *rnd = $1;

        if (rnd == NULL) {
                PyErr_WarnEx(NULL, "Error: requested radionuclide not found in database\n",1);
                $result = Py_None;
        }
        else {
                PyObject *dict = PyDict_New();
                PyDict_SetItemString(dict, "name",PyString_FromString(rnd->name));
                PyDict_SetItemString(dict, "Z",PyInt_FromLong(rnd->Z));
                PyDict_SetItemString(dict, "A",PyInt_FromLong(rnd->A));
                PyDict_SetItemString(dict, "N",PyInt_FromLong(rnd->N));
                PyDict_SetItemString(dict, "Z_xray",PyInt_FromLong(rnd->Z_xray));
                PyDict_SetItemString(dict, "nXrays",PyInt_FromLong(rnd->nXrays));
                PyDict_SetItemString(dict, "nGammas",PyInt_FromLong(rnd->nGammas));
                PyObject *XrayLines = PyList_New(rnd->nXrays);
                PyObject *XrayIntensities= PyList_New(rnd->nXrays);
                PyObject *GammaEnergies= PyList_New(rnd->nGammas);
                PyObject *GammaIntensities= PyList_New(rnd->nGammas);
                for (i = 0 ; i < rnd->nXrays ; i++) {
                       PyList_SetItem(XrayLines, i, PyInt_FromLong(rnd->XrayLines[i]));
                       PyList_SetItem(XrayIntensities, i, PyFloat_FromDouble(rnd->XrayIntensities[i]));
                }
                for (i = 0 ; i < rnd->nGammas ; i++) {
                       PyList_SetItem(GammaEnergies, i, PyFloat_FromDouble(rnd->GammaEnergies[i]));
                       PyList_SetItem(GammaIntensities, i, PyFloat_FromDouble(rnd->GammaIntensities[i]));
                }
                PyDict_SetItemString(dict, "XrayLines", XrayLines);
                PyDict_SetItemString(dict, "XrayIntensities", XrayIntensities);
                PyDict_SetItemString(dict, "GammaEnergies", GammaEnergies);
                PyDict_SetItemString(dict, "GammaIntensities", GammaIntensities);

                FreeRadioNuclideData(rnd);
                $result = dict;
        }

}
%typemap(out) struct compoundDataNIST * {
        int i;
        struct compoundDataNIST *cdn = $1;

        if (cdn == NULL) {
                PyErr_WarnEx(NULL, "Error: requested NIST compound not found in database\n",1);
                $result = Py_None;
        }
        else {
                PyObject *dict = PyDict_New();
                PyDict_SetItemString(dict, "name",PyString_FromString(cdn->name));
                PyDict_SetItemString(dict, "nElements",PyInt_FromLong((int) cdn->nElements));
                PyDict_SetItemString(dict, "density",PyFloat_FromDouble(cdn->density));
                PyObject *Elements = PyList_New(cdn->nElements);
                PyObject *massFractions = PyList_New(cdn->nElements);
                for (i = 0 ; i < cdn->nElements ; i++) {
                       PyList_SetItem(Elements, i, PyInt_FromLong(cdn->Elements[i]));
                       PyList_SetItem(massFractions, i, PyFloat_FromDouble(cdn->massFractions[i]));
                }
                PyDict_SetItemString(dict, "Elements", Elements);
                PyDict_SetItemString(dict, "massFractions", massFractions);
                FreeCompoundDataNIST(cdn);
                $result = dict;
        }

}
%typemap(out) struct compoundData * {
        int i;
        struct compoundData *cd = $1;
        if (cd) {
                PyObject *dict = PyDict_New();
                PyDict_SetItemString(dict, "nElements",PyInt_FromLong((long) cd->nElements));
                PyDict_SetItemString(dict, "nAtomsAll",PyFloat_FromDouble(cd->nAtomsAll));
                PyObject *elements=PyList_New(cd->nElements);
                PyObject *massfractions=PyList_New(cd->nElements);
                PyObject *nAtoms=PyList_New(cd->nElements);
                for (i=0 ; i < cd->nElements ; i++) {
                        PyObject *o = PyInt_FromLong((long) cd->Elements[i]);
                        PyList_SetItem(elements, i, o);
                        o = PyFloat_FromDouble(cd->massFractions[i]);
                        PyList_SetItem(massfractions, i, o);
                        o = PyFloat_FromDouble(cd->nAtoms[i]);
                        PyList_SetItem(nAtoms, i, o);
                }
                PyDict_SetItemString(dict, "Elements", elements);
                PyDict_SetItemString(dict, "massFractions", massfractions);
                PyDict_SetItemString(dict, "nAtoms", nAtoms);
                PyDict_SetItemString(dict, "molarMass",PyFloat_FromDouble(cd->molarMass));
                FreeCompoundData(cd);
                $result=dict;
        }
        else {
                PyErr_WarnEx(NULL, "CompoundParser Error",1);
                $result=Py_None;
                goto fail;
        }

}



%typemap(out) xrlComplex {
        xrlComplex c = $1;
        PyObject *cp = PyComplex_FromDoubles(c.re, c.im);

        $result = cp;
}


%typemap(out) Crystal_Struct * {
        Crystal_Struct *cs = $1;
        int i;
        if (cs == NULL) {
                PyErr_WarnEx(NULL, "Crystal_GetCrystal Error: crystal not found",1);
                $result = Py_None;
        }
        else {
             PyObject *dict = PyDict_New();
             PyDict_SetItemString(dict, "name",PyString_FromString(cs->name));
             PyDict_SetItemString(dict, "a",PyFloat_FromDouble(cs->a));
             PyDict_SetItemString(dict, "b",PyFloat_FromDouble(cs->b));
             PyDict_SetItemString(dict, "c",PyFloat_FromDouble(cs->c));
             PyDict_SetItemString(dict, "alpha",PyFloat_FromDouble(cs->alpha));
             PyDict_SetItemString(dict, "beta",PyFloat_FromDouble(cs->beta));
             PyDict_SetItemString(dict, "gamma",PyFloat_FromDouble(cs->gamma));
             PyDict_SetItemString(dict, "volume",PyFloat_FromDouble(cs->volume));
             PyDict_SetItemString(dict, "n_atom",PyInt_FromLong((int) cs->n_atom));
             PyObject *atom = PyList_New(cs->n_atom);
             PyDict_SetItemString(dict, "atom", atom);
             for (i = 0 ; i < cs->n_atom ; i++) {
                PyObject *dict_temp = PyDict_New();
                PyDict_SetItemString(dict_temp, "Zatom",PyInt_FromLong((int) cs->atom[i].Zatom));
                PyDict_SetItemString(dict_temp, "fraction",PyFloat_FromDouble(cs->atom[i].fraction));
                PyDict_SetItemString(dict_temp, "x",PyFloat_FromDouble(cs->atom[i].x));
                PyDict_SetItemString(dict_temp, "y",PyFloat_FromDouble(cs->atom[i].y));
                PyDict_SetItemString(dict_temp, "z",PyFloat_FromDouble(cs->atom[i].z));
                PyList_SetItem(atom, i, dict_temp);
             }
             /* store cpointer in dictionary */
%#ifdef _WIN64
             PyDict_SetItemString(dict, "cpointer",PyInt_FromLong((long long) cs));
%#else
             PyDict_SetItemString(dict, "cpointer",PyInt_FromLong((long) cs));
%#endif

             $result = dict;
        }
}
%typemap(in) Crystal_Struct * {
        /* cpointer should be used if present and valid */
        PyObject *dict = $input;
        PyObject *cpointer = NULL;
        PyObject *temp = NULL;
        Crystal_Struct *cs = NULL;

        if (PyDict_Check(dict) == 0) {
               PyErr_SetString(PyExc_TypeError,"Expected dictionary argument");
               $1 = NULL;
               goto fail;
        }
        else {
                /* look for cpointer */
                cpointer = PyDict_GetItemString(dict,"cpointer");
                if (cpointer != NULL) {
                        /* convert to long */
%#ifdef _WIN64
                        long long cpointer_l = PyInt_AsLong(cpointer);
%#else
                        long cpointer_l = PyInt_AsLong(cpointer);
%#endif
                        if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"Invalid cpointer value");
                                $1 = NULL;
                                goto fail;
                        }
                        else {
                                $1 = (Crystal_Struct *) cpointer_l;
                        }
                }
                /* no cpointer found -> read from structure */
                else {
                        /* name */
                         cs = malloc(sizeof(Crystal_Struct));
                         temp = PyDict_GetItemString(dict,"name");
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"Name key not present");
                                $1 = NULL;
                                goto fail;
                         }
                         cs->name = PyString_AsString(temp);
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"Name key not a string");
                                $1 = NULL;
                                goto fail;
                         }
                        /* a */
                         temp = PyDict_GetItemString(dict,"a");
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"a key not present");
                                $1 = NULL;
                                goto fail;
                         }
                         cs->a = PyFloat_AsDouble(temp);
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"a key not a number");
                                $1 = NULL;
                                goto fail;
                         }
                        /* b */
                         temp = PyDict_GetItemString(dict,"b");
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"b key not present");
                                $1 = NULL;
                                goto fail;
                         }
                         cs->b = PyFloat_AsDouble(temp);
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"b key not a number");
                                $1 = NULL;
                                goto fail;
                         }
                        /* c */
                         temp = PyDict_GetItemString(dict,"c");
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"c key not present");
                                $1 = NULL;
                                goto fail;
                         }
                         cs->c = PyFloat_AsDouble(temp);
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"c key not a number");
                                $1 = NULL;
                                goto fail;
                         }
                        /* alpha */
                         temp = PyDict_GetItemString(dict,"alpha");
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"alpha key not present");
                                $1 = NULL;
                                goto fail;
                         }
                         cs->alpha = PyFloat_AsDouble(temp);
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"alpha key not a number");
                                $1 = NULL;
                                goto fail;
                         }
                        /* beta */
                         temp = PyDict_GetItemString(dict,"beta");
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"beta key not present");
                                $1 = NULL;
                                goto fail;
                         }
                         cs->beta = PyFloat_AsDouble(temp);
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"beta key not a number");
                                $1 = NULL;
                                goto fail;
                         }
                        /* gamma */
                         temp = PyDict_GetItemString(dict,"gamma");
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"gamma key not present");
                                $1 = NULL;
                                goto fail;
                         }
                         cs->gamma = PyFloat_AsDouble(temp);
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"gamma key not a number");
                                $1 = NULL;
                                goto fail;
                         }
                        /* volume */
                         temp = PyDict_GetItemString(dict,"volume");
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"volume key not present");
                                $1 = NULL;
                                goto fail;
                         }
                         cs->volume = PyFloat_AsDouble(temp);
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"volume key not a number");
                                $1 = NULL;
                                goto fail;
                         }
                        /* n_atom */
                         temp = PyDict_GetItemString(dict,"n_atom");
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"n_atom key not present");
                                $1 = NULL;
                                goto fail;
                         }
                         cs->n_atom = PyInt_AsLong(temp);
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"n_atom key not a number");
                                $1 = NULL;
                                goto fail;
                         }
                         if (cs->n_atom < 1) {
                                PyErr_SetString(PyExc_RuntimeError,"n_atom value must be greater than zero");
                                $1 = NULL;
                                goto fail;

                         }
                         /* atom */
                         cs->atom = (Crystal_Atom *) malloc(sizeof(Crystal_Atom)*cs->n_atom);
                         temp = PyDict_GetItemString(dict,"atom");
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"atom key not present");
                                $1 = NULL;
                                goto fail;
                         }
                         Py_ssize_t n_atom = PyList_Size(temp);
                         if (PyErr_Occurred() != NULL) {
                                PyErr_SetString(PyErr_Occurred(),"atom key not a list");
                                $1 = NULL;
                                goto fail;
                         }
                         if (n_atom != cs->n_atom) {
                                PyErr_SetString(PyExc_RuntimeError,"n_atom value differs from number of elements");
                                $1 = NULL;
                                goto fail;
                         }
                         int i;
                         PyObject *atom;
                         for (i=0 ; i < n_atom ; i++) {
                                atom = PyList_GetItem(temp,i);
                                PyObject *temp2;
                                temp2 = PyDict_GetItemString(atom,"Zatom");
                                if (PyErr_Occurred() != NULL) {
                                        PyErr_SetString(PyErr_Occurred(),"Zatom key not present");
                                        $1 = NULL;
                                        goto fail;
                                }
                                cs->atom[i].Zatom = PyInt_AsLong(temp2);
                                if (PyErr_Occurred() != NULL) {
                                        PyErr_SetString(PyErr_Occurred(),"Zatom key not a number");
                                        $1 = NULL;
                                        goto fail;
                                }
                                temp2 = PyDict_GetItemString(atom,"fraction");
                                if (PyErr_Occurred() != NULL) {
                                        PyErr_SetString(PyErr_Occurred(),"fraction key not present");
                                        $1 = NULL;
                                        goto fail;
                                }
                                cs->atom[i].fraction = PyFloat_AsDouble(temp2);
                                if (PyErr_Occurred() != NULL) {
                                        PyErr_SetString(PyErr_Occurred(),"fraction key not a number");
                                        $1 = NULL;
                                        goto fail;
                                }
                                temp2 = PyDict_GetItemString(atom,"x");
                                if (PyErr_Occurred() != NULL) {
                                        PyErr_SetString(PyErr_Occurred(),"x key not present");
                                        $1 = NULL;
                                        goto fail;
                                }
                                cs->atom[i].x = PyFloat_AsDouble(temp2);
                                if (PyErr_Occurred() != NULL) {
                                        PyErr_SetString(PyErr_Occurred(),"x key not a number");
                                        $1 = NULL;
                                        goto fail;
                                }
                                temp2 = PyDict_GetItemString(atom,"y");
                                if (PyErr_Occurred() != NULL) {
                                        PyErr_SetString(PyErr_Occurred(),"y key not present");
                                        $1 = NULL;
                                        goto fail;
                                }
                                cs->atom[i].y = PyFloat_AsDouble(temp2);
                                if (PyErr_Occurred() != NULL) {
                                        PyErr_SetString(PyErr_Occurred(),"y key not a number");
                                        $1 = NULL;
                                        goto fail;
                                }
                                temp2 = PyDict_GetItemString(atom,"z");
                                if (PyErr_Occurred() != NULL) {
                                        PyErr_SetString(PyErr_Occurred(),"z key not present");
                                        $1 = NULL;
                                        goto fail;
                                }
                                cs->atom[i].z = PyFloat_AsDouble(temp2);
                                if (PyErr_Occurred() != NULL) {
                                        PyErr_SetString(PyErr_Occurred(),"z key not a number");
                                        $1 = NULL;
                                        goto fail;
                                }
                         }
                         $1=cs;
                }
        }
}
#elif defined(SWIGPERL)

#define STORE_HASH(key,value,hv) if (hv_store(hv,key,strlen(key),value,0) == NULL) {\
        goto fail;\
        }


%typemap(out) char ** {
        int i;
        char **list = $1;


        if (argvi >= items) {
                EXTEND(sp,1);
        }

        AV *res = newAV();

        for (i = 0 ; list[i] != NULL ; i++) {
                av_push(res, newSVpvn(list[i], strlen(list[i])));
                xrlFree(list[i]);
        }
        xrlFree(list);
        $result = sv_2mortal(newRV_noinc((SV*) res));

        argvi++;
}

%typemap(out) struct radioNuclideData * {
        int i;
        struct radioNuclideData *rnd = $1;

        if (argvi >= items) {
                EXTEND(sp,1);
        }
        if (rnd == NULL) {
                fprintf(stderr,"Error: requested radionuclide not found in database\n");
                $result = &PL_sv_undef;
        }
        else {
                HV *hash = newHV();
                STORE_HASH("name", newSVpvn(rnd->name, strlen(rnd->name)),hash)
                STORE_HASH("Z", newSViv(rnd->Z),hash)
                STORE_HASH("A", newSViv(rnd->A),hash)
                STORE_HASH("N", newSViv(rnd->N),hash)
                STORE_HASH("Z_xray", newSViv(rnd->Z_xray),hash)
                STORE_HASH("nXrays", newSViv(rnd->nXrays),hash)
                STORE_HASH("nGammas", newSViv(rnd->nGammas),hash)
                AV *XrayLines = newAV();
                AV *XrayIntensities = newAV();
                AV *GammaEnergies = newAV();
                AV *GammaIntensities = newAV();
                STORE_HASH("XrayLines", newRV_noinc((SV*) XrayLines),hash)
                STORE_HASH("XrayIntensities", newRV_noinc((SV*) XrayIntensities),hash)
                for (i = 0 ; i < rnd->nXrays; i++) {
                        av_push(XrayLines, newSViv(rnd->XrayLines[i]));
                        av_push(XrayIntensities, newSVnv(rnd->XrayIntensities[i]));
                }
                STORE_HASH("GammaEnergies", newRV_noinc((SV*) GammaEnergies),hash)
                STORE_HASH("GammaIntensities", newRV_noinc((SV*) GammaIntensities),hash)
                for (i = 0 ; i < rnd->nGammas; i++) {
                        av_push(GammaEnergies, newSVnv(rnd->GammaEnergies[i]));
                        av_push(GammaIntensities, newSVnv(rnd->GammaIntensities[i]));
                }
                FreeRadioNuclideData(rnd);
                $result = sv_2mortal(newRV_noinc((SV*) hash));
        }

        argvi++;
}

%typemap(out) struct compoundDataNIST * {
        int i;
        struct compoundDataNIST *cdn = $1;

        if (argvi >= items) {
                EXTEND(sp,1);
        }
        if (cdn == NULL) {
                fprintf(stderr,"Error: requested NIST compound not found in database\n");
                $result = &PL_sv_undef;
        }
        else {
                HV *hash = newHV();
                STORE_HASH("name", newSVpvn(cdn->name, strlen(cdn->name)),hash)
                STORE_HASH("nElements", newSViv(cdn->nElements),hash)
                STORE_HASH("density", newSVnv(cdn->density),hash)
                AV *Elements = newAV();
                AV *massFractions = newAV();
                STORE_HASH("Elements", newRV_noinc((SV*) Elements),hash)
                STORE_HASH("massFractions", newRV_noinc((SV*) massFractions),hash)
                for (i = 0 ; i < cdn->nElements ; i++) {
                        av_push(Elements, newSViv(cdn->Elements[i]));
                        av_push(massFractions, newSVnv(cdn->massFractions[i]));
                }
                FreeCompoundDataNIST(cdn);
                $result = sv_2mortal(newRV_noinc((SV*) hash));
        }

        argvi++;
}

%typemap(out) struct compoundData *  {
        int i;
        struct compoundData *cd = $1;
        if (argvi >= items) {
                EXTEND(sp,1);
        }
        if (cd != NULL) {
                HV *hash = newHV();
                STORE_HASH("nElements", newSViv(cd->nElements),hash)
                STORE_HASH("nAtomsAll", newSVnv(cd->nAtomsAll),hash)
                AV *Elements = newAV();
                AV *massFractions = newAV();
                AV *nAtoms = newAV();
                STORE_HASH("Elements", newRV_noinc((SV*) Elements),hash)
                STORE_HASH("massFractions", newRV_noinc((SV*) massFractions),hash)
                STORE_HASH("nAtoms", newRV_noinc((SV*) nAtoms),hash)
                for (i = 0 ; i < cd->nElements ; i++) {
                        av_push(Elements, newSViv(cd->Elements[i]));
                        av_push(massFractions, newSVnv(cd->massFractions[i]));
                        av_push(nAtoms, newSVnv(cd->nAtoms[i]));
                }
                STORE_HASH("molarMass", newSVnv(cd->molarMass),hash)
                FreeCompoundData(cd);

                $result = sv_2mortal(newRV_noinc((SV*) hash));
        }
        else {
                $result = &PL_sv_undef;
        }

        argvi++;

}

%typemap(out) xrlComplex {
        xrlComplex c = $1;
        int count;
        if (argvi >= items) {
                EXTEND(sp,1);
        }
        dSP;
        ENTER;
        SAVETMPS;
        PUSHMARK(SP);
        XPUSHs(sv_2mortal(newSVnv(c.re)));
        XPUSHs(sv_2mortal(newSVnv(c.im)));
        PUTBACK;
        count = call_pv("Math::Complex::cplx",G_SCALAR);
        SPAGAIN;
        if (count != 1)
                croak("Big Perl trouble\n");
        SV *perl_result = newSVsv(POPs);
        PUTBACK;
        FREETMPS;
        LEAVE;

        $result = sv_2mortal(perl_result);
        argvi++;
}
%typemap(out) Crystal_Struct * {
        Crystal_Struct *cs = $1;
        int i;
        if (argvi >= items) {
                EXTEND(sp,1);
        }
        if (cs == NULL) {
                fprintf(stderr,"Crystal_GetCrystal Error: crystal not found\n");
                $result = &PL_sv_undef;
        }
        else {
                HV *rv = newHV();
                STORE_HASH("name", newSVpvn(cs->name,strlen(cs->name)), rv)
                STORE_HASH("a", newSVnv(cs->a), rv)
                STORE_HASH("b", newSVnv(cs->b), rv)
                STORE_HASH("c", newSVnv(cs->c), rv)
                STORE_HASH("alpha", newSVnv(cs->alpha), rv)
                STORE_HASH("beta", newSVnv(cs->beta), rv)
                STORE_HASH("gamma", newSVnv(cs->gamma), rv)
                STORE_HASH("volume", newSVnv(cs->volume), rv)
                STORE_HASH("n_atom", newSViv(cs->n_atom), rv)
                AV *atoms = newAV();
                STORE_HASH("atom", newRV_noinc((SV*) atoms), rv)
                for (i = 0 ; i < cs->n_atom ; i++) {
                        HV *atom = newHV();
                        STORE_HASH("Zatom", newSViv(cs->atom[i].Zatom), atom)
                        STORE_HASH("fraction", newSVnv(cs->atom[i].fraction), atom)
                        STORE_HASH("x", newSVnv(cs->atom[i].x), atom)
                        STORE_HASH("y", newSVnv(cs->atom[i].y), atom)
                        STORE_HASH("z", newSVnv(cs->atom[i].z), atom)
                        av_push(atoms, newRV_noinc((SV*) atom));
                }
                STORE_HASH("cpointer", newSViv((IV) cs), rv)
                $result = sv_2mortal(newRV_noinc((SV*) rv));

        }

        argvi++;

}

%typemap(in) Crystal_Struct * {
        /* unref to obtain the hash */
        SV *ref = $input;
        Crystal_Struct *cs;
        SV **temp;
        if (SvROK(ref) && SvTYPE(SvRV(ref)) == SVt_PVHV) {
                HV* hash = (HV*) SvRV(ref);
                /* get cpointer */
                SV **cpointerPtr = hv_fetch(hash, "cpointer", 8, FALSE);
                if (cpointerPtr != NULL) {
                        cs = (Crystal_Struct *) SvIV(*cpointerPtr);
                }
                else {
                        /* no cpointer found -> read from hash */
                         cs = (Crystal_Struct *) malloc(sizeof(Crystal_Struct));
                         temp = hv_fetch(hash, "name", strlen("name"), FALSE);
                         if (temp == NULL) {
                                SWIG_exception(SWIG_RuntimeError,"name hash key not present");
                         }
                         if (SvPOK(*temp)) {
                                cs->name = SvPVX(*temp);
                         }
                         else {
                                SWIG_exception(SWIG_RuntimeError,"name hash value not a string");
                         }
                         /* a */
                         temp = hv_fetch(hash, "a", strlen("a"), FALSE);
                         if (temp == NULL) {
                                SWIG_exception(SWIG_RuntimeError,"a hash key not present");
                         }
                         if (SvNOK(*temp)) {
                                cs->a = SvNVX(*temp);
                         }
                         else {
                                SWIG_exception(SWIG_RuntimeError,"a hash value not a float");
                         }
                         /* b */
                         temp = hv_fetch(hash, "b", strlen("b"), FALSE);
                         if (temp == NULL) {
                                SWIG_exception(SWIG_RuntimeError,"b hash key not present");
                         }
                         if (SvNOK(*temp)) {
                                cs->b = SvNVX(*temp);
                         }
                         else {
                                SWIG_exception(SWIG_RuntimeError,"b hash value not a float");
                         }
                         /* c */
                         temp = hv_fetch(hash, "c", strlen("c"), FALSE);
                         if (temp == NULL) {
                                SWIG_exception(SWIG_RuntimeError,"c hash key not present");
                         }
                         if (SvNOK(*temp)) {
                                cs->c = SvNVX(*temp);
                         }
                         else {
                                SWIG_exception(SWIG_RuntimeError,"c hash value not a float");
                         }
                         /* alpha */
                         temp = hv_fetch(hash, "alpha", strlen("alpha"), FALSE);
                         if (temp == NULL) {
                                SWIG_exception(SWIG_RuntimeError,"alpha hash key not present");
                         }
                         if (SvNOK(*temp)) {
                                cs->alpha = SvNVX(*temp);
                         }
                         else {
                                SWIG_exception(SWIG_RuntimeError,"alpha hash value not a float");
                         }
                         /* beta */
                         temp = hv_fetch(hash, "beta", strlen("beta"), FALSE);
                         if (temp == NULL) {
                                SWIG_exception(SWIG_RuntimeError,"beta hash key not present");
                         }
                         if (SvNOK(*temp)) {
                                cs->beta = SvNVX(*temp);
                         }
                         else {
                                SWIG_exception(SWIG_RuntimeError,"beta hash value not a float");
                         }
                         /* gamma */
                         temp = hv_fetch(hash, "gamma", strlen("gamma"), FALSE);
                         if (temp == NULL) {
                                SWIG_exception(SWIG_RuntimeError,"gamma hash key not present");
                         }
                         if (SvNOK(*temp)) {
                                cs->gamma = SvNVX(*temp);
                         }
                         else {
                                SWIG_exception(SWIG_RuntimeError,"gamma hash value not a float");
                         }
                         /* volume */
                         temp = hv_fetch(hash, "volume", strlen("volume"), FALSE);
                         if (temp == NULL) {
                                SWIG_exception(SWIG_RuntimeError,"volume hash key not present");
                         }
                         if (SvNOK(*temp)) {
                                cs->volume = SvNVX(*temp);
                         }
                         else {
                                SWIG_exception(SWIG_RuntimeError,"volume hash value not a float");
                         }
                         /* n_atom */
                         temp = hv_fetch(hash, "n_atom", strlen("n_atom"), FALSE);
                         if (temp == NULL) {
                                SWIG_exception(SWIG_RuntimeError,"n_atom hash key not present");
                         }
                         if (SvIOK(*temp)) {
                                cs->n_atom = SvIVX(*temp);
                         }
                         else {
                                SWIG_exception(SWIG_RuntimeError,"n_atom hash value not an integer");
                         }
                         if (cs->n_atom < 1) {
                                SWIG_exception(SWIG_RuntimeError,"n_atom hash value must be greater than zero");
                         }
                         /* atom */
                         cs->atom = (Crystal_Atom *) malloc(sizeof(Crystal_Atom)*cs->n_atom);
                         temp = hv_fetch(hash, "atom", strlen("atom"), FALSE);
                         if (temp == NULL) {
                                SWIG_exception(SWIG_RuntimeError,"atom hash key not present");
                         }
                         else if (!SvROK(*temp)) {
                                SWIG_exception(SWIG_RuntimeError,"atom hash value not a reference");
                         }
                         else if (SvTYPE(SvRV(*temp)) != SVt_PVAV) {
                                SWIG_exception(SWIG_RuntimeError,"atom hash value not an array");
                         }
                         AV *atom = (AV*) SvRV(*temp);
                         if (av_len(atom)+1 != cs->n_atom) {
                                SWIG_exception(SWIG_RuntimeError,"n_atom hash value differs from number of elements");
                         }
                         int i;
                         for (i = 0 ; i < cs->n_atom ; i++) {
                                SV ** atomel = av_fetch(atom, 0, FALSE);
                                /* chech if it is a reference */
                                if (!SvROK(*atomel)) {
                                        SWIG_exception(SWIG_TypeError,"elements of atom array must be references");
                                }
                                else if (SvTYPE(SvRV(*atomel)) != SVt_PVHV) {
                                        SWIG_exception(SWIG_TypeError,"elements of atom array must be references to hash");
                                }
                                HV *atomHash = (HV*) SvRV(*atomel);
                                /* Zatom */
                                temp = hv_fetch(atomHash, "Zatom", 5, FALSE);
                                if (temp == NULL) {
                                        SWIG_exception(SWIG_RuntimeError, "Zatom hash key not present");
                                }
                                if (SvIOK(*temp)) {
                                        cs->atom[i].Zatom = SvIVX(*temp);
                                }
                                else {
                                        SWIG_exception(SWIG_RuntimeError,"Zatom hash key not an integer");
                                }
                                /* fraction */
                                temp = hv_fetch(atomHash, "fraction", 8, FALSE);
                                if (temp == NULL) {
                                        SWIG_exception(SWIG_RuntimeError, "fraction hash key not present");
                                }
                                if (SvNOK(*temp)) {
                                        cs->atom[i].fraction = SvNVX(*temp);
                                }
                                else {
                                        SWIG_exception(SWIG_RuntimeError,"fraction hash key not a real number");
                                }
                                /* x */
                                temp = hv_fetch(atomHash, "x", 1, FALSE);
                                if (temp == NULL) {
                                        SWIG_exception(SWIG_RuntimeError, "x hash key not present");
                                }
                                if (SvNOK(*temp)) {
                                        cs->atom[i].x = SvNVX(*temp);
                                }
                                else {
                                        SWIG_exception(SWIG_RuntimeError,"x hash key not a real number");
                                }
                                /* y */
                                temp = hv_fetch(atomHash, "y", 1, FALSE);
                                if (temp == NULL) {
                                        SWIG_exception(SWIG_RuntimeError, "y hash key not present");
                                }
                                if (SvNOK(*temp)) {
                                        cs->atom[i].y = SvNVX(*temp);
                                }
                                else {
                                        SWIG_exception(SWIG_RuntimeError,"y hash key not a real number");
                                }
                                /* z */
                                temp = hv_fetch(atomHash, "z", 1, FALSE);
                                if (temp == NULL) {
                                        SWIG_exception(SWIG_RuntimeError, "z hash key not present");
                                }
                                if (SvNOK(*temp)) {
                                        cs->atom[i].z = SvNVX(*temp);
                                }
                                else {
                                        SWIG_exception(SWIG_RuntimeError,"z hash key not a real number");
                                }
                         }
                }
                $1 = cs;
                argvi++;
        }
        else {
                SWIG_exception(SWIG_TypeError,"Argument must be reference to hash");
        }
}
#endif

#ifdef SWIGRUBY
%typemap(out) char ** {
        int i;
        char **list = $1;

        VALUE res = rb_ary_new();
        for (i = 0 ; list[i] != NULL ; i++) {
                rb_ary_push(res, rb_str_new2(list[i]));
                xrlFree(list[i]);
        }
        xrlFree(list);
        $result = res;
}
%typemap(out) struct radioNuclideData * {
        int i;
        struct radioNuclideData *rnd = $1;

        if (rnd== NULL) {
                fprintf(stderr, "Error: requested radionuclide not found in database\n");
                $result = Qnil;
        }
        else {
                VALUE rv = rb_hash_new();
                rb_hash_aset(rv, rb_str_new2("name"), rb_str_new2(rnd->name));
                rb_hash_aset(rv, rb_str_new2("Z"), INT2FIX(rnd->Z));
                rb_hash_aset(rv, rb_str_new2("A"), INT2FIX(rnd->A));
                rb_hash_aset(rv, rb_str_new2("N"), INT2FIX(rnd->N));
                rb_hash_aset(rv, rb_str_new2("Z_xray"), INT2FIX(rnd->Z_xray));
                rb_hash_aset(rv, rb_str_new2("nXrays"), INT2FIX(rnd->nXrays));
                rb_hash_aset(rv, rb_str_new2("nGammas"), INT2FIX(rnd->nGammas));
                VALUE XrayLines, XrayIntensities, GammaEnergies, GammaIntensities;
                XrayLines = rb_ary_new2(rnd->nXrays);
                XrayIntensities= rb_ary_new2(rnd->nXrays);
                GammaEnergies = rb_ary_new2(rnd->nGammas);
                GammaIntensities= rb_ary_new2(rnd->nGammas);
                for (i = 0 ; i < rnd->nXrays ; i++) {
                        rb_ary_store(XrayLines, (long) i , INT2FIX(rnd->XrayLines[i]));
                        rb_ary_store(XrayIntensities, (long) i , rb_float_new(rnd->XrayIntensities[i]));
                }
                for (i = 0 ; i < rnd->nGammas; i++) {
                        rb_ary_store(GammaEnergies, (long) i , rb_float_new(rnd->GammaEnergies[i]));
                        rb_ary_store(GammaIntensities, (long) i , rb_float_new(rnd->GammaIntensities[i]));
                }
                rb_hash_aset(rv, rb_str_new2("XrayLines"), XrayLines);
                rb_hash_aset(rv, rb_str_new2("XrayIntensities"), XrayIntensities);
                rb_hash_aset(rv, rb_str_new2("GammaEnergies"), GammaEnergies);
                rb_hash_aset(rv, rb_str_new2("GammaIntensities"), GammaIntensities);
                FreeRadioNuclideData(rnd);
                $result = rv;
        }

}
%typemap(out) struct compoundDataNIST * {
        int i;
        struct compoundDataNIST *cdn = $1;

        if (cdn == NULL) {
                fprintf(stderr, "Error: requested NIST compound not found in database\n");
                $result = Qnil;
        }
        else {
                VALUE rv = rb_hash_new();
                rb_hash_aset(rv, rb_str_new2("name"), rb_str_new2(cdn->name));
                rb_hash_aset(rv, rb_str_new2("nElements"), INT2FIX(cdn->nElements));
                rb_hash_aset(rv, rb_str_new2("density"), rb_float_new(cdn->density));
                VALUE elements, massFractions;
                elements = rb_ary_new2(cdn->nElements);
                massFractions = rb_ary_new2(cdn->nElements);
                for (i = 0 ; i < cdn->nElements ; i++) {
                        rb_ary_store(elements, (long) i , INT2FIX(cdn->Elements[i]));
                        rb_ary_store(massFractions, (long) i , rb_float_new(cdn->massFractions[i]));
                }
                rb_hash_aset(rv, rb_str_new2("Elements"), elements);
                rb_hash_aset(rv, rb_str_new2("massFractions"), massFractions);
                FreeCompoundDataNIST(cdn);
                $result = rv;
        }

}
%typemap(out) struct compoundData * {
        int i;
        struct compoundData *cd = $1;

        if (cd == NULL) {
               $result = Qnil;
        }
        else {
                VALUE rv;
                rv = rb_hash_new();
                rb_hash_aset(rv, rb_str_new2("nElements"), INT2FIX(cd->nElements));
                rb_hash_aset(rv, rb_str_new2("nAtomsAll"), rb_float_new(cd->nAtomsAll));
                VALUE elements, massFractions, nAtoms;
                elements = rb_ary_new2(cd->nElements);
                massFractions = rb_ary_new2(cd->nElements);
                nAtoms = rb_ary_new2(cd->nElements);
                for (i = 0 ; i < cd->nElements ; i++) {
                        rb_ary_store(elements, (long) i , INT2FIX(cd->Elements[i]));
                        rb_ary_store(massFractions, (long) i , rb_float_new(cd->massFractions[i]));
                        rb_ary_store(nAtoms, (long) i , rb_float_new(cd->nAtoms[i]));
                }
                rb_hash_aset(rv, rb_str_new2("Elements"), elements);
                rb_hash_aset(rv, rb_str_new2("massFractions"), massFractions);
                rb_hash_aset(rv, rb_str_new2("nAtoms"), nAtoms);
                rb_hash_aset(rv, rb_str_new2("molarMass"), rb_float_new(cd->molarMass));
                FreeCompoundData(cd);
                $result = rv;
        }

}

%typemap(out) Crystal_Struct * {
        Crystal_Struct *cs = $1;

        if (cs == NULL) {
                fprintf(stderr,"Crystal_GetCrystal Error: crystal not found");
                $result = Qnil;
        }
        else {
                VALUE rv;
                rv = rb_hash_new();
                rb_hash_aset(rv, rb_str_new2("name"), rb_str_new2(cs->name));
                rb_hash_aset(rv, rb_str_new2("a"), rb_float_new(cs->a));
                rb_hash_aset(rv, rb_str_new2("b"), rb_float_new(cs->b));
                rb_hash_aset(rv, rb_str_new2("c"), rb_float_new(cs->c));
                rb_hash_aset(rv, rb_str_new2("alpha"), rb_float_new(cs->alpha));
                rb_hash_aset(rv, rb_str_new2("beta"), rb_float_new(cs->beta));
                rb_hash_aset(rv, rb_str_new2("gamma"), rb_float_new(cs->gamma));
                rb_hash_aset(rv, rb_str_new2("volume"), rb_float_new(cs->volume));
                rb_hash_aset(rv, rb_str_new2("n_atom"), INT2FIX(cs->n_atom));
                VALUE atoms = rb_ary_new2(cs->n_atom);
                rb_hash_aset(rv, rb_str_new2("atom"), atoms);
                int i;
                for (i = 0 ; i < cs->n_atom ; i++) {
                        VALUE atom = rb_hash_new();
                        rb_hash_aset(atom, rb_str_new2("Zatom"), INT2FIX(cs->atom[i].Zatom));
                        rb_hash_aset(atom, rb_str_new2("fraction"), rb_float_new(cs->atom[i].fraction));
                        rb_hash_aset(atom, rb_str_new2("x"), rb_float_new(cs->atom[i].x));
                        rb_hash_aset(atom, rb_str_new2("y"), rb_float_new(cs->atom[i].y));
                        rb_hash_aset(atom, rb_str_new2("z"), rb_float_new(cs->atom[i].z));
                        rb_ary_store(atoms, (long) i, atom);
                }
                rb_hash_aset(rv, rb_str_new2("cpointer"), INT2FIX((long) cs));
                $result = rv;
        }
}

%typemap(in) Crystal_Struct * {
        VALUE input = $input;
        Crystal_Struct *cs;
        int cpointer_found = 0;

        if (TYPE(input) != T_HASH) {
                SWIG_exception(SWIG_TypeError,"Argument must be a hash");
        }

        /* get cpointer if available */
        VALUE cpointer;
        if ((cpointer = rb_hash_aref(input, rb_str_new2("cpointer"))) != Qnil ) {
                /* cpointer found */
                cs = (Crystal_Struct *) FIX2LONG(cpointer);
                cpointer_found = 1;
        }
        else {
                /* not a cpointer -> we'll have to analyze the complete hash then :-( */
                VALUE temp;
                cs = (Crystal_Struct *)  malloc(sizeof(Crystal_Struct));

                /* name */
                temp = rb_hash_aref(input, rb_str_new2("name"));
                if (temp == Qnil || TYPE(temp) != T_STRING) {
                        SWIG_exception(SWIG_RuntimeError,"name hash key not present or not a string");
                }
                cs->name = strdup(StringValuePtr(temp));
                /* a */
                temp = rb_hash_aref(input, rb_str_new2("a"));
                if (temp == Qnil || TYPE(temp) != T_FLOAT) {
                        SWIG_exception(SWIG_RuntimeError,"a hash key not present or not a float");
                }
                cs->a = NUM2DBL(temp);
                /* b */
                temp = rb_hash_aref(input, rb_str_new2("b"));
                if (temp == Qnil || TYPE(temp) != T_FLOAT) {
                        SWIG_exception(SWIG_RuntimeError,"b hash key not present or not a float");
                }
                cs->b = NUM2DBL(temp);
                /* c */
                temp = rb_hash_aref(input, rb_str_new2("c"));
                if (temp == Qnil || TYPE(temp) != T_FLOAT) {
                        SWIG_exception(SWIG_RuntimeError,"c hash key not present or not a float");
                }
                cs->c = NUM2DBL(temp);
                /* alpha */
                temp = rb_hash_aref(input, rb_str_new2("alpha"));
                if (temp == Qnil || TYPE(temp) != T_FLOAT) {
                        SWIG_exception(SWIG_RuntimeError,"alpha hash key not present or not a float");
                }
                cs->alpha = NUM2DBL(temp);
                /* beta */
                temp = rb_hash_aref(input, rb_str_new2("beta"));
                if (temp == Qnil || TYPE(temp) != T_FLOAT) {
                        SWIG_exception(SWIG_RuntimeError,"beta hash key not present or not a float");
                }
                cs->beta = NUM2DBL(temp);
                /* gamma */
                temp = rb_hash_aref(input, rb_str_new2("gamma"));
                if (temp == Qnil || TYPE(temp) != T_FLOAT) {
                        SWIG_exception(SWIG_RuntimeError,"gamma hash key not present or not a float");
                }
                cs->gamma = NUM2DBL(temp);
                /* volume */
                temp = rb_hash_aref(input, rb_str_new2("volume"));
                if (temp == Qnil || TYPE(temp) != T_FLOAT) {
                        SWIG_exception(SWIG_RuntimeError,"volume hash key not present or not a float");
                }
                cs->volume = NUM2DBL(temp);
                /* n_atom */
                temp = rb_hash_aref(input, rb_str_new2("n_atom"));
                if (temp == Qnil || TYPE(temp) != T_FIXNUM) {
                        SWIG_exception(SWIG_RuntimeError,"n_atom hash key not present or not an integer");
                }
                cs->n_atom = FIX2INT(temp);
                if (cs->n_atom < 1) {
                       SWIG_exception(SWIG_RuntimeError,"n_atom hash value must be greater than zero");
                }
                cs->atom = (Crystal_Atom *) malloc(sizeof(Crystal_Atom)*cs->n_atom);

                /* atom */
                VALUE atoms = rb_hash_aref(input, rb_str_new2("atom"));
                if (atoms == Qnil || TYPE(atoms) != T_ARRAY) {
                        SWIG_exception(SWIG_RuntimeError,"atom hash key not present or not an array");
                }
                if (RARRAY_LEN(atoms) != cs->n_atom) {
                        SWIG_exception(SWIG_RuntimeError,"n_atom hash value differs from number of elements");
                }
                long i;
                for (i = 0 ; i < cs->n_atom ; i++) {
                        VALUE atom = rb_ary_entry(atoms, i);
                        if (atom == Qnil || TYPE(atom) != T_HASH) {
                                SWIG_exception(SWIG_RuntimeError,"elements of atom array must be hashes");
                        }
                        /* Zatom */
                        temp = rb_hash_aref(atom, rb_str_new2("Zatom"));
                        if (temp == Qnil || TYPE(temp) != T_FIXNUM) {
                                SWIG_exception(SWIG_RuntimeError,"Zatom hash key missing or not an integer");
                        }
                        cs->atom[i].Zatom = FIX2INT(temp);
                        /* fraction */
                        temp = rb_hash_aref(atom, rb_str_new2("fraction"));
                        if (temp == Qnil || TYPE(temp) != T_FLOAT) {
                                SWIG_exception(SWIG_RuntimeError,"fraction hash key missing or not a float");
                        }
                        cs->atom[i].fraction = NUM2DBL(temp);
                        /* x */
                        temp = rb_hash_aref(atom, rb_str_new2("x"));
                        if (temp == Qnil || TYPE(temp) != T_FLOAT) {
                                SWIG_exception(SWIG_RuntimeError,"x hash key missing or not a float");
                        }
                        cs->atom[i].x = NUM2DBL(temp);
                        /* y */
                        temp = rb_hash_aref(atom, rb_str_new2("y"));
                        if (temp == Qnil || TYPE(temp) != T_FLOAT) {
                                SWIG_exception(SWIG_RuntimeError,"y hash key missing or not a float");
                        }
                        cs->atom[i].y = NUM2DBL(temp);
                        /* z */
                        temp = rb_hash_aref(atom, rb_str_new2("z"));
                        if (temp == Qnil || TYPE(temp) != T_FLOAT) {
                                SWIG_exception(SWIG_RuntimeError,"z hash key missing or not a float");
                        }
                        cs->atom[i].z = NUM2DBL(temp);

                }


        }

        $1 = cs;

}

%typemap(out) xrlComplex {
        xrlComplex c = $1;
%#ifdef T_COMPLEX
        /* Ruby 1.9 */
        VALUE cp = rb_Complex2(rb_float_new(c.re),rb_float_new(c.im));
%#else
        /* Ruby 1.8 */
        VALUE *args = (VALUE *) malloc(sizeof(VALUE)*2);
        args[0] = rb_float_new(c.re);
        args[1] = rb_float_new(c.im);
        /*VALUE cp = rb_class_new_instance(2, args, rb_path2class("Complex"));*/
        VALUE cp = rb_class_new_instance(2,args,rb_const_get(rb_cObject, rb_intern("Complex")));
%#endif
        $result = cp;
}

#endif


#ifdef SWIGPHP5

%typemap(out) char ** {
        int i = 0;
        char **list = $1;

        array_init(return_value);
        for (i = 0 ; list[i] != NULL ; i++) {
                add_index_string(return_value, i, list[i], 1);
                xrlFree(list[i]);
        }
        xrlFree(list);
}

%typemap(out) struct radioNuclideData * {
        int i;
        struct radioNuclideData *rnd = $1;

        if (rnd == NULL) {
                php_log_err("Error: requested radionuclide not found in database\n");
                RETURN_NULL();
        }
        array_init(return_value);
        add_assoc_string(return_value, "name", rnd->name, 1);
        add_assoc_long(return_value, "Z", rnd->Z);
        add_assoc_long(return_value, "A", rnd->A);
        add_assoc_long(return_value, "N", rnd->N);
        add_assoc_long(return_value, "Z_xray", rnd->Z_xray);
        add_assoc_long(return_value, "nXrays", rnd->nXrays);
        add_assoc_long(return_value, "nGammas", rnd->nGammas);
        zval *XrayLines, *XrayIntensities, *GammaEnergies, *GammaIntensities;

        ALLOC_INIT_ZVAL(XrayLines);
        ALLOC_INIT_ZVAL(XrayIntensities);
        ALLOC_INIT_ZVAL(GammaEnergies);
        ALLOC_INIT_ZVAL(GammaIntensities);
        array_init(XrayLines);
        array_init(XrayIntensities);
        array_init(GammaEnergies);
        array_init(GammaIntensities);
        for (i = 0 ; i < rnd->nXrays ; i++) {
                add_index_long(XrayLines, i, rnd->XrayLines[i]);
                add_index_double(XrayIntensities, i, rnd->XrayIntensities[i]);
        }
        for (i = 0 ; i < rnd->nGammas ; i++) {
                add_index_double(GammaEnergies, i, rnd->GammaEnergies[i]);
                add_index_double(GammaIntensities, i, rnd->GammaIntensities[i]);
        }
        add_assoc_zval(return_value, "XrayLines", XrayLines);
        add_assoc_zval(return_value, "XrayIntensities", XrayIntensities);
        add_assoc_zval(return_value, "GammaEnergies", GammaEnergies);
        add_assoc_zval(return_value, "GammaIntensities", GammaIntensities);
        FreeRadioNuclideData(rnd);
}
%typemap(out) struct compoundDataNIST * {
        int i;
        struct compoundDataNIST *cdn = $1;

        if (cdn == NULL) {
                php_log_err("Error: requested NIST compound not found in database\n");
                RETURN_NULL();
        }
        array_init(return_value);
        add_assoc_string(return_value, "name", cdn->name, 1);
        add_assoc_long(return_value, "nElements", cdn->nElements);
        add_assoc_double(return_value, "density", cdn->density);
        zval *Elements, *massFractions;

        ALLOC_INIT_ZVAL(Elements);
        ALLOC_INIT_ZVAL(massFractions);
        array_init(Elements);
        array_init(massFractions);
        for (i = 0 ; i < cdn->nElements ; i++) {
                add_index_long(Elements, i, cdn->Elements[i]);
                add_index_double(massFractions, i, cdn->massFractions[i]);
        }
        add_assoc_zval(return_value, "Elements", Elements);
        add_assoc_zval(return_value, "massFractions", massFractions);
        FreeCompoundDataNIST(cdn);
}

%typemap(out) struct compoundData * {
        int i;
        struct compoundData *cd = $1;

        if (cd == NULL) {
                php_log_err("CompoundParser Error\n");
                RETURN_NULL();
        }
        array_init(return_value);
        add_assoc_long(return_value, "nElements", cd->nElements);
        add_assoc_double(return_value, "nAtomsAll", cd->nAtomsAll);
        zval *Elements, *massFractions, *nAtoms;

        ALLOC_INIT_ZVAL(Elements);
        ALLOC_INIT_ZVAL(massFractions);
        ALLOC_INIT_ZVAL(nAtoms);
        array_init(Elements);
        array_init(massFractions);
        array_init(nAtoms);
        for (i = 0 ; i < cd->nElements ; i++) {
                add_index_long(Elements, i, cd->Elements[i]);
                add_index_double(massFractions, i, cd->massFractions[i]);
                add_index_double(nAtoms, i, cd->nAtoms[i]);
        }
        add_assoc_zval(return_value, "Elements", Elements);
        add_assoc_zval(return_value, "massFractions", massFractions);
        add_assoc_zval(return_value, "nAtoms", nAtoms);
        add_assoc_double(return_value, "molarMass", cd->molarMass);
        FreeCompoundData(cd);
}

%typemap(out) xrlComplex {
        xrlComplex c = $1;

        array_init(return_value);
        add_assoc_double(return_value, "re", c.re);
        add_assoc_double(return_value, "im", c.im);
}
%typemap(in) xrlComplex {
        xrlComplex c;

        if (Z_TYPE_PP($input) != IS_ARRAY) {
                SWIG_exception(SWIG_TypeError,"Argument must be an array");
        }
        if (!zend_hash_exists(Z_ARRVAL_PP($input), "re", sizeof("re"))) {
                SWIG_exception(SWIG_TypeError, "re hash key not present");
        }
        if (!zend_hash_exists(Z_ARRVAL_PP($input), "im", sizeof("im"))) {
                SWIG_exception(SWIG_TypeError, "im hash key not present");
        }
        zval **re, **im;
        zend_hash_find(Z_ARRVAL_PP($input), "re", sizeof("re"), (void **) &re);
        zend_hash_find(Z_ARRVAL_PP($input), "im", sizeof("im"), (void **) &im);
        convert_to_double_ex(re);
        convert_to_double_ex(im);
        c.re = Z_DVAL_PP(re);
        c.im = Z_DVAL_PP(im);
        $1 = c;
}

%typemap(out) Crystal_Struct * {
        Crystal_Struct *cs = $1;
        int i;
        if (cs == NULL) {
                php_log_err("Crystal_GetCrystal Error: crystal not found");
                RETURN_NULL();
        }

        array_init(return_value);
        add_assoc_string(return_value, "name", cs->name, 1);
        add_assoc_double(return_value, "a", cs->a);
        add_assoc_double(return_value, "b", cs->b);
        add_assoc_double(return_value, "c", cs->c);
        add_assoc_double(return_value, "alpha", cs->alpha);
        add_assoc_double(return_value, "beta", cs->beta);
        add_assoc_double(return_value, "gamma", cs->gamma);
        add_assoc_double(return_value, "volume", cs->volume);
        add_assoc_long(return_value, "n_atom", cs->n_atom);
        zval *atom;
        ALLOC_INIT_ZVAL(atom);
        array_init(atom);
        add_assoc_zval(return_value, "atom", atom);
        for (i = 0 ; i < cs->n_atom ; i++) {
                zval *dict_temp;
                ALLOC_INIT_ZVAL(dict_temp);
                array_init(dict_temp);
                add_assoc_long(dict_temp, "Zatom", cs->atom[i].Zatom);
                add_assoc_double(dict_temp, "fraction", cs->atom[i].fraction);
                add_assoc_double(dict_temp, "x", cs->atom[i].x);
                add_assoc_double(dict_temp, "y", cs->atom[i].y);
                add_assoc_double(dict_temp, "z", cs->atom[i].z);
                add_index_zval(atom, i, dict_temp);
        }
        add_assoc_zval(return_value, "cpointer", (zval*) cs);
}
%typemap(in) Crystal_Struct * {
        /* cpointer should be used if present and valid */

        if (Z_TYPE_PP($input) != IS_ARRAY) {
                SWIG_exception(SWIG_TypeError,"Argument must be an array");
        }
        if (zend_hash_exists(Z_ARRVAL_PP($input), "cpointer", sizeof("cpointer"))) {
                /* cpointer found */
                zval *cpointer;
                zend_hash_find(Z_ARRVAL_PP($input), "cpointer", sizeof("cpointer"), (void **) &cpointer);
                $1 = (Crystal_Struct *) Z_ARRVAL_P(cpointer);
        }
        else {
                Crystal_Struct *cs = malloc(sizeof(Crystal_Struct));
                zval *temp1, **temp2;
                if (zend_hash_find(Z_ARRVAL_PP($input), "name", sizeof("name"), (void **) &temp2) == FAILURE) {
                        SWIG_exception(SWIG_TypeError,"Name key not present");
                }
                cs->name = strdup(Z_STRVAL_PP(temp2));

                if (zend_hash_find(Z_ARRVAL_PP($input), "a", sizeof("a"), (void **) &temp2) == FAILURE) {
                        SWIG_exception(SWIG_TypeError,"a key not present");
                }
                convert_to_double_ex(temp2);
                cs->a = Z_DVAL_PP(temp2);

                if (zend_hash_find(Z_ARRVAL_PP($input), "b", sizeof("b"), (void **) &temp2) == FAILURE) {
                        SWIG_exception(SWIG_TypeError,"b key not present");
                }
                convert_to_double_ex(temp2);
                cs->b = Z_DVAL_PP(temp2);

                if (zend_hash_find(Z_ARRVAL_PP($input), "c", sizeof("c"), (void **) &temp2) == FAILURE) {
                        SWIG_exception(SWIG_TypeError,"c key not present");
                }
                convert_to_double_ex(temp2);
                cs->c = Z_DVAL_PP(temp2);

                if (zend_hash_find(Z_ARRVAL_PP($input), "alpha", sizeof("alpha"), (void **) &temp2) == FAILURE) {
                        SWIG_exception(SWIG_TypeError,"alpha key not present");
                }
                convert_to_double_ex(temp2);
                cs->alpha = Z_DVAL_PP(temp2);

                if (zend_hash_find(Z_ARRVAL_PP($input), "beta", sizeof("beta"), (void **) &temp2) == FAILURE) {
                        SWIG_exception(SWIG_TypeError,"beta key not present");
                }
                convert_to_double_ex(temp2);
                cs->beta= Z_DVAL_PP(temp2);

                if (zend_hash_find(Z_ARRVAL_PP($input), "gamma", sizeof("gamma"), (void **) &temp2) == FAILURE) {
                        SWIG_exception(SWIG_TypeError,"gamma key not present");
                }
                convert_to_double_ex(temp2);
                cs->gamma = Z_DVAL_PP(temp2);

                if (zend_hash_find(Z_ARRVAL_PP($input), "volume", sizeof("volume"), (void **) &temp2) == FAILURE) {
                        SWIG_exception(SWIG_TypeError,"volume key not present");
                }
                convert_to_double_ex(temp2);
                cs->volume = Z_DVAL_PP(temp2);

                if (zend_hash_find(Z_ARRVAL_PP($input), "n_atom", sizeof("n_atom"), (void **) &temp2) == FAILURE) {
                        SWIG_exception(SWIG_TypeError,"n_atom key not present");
                }
                convert_to_long_ex(temp2);
                cs->n_atom = (int) Z_LVAL_PP(temp2);

                zval **atom;

                if (zend_hash_find(Z_ARRVAL_PP($input), "atom", sizeof("atom"), (void **) &atom) == FAILURE) {
                        SWIG_exception(SWIG_TypeError,"atom key not present");
                }

                if (Z_TYPE_PP(atom) != IS_ARRAY) {
                        SWIG_exception(SWIG_TypeError,"atom must be an array");
                }
                int i;
                cs->atom = (Crystal_Atom *) malloc(sizeof(Crystal_Atom)*cs->n_atom);
                for (i = 0 ; i < cs->n_atom ; i++) {
                        zval **this_atom;
                        if (zend_hash_index_find(Z_ARRVAL_PP(atom), i, (void **) &this_atom) == FAILURE) {
                                SWIG_exception(SWIG_TypeError,"atom member not found\n");
                        }

                        if (zend_hash_find(Z_ARRVAL_PP(this_atom), "Zatom", sizeof("Zatom"), (void **) &temp2) == FAILURE) {
                                SWIG_exception(SWIG_TypeError,"Zatom key not found\n");
                        }
                        convert_to_long_ex(temp2);
                        cs->atom[i].Zatom = (int) Z_LVAL_PP(temp2);

                        if (zend_hash_find(Z_ARRVAL_PP(this_atom), "fraction", sizeof("fraction"), (void **) &temp2) == FAILURE) {
                                SWIG_exception(SWIG_TypeError,"fraction key not found\n");
                        }
                        convert_to_double_ex(temp2);
                        cs->atom[i].fraction = (double) Z_DVAL_PP(temp2);

                        if (zend_hash_find(Z_ARRVAL_PP(this_atom), "x", sizeof("x"), (void **) &temp2) == FAILURE) {
                                SWIG_exception(SWIG_TypeError,"x key not found\n");
                        }
                        convert_to_double_ex(temp2);
                        cs->atom[i].x = (double) Z_DVAL_PP(temp2);

                        if (zend_hash_find(Z_ARRVAL_PP(this_atom), "y", sizeof("y"), (void **) &temp2) == FAILURE) {
                                SWIG_exception(SWIG_TypeError,"y key not found\n");
                        }
                        convert_to_double_ex(temp2);
                        cs->atom[i].y = (double) Z_DVAL_PP(temp2);

                        if (zend_hash_find(Z_ARRVAL_PP(this_atom), "z", sizeof("z"), (void **) &temp2) == FAILURE) {
                                SWIG_exception(SWIG_TypeError,"z key not found\n");
                        }
                        convert_to_double_ex(temp2);
                        cs->atom[i].z = (double) Z_DVAL_PP(temp2);

                }
                $1 = cs;
        }

}
#endif

#ifdef SWIGPHP7

%typemap(out) char ** {
        int i = 0;
        char **list = $1;

        array_init(return_value);
        for (i = 0 ; list[i] != NULL ; i++) {
                add_index_string(return_value, i, list[i]);
                xrlFree(list[i]);
        }
        xrlFree(list);
}

%typemap(out) struct radioNuclideData * {
        int i;
        struct radioNuclideData *rnd = $1;

        if (rnd == NULL) {
                php_log_err("Error: requested radionuclide not found in database\n");
                RETURN_NULL();
        }
        array_init(return_value);
        add_assoc_string(return_value, "name", rnd->name);
        add_assoc_long(return_value, "Z", rnd->Z);
        add_assoc_long(return_value, "A", rnd->A);
        add_assoc_long(return_value, "N", rnd->N);
        add_assoc_long(return_value, "Z_xray", rnd->Z_xray);
        add_assoc_long(return_value, "nXrays", rnd->nXrays);
        add_assoc_long(return_value, "nGammas", rnd->nGammas);
        zval XrayLines, XrayIntensities, GammaEnergies, GammaIntensities;

        array_init(&XrayLines);
        array_init(&XrayIntensities);
        array_init(&GammaEnergies);
        array_init(&GammaIntensities);
        for (i = 0 ; i < rnd->nXrays ; i++) {
                add_index_long(&XrayLines, i, rnd->XrayLines[i]);
                add_index_double(&XrayIntensities, i, rnd->XrayIntensities[i]);
        }
        for (i = 0 ; i < rnd->nGammas ; i++) {
                add_index_double(&GammaEnergies, i, rnd->GammaEnergies[i]);
                add_index_double(&GammaIntensities, i, rnd->GammaIntensities[i]);
        }
        add_assoc_zval(return_value, "XrayLines", &XrayLines);
        add_assoc_zval(return_value, "XrayIntensities", &XrayIntensities);
        add_assoc_zval(return_value, "GammaEnergies", &GammaEnergies);
        add_assoc_zval(return_value, "GammaIntensities", &GammaIntensities);
        FreeRadioNuclideData(rnd);
}
%typemap(out) struct compoundDataNIST * {
        int i;
        struct compoundDataNIST *cdn = $1;

        if (cdn == NULL) {
                php_log_err("Error: requested NIST compound not found in database\n");
                RETURN_NULL();
        }
        array_init(return_value);
        add_assoc_string(return_value, "name", cdn->name);
        add_assoc_long(return_value, "nElements", cdn->nElements);
        add_assoc_double(return_value, "density", cdn->density);
        zval Elements, massFractions;

        array_init(&Elements);
        array_init(&massFractions);
        for (i = 0 ; i < cdn->nElements ; i++) {
                add_index_long(&Elements, i, cdn->Elements[i]);
                add_index_double(&massFractions, i, cdn->massFractions[i]);
        }
        add_assoc_zval(return_value, "Elements", &Elements);
        add_assoc_zval(return_value, "massFractions", &massFractions);
        FreeCompoundDataNIST(cdn);
}
%typemap(out) struct compoundData * {
        int i;
        struct compoundData *cd = $1;

        if (cd == NULL) {
                php_log_err("CompoundParser Error\n");
                RETURN_NULL();
        }
        array_init(return_value);
        add_assoc_long(return_value, "nElements", cd->nElements);
        add_assoc_double(return_value, "nAtomsAll", cd->nAtomsAll);
        zval Elements, massFractions, nAtoms;

        array_init(&Elements);
        array_init(&massFractions);
        array_init(&nAtoms);
        for (i = 0 ; i < cd->nElements ; i++) {
                add_index_long(&Elements, i, cd->Elements[i]);
                add_index_double(&massFractions, i, cd->massFractions[i]);
                add_index_double(&nAtoms, i, cd->nAtoms[i]);
        }
        add_assoc_zval(return_value, "Elements", &Elements);
        add_assoc_zval(return_value, "massFractions", &massFractions);
        add_assoc_zval(return_value, "nAtoms", &nAtoms);
        add_assoc_double(return_value, "molarMass", cd->molarMass);
        FreeCompoundData(cd);
}
%typemap(out) xrlComplex {
        xrlComplex c = $1;

        array_init(return_value);
        add_assoc_double(return_value, "re", c.re);
        add_assoc_double(return_value, "im", c.im);
}
%typemap(in) xrlComplex {
        xrlComplex c;

        if (Z_TYPE($input) != IS_ARRAY) {
                SWIG_exception(SWIG_TypeError,"Argument must be an array");
        }
        if (!zend_hash_str_exists(Z_ARRVAL($input), "im", sizeof("im")-1)) {
                SWIG_exception(SWIG_TypeError, "re hash key not present");
        }
        if (!zend_hash_str_exists(Z_ARRVAL($input), "im", sizeof("im")-1)) {
                SWIG_exception(SWIG_TypeError, "im hash key not present");
        }
        zval *re, *im;
        re = zend_hash_str_find(Z_ARRVAL($input), "re", sizeof("re")-1);
        im = zend_hash_str_find(Z_ARRVAL($input), "im", sizeof("im")-1);
        convert_to_double_ex(re);
        convert_to_double_ex(im);
        c.re = Z_DVAL_P(re);
        c.im = Z_DVAL_P(im);
        $1 = c;
}
%typemap(out) Crystal_Struct * {
        Crystal_Struct *cs = $1;
        int i;
        if (cs == NULL) {
                php_log_err("Crystal_GetCrystal Error: crystal not found");
                RETURN_NULL();
        }

        array_init(return_value);
        add_assoc_string(return_value, "name", cs->name);
        add_assoc_double(return_value, "a", cs->a);
        add_assoc_double(return_value, "b", cs->b);
        add_assoc_double(return_value, "c", cs->c);
        add_assoc_double(return_value, "alpha", cs->alpha);
        add_assoc_double(return_value, "beta", cs->beta);
        add_assoc_double(return_value, "gamma", cs->gamma);
        add_assoc_double(return_value, "volume", cs->volume);
        add_assoc_long(return_value, "n_atom", cs->n_atom);
        zval atom;
        array_init(&atom);
        add_assoc_zval(return_value, "atom", &atom);
        for (i = 0 ; i < cs->n_atom ; i++) {
                zval dict_temp;
                array_init(&dict_temp);
                add_assoc_long(&dict_temp, "Zatom", cs->atom[i].Zatom);
                add_assoc_double(&dict_temp, "fraction", cs->atom[i].fraction);
                add_assoc_double(&dict_temp, "x", cs->atom[i].x);
                add_assoc_double(&dict_temp, "y", cs->atom[i].y);
                add_assoc_double(&dict_temp, "z", cs->atom[i].z);
                add_index_zval(&atom, i, &dict_temp);
        }
        add_assoc_long(return_value, "cpointer", (zend_long) cs);
}
%typemap(in) Crystal_Struct * {
        /* cpointer should be used if present and valid */

        if (Z_TYPE($input) != IS_ARRAY) {
                SWIG_exception(SWIG_TypeError,"Argument must be an array");
        }
        if (zend_hash_str_exists(Z_ARRVAL($input), "cpointer", sizeof("cpointer")-1)) {
                /* cpointer found */
                zval *cpointer;
                cpointer = zend_hash_str_find(Z_ARRVAL($input), "cpointer", sizeof("cpointer")-1);
                convert_to_long_ex(cpointer);
                Crystal_Struct *cs = (Crystal_Struct *) Z_LVAL_P(cpointer);
                $1 = cs;
        }
        else {
                Crystal_Struct *cs = malloc(sizeof(Crystal_Struct));
                zval *temp;
                if ((temp = zend_hash_str_find(Z_ARRVAL($input), "name", sizeof("name")-1)) == NULL) {
                        SWIG_exception(SWIG_TypeError,"Name key not present");
                }
                cs->name = strdup(Z_STRVAL_P(temp));

                if ((temp = zend_hash_str_find(Z_ARRVAL($input), "a", sizeof("a")-1)) == NULL) {
                        SWIG_exception(SWIG_TypeError,"a key not present");
                }
                convert_to_double_ex(temp);
                cs->a = Z_DVAL_P(temp);

                if ((temp = zend_hash_str_find(Z_ARRVAL($input), "b", sizeof("b")-1)) == NULL) {
                        SWIG_exception(SWIG_TypeError,"b key not present");
                }
                convert_to_double_ex(temp);
                cs->b = Z_DVAL_P(temp);

                if ((temp = zend_hash_str_find(Z_ARRVAL($input), "c", sizeof("c")-1)) == NULL) {
                        SWIG_exception(SWIG_TypeError,"c key not present");
                }
                convert_to_double_ex(temp);
                cs->c = Z_DVAL_P(temp);

                if ((temp = zend_hash_str_find(Z_ARRVAL($input), "alpha", sizeof("alpha")-1)) == NULL) {
                        SWIG_exception(SWIG_TypeError,"alpha key not present");
                }
                convert_to_double_ex(temp);
                cs->alpha = Z_DVAL_P(temp);

                if ((temp = zend_hash_str_find(Z_ARRVAL($input), "beta", sizeof("beta")-1)) == NULL) {
                        SWIG_exception(SWIG_TypeError,"beta key not present");
                }
                convert_to_double_ex(temp);
                cs->beta= Z_DVAL_P(temp);

                if ((temp = zend_hash_str_find(Z_ARRVAL($input), "gamma", sizeof("gamma")-1)) == NULL) {
                        SWIG_exception(SWIG_TypeError,"gamma key not present");
                }
                convert_to_double_ex(temp);
                cs->gamma = Z_DVAL_P(temp);

                if ((temp = zend_hash_str_find(Z_ARRVAL($input), "volume", sizeof("volume")-1)) == NULL) {
                        SWIG_exception(SWIG_TypeError,"volume key not present");
                }
                convert_to_double_ex(temp);
                cs->volume = Z_DVAL_P(temp);

                if ((temp = zend_hash_str_find(Z_ARRVAL($input), "n_atom", sizeof("n_atom")-1)) == NULL) {
                        SWIG_exception(SWIG_TypeError,"n_atom key not present");
                }
                convert_to_long_ex(temp);
                cs->n_atom = (int) Z_LVAL_P(temp);

                zval *atom;

                if ((atom = zend_hash_str_find(Z_ARRVAL($input), "atom", sizeof("atom")-1)) == NULL) {
                        SWIG_exception(SWIG_TypeError,"atom key not present");
                }

                if (Z_TYPE_P(atom) != IS_ARRAY) {
                        SWIG_exception(SWIG_TypeError,"atom must be an array");
                }
                int i;
                cs->atom = (Crystal_Atom *) malloc(sizeof(Crystal_Atom)*cs->n_atom);
                for (i = 0 ; i < cs->n_atom ; i++) {
                        zval *this_atom;
                        if ((this_atom = zend_hash_index_find(Z_ARRVAL_P(atom), i)) == NULL) {
                                SWIG_exception(SWIG_TypeError,"atom member not found\n");
                        }
                        zval *temp2;
                        if ((temp2 = zend_hash_str_find(Z_ARRVAL_P(this_atom), "Zatom", sizeof("Zatom")-1)) == NULL) {
                                SWIG_exception(SWIG_TypeError,"Zatom key not found\n");
                        }
                        convert_to_long_ex(temp2);
                        cs->atom[i].Zatom = (int) Z_LVAL_P(temp2);

                        if ((temp2 = zend_hash_str_find(Z_ARRVAL_P(this_atom), "fraction", sizeof("fraction")-1)) == NULL) {
                                SWIG_exception(SWIG_TypeError,"fraction key not found\n");
                        }
                        convert_to_double_ex(temp2);
                        cs->atom[i].fraction = (double) Z_DVAL_P(temp2);

                        if ((temp2 = zend_hash_str_find(Z_ARRVAL_P(this_atom), "x", sizeof("x")-1)) == NULL) {
                                SWIG_exception(SWIG_TypeError,"x key not found\n");
                        }
                        convert_to_double_ex(temp2);
                        cs->atom[i].x = (double) Z_DVAL_P(temp2);

                        if ((temp2 = zend_hash_str_find(Z_ARRVAL_P(this_atom), "y", sizeof("y")-1)) == NULL) {
                                SWIG_exception(SWIG_TypeError,"y key not found\n");
                        }
                        convert_to_double_ex(temp2);
                        cs->atom[i].y = (double) Z_DVAL_P(temp2);

                        if ((temp2 = zend_hash_str_find(Z_ARRVAL_P(this_atom), "z", sizeof("z")-1)) == NULL) {
                                SWIG_exception(SWIG_TypeError,"z key not found\n");
                        }
                        convert_to_double_ex(temp2);
                        cs->atom[i].z = (double) Z_DVAL_P(temp2);

                }
                $1 = cs;
        }

}
#endif


%include "xraylib.h"
