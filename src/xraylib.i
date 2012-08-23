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

%module xraylib
%include "typemaps.i"
%include "exception.i"

%apply float *OUTPUT { float* f0, float* f_primep, float* f_prime2 }




%{
#ifdef SWIGPYTHON
#undef c_abs
#endif
#include "xraylib.h"
%}

#ifndef SWIGLUA
%ignore c_abs;
%ignore c_mul;
#endif

%ignore Crystal_AddCrystal;
%ignore Crystal_ReadFile;
%ignore Crystal_Array;



#ifndef SWIGJAVA
%typemap(in,numinputs=0) struct compoundData* (struct compoundData temp) {
        temp.nElements=0;
        temp.nAtomsAll=0;
        temp.Elements=NULL;
        temp.massFractions=NULL;
        $1 = &temp;
}
%typemap(in, numinputs=0) Crystal_Array* c_array {
   /* do not use crystal_array argument for now... */
   $1 = NULL;
}
#endif

#ifdef SWIGLUA
%typemap(argout) struct compoundData * cd {
        int i;
        struct compoundData *cd = $1;

        if (cd->nElements == 0) {
                lua_pushnil(L);
                SWIG_arg++;
        }
        else {
                lua_newtable(L);

                lua_pushstring(L, "nElements");
                lua_pushinteger(L, cd->nElements);
                lua_settable(L,-3);

                lua_pushstring(L, "nAtomsAll");
                lua_pushinteger(L, cd->nAtomsAll);
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

                lua_pushvalue(L, -1);

                SWIG_arg++;
        }
}

%typemap(out) Complex {
        Complex c = $1;

        lua_newtable(L);

        lua_pushstring(L, "re");
        lua_pushnumber(L, c.re);
        lua_settable(L,-3);

        lua_pushstring(L, "im");
        lua_pushnumber(L, c.im);
        lua_settable(L,-3);

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

%typemap(in) Complex {
        Complex c;
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
                        size_t n_atom = lua_objlen(L, -1);
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
%typemap(argout) struct compoundData * cd {
        PyObject *dict = PyDict_New();
        int i;
        struct compoundData *cd = $1;
        if (cd->nElements > 0) {
                PyDict_SetItemString(dict, "nElements",PyInt_FromLong((long) cd->nElements)); 
                PyDict_SetItemString(dict, "nAtomsAll",PyInt_FromLong((long) cd->nAtomsAll)); 
                PyObject *elements=PyList_New(cd->nElements);
                PyObject *massfractions=PyList_New(cd->nElements);
                for (i=0 ; i < cd->nElements ; i++) {
                        PyObject *o = PyInt_FromLong((long) cd->Elements[i]);
                        PyList_SetItem(elements, i, o);
                        o = PyFloat_FromDouble((double) cd->massFractions[i]);
                        PyList_SetItem(massfractions, i, o);
                }
                PyDict_SetItemString(dict, "Elements", elements); 
                PyDict_SetItemString(dict, "massFractions", massfractions); 
                free(cd->Elements);
                free(cd->massFractions);
                $result=dict;
        }
        else {
                PyErr_WarnEx(NULL, "CompoundParser Error",1);
                $result=Py_None;
                goto fail;
        }

}



%typemap(out) Complex {
        Complex c = $1;
        PyObject *cp = PyComplex_FromDoubles((double) c.re, (double) c.im);

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
             PyDict_SetItemString(dict, "a",PyFloat_FromDouble((double) cs->a)); 
             PyDict_SetItemString(dict, "b",PyFloat_FromDouble((double) cs->b)); 
             PyDict_SetItemString(dict, "c",PyFloat_FromDouble((double) cs->c)); 
             PyDict_SetItemString(dict, "alpha",PyFloat_FromDouble((double) cs->alpha)); 
             PyDict_SetItemString(dict, "beta",PyFloat_FromDouble((double) cs->beta)); 
             PyDict_SetItemString(dict, "gamma",PyFloat_FromDouble((double) cs->gamma)); 
             PyDict_SetItemString(dict, "volume",PyFloat_FromDouble((double) cs->volume)); 
             PyDict_SetItemString(dict, "n_atom",PyInt_FromLong((int) cs->n_atom)); 
             PyObject *atom = PyList_New(cs->n_atom);
             PyDict_SetItemString(dict, "atom", atom); 
             for (i = 0 ; i < cs->n_atom ; i++) {
                PyObject *dict_temp = PyDict_New();
                PyDict_SetItemString(dict_temp, "Zatom",PyInt_FromLong((int) cs->atom[i].Zatom)); 
                PyDict_SetItemString(dict_temp, "fraction",PyFloat_FromDouble((double) cs->atom[i].fraction)); 
                PyDict_SetItemString(dict_temp, "x",PyFloat_FromDouble((double) cs->atom[i].x)); 
                PyDict_SetItemString(dict_temp, "y",PyFloat_FromDouble((double) cs->atom[i].y)); 
                PyDict_SetItemString(dict_temp, "z",PyFloat_FromDouble((double) cs->atom[i].z)); 
                PyList_SetItem(atom, i, dict_temp);
             }
             /* store cpointer in dictionary */
             PyDict_SetItemString(dict, "cpointer",PyInt_FromLong((long) cs)); 

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
                        long cpointer_l = PyInt_AsLong(cpointer); 
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
                         cs = (Crystal_Struct *) malloc(sizeof(Crystal_Struct));
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



%typemap(argout) struct compoundData * cd {
        int i;
        struct compoundData *cd = $1;
        if (argvi >= items) {
                EXTEND(sp,1);
        }
        if (cd->nElements > 0) {
                HV *hash = newHV();
                STORE_HASH("nElements", newSViv(cd->nElements),hash)
                STORE_HASH("nAtomsAll", newSViv(cd->nAtomsAll),hash)
                AV *Elements = newAV();
                AV *massFractions = newAV();
                STORE_HASH("Elements", newRV_noinc((SV*) Elements),hash)
                STORE_HASH("massFractions", newRV_noinc((SV*) massFractions),hash)
                for (i = 0 ; i < cd->nElements ; i++) {
                        av_push(Elements, newSViv(cd->Elements[i]));
                        av_push(massFractions, newSVnv(cd->massFractions[i]));
                }

                $result = sv_2mortal(newRV_noinc((SV*) hash));
        }
        else {
                $result = &PL_sv_undef;
        }
        
        argvi++;

}

%typemap(out) Complex {
        Complex c = $1;
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
                fprintf(stdout,"Crystal_GetCrystal Error: crystal not found");
                $result = &PL_sv_undef;
        }
        else {
                HV *rv = newHV();
                hv_store(rv, "name", 4, newSVpvn(cs->name,strlen(cs->name)),0);
                hv_store(rv, "a", 1, newSVnv(cs->a),0);
                hv_store(rv, "b", 1, newSVnv(cs->b),0);
                hv_store(rv, "c", 1, newSVnv(cs->c),0);
                hv_store(rv, "alpha", 5, newSVnv(cs->alpha),0);
                hv_store(rv, "beta", 4, newSVnv(cs->beta),0);
                hv_store(rv, "gamma", 5, newSVnv(cs->gamma),0);
                hv_store(rv, "volume", 6, newSVnv(cs->volume),0);
                hv_store(rv, "n_atom", 6, newSViv(cs->n_atom),0);
                AV *atoms = newAV();
                hv_store(rv, "atom", 4, newRV_noinc((SV*) atoms),0);
                for (i = 0 ; i < cs->n_atom ; i++) {
                        HV *atom = newHV();
                        hv_store(atom, "Zatom", 5, newSViv(cs->atom[i].Zatom),0);
                        hv_store(atom, "fraction", 8, newSVnv(cs->atom[i].fraction),0);
                        hv_store(atom, "x", 1, newSVnv(cs->atom[i].x),0);
                        hv_store(atom, "y", 1, newSVnv(cs->atom[i].y),0);
                        hv_store(atom, "z", 1, newSVnv(cs->atom[i].z),0);
                        av_push(atoms, newRV_noinc((SV*) atom));
                }
                hv_store(rv, "cpointer", 8, newSViv((IV) cs),0);
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




%include "xraylib.h"





