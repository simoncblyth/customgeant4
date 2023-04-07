#pragma once
/**
C4GS.h : Genstep Label : Equivalent to sysrap/sgs.h
------------------------------------------------------
**/


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define C4GS_METHOD __host__ __device__ __forceinline__
#else
#    define C4GS_METHOD inline 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
     #include <string>
     #include "C4Pho.h"
#endif


struct C4GS
{
    int index ;     // 0-based index of genstep in the event 
    int photons ;   // number of photons in the genstep
    int offset ;    // photon offset in the sequence of gensteps, ie number of photons in event before this genstep
    int gentype  ;  // OpticksGenstep_ enum 
   
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    static C4GS  Make(int _index, int _photons, int _offset, int _gentype); 
    C4Pho MakePho(unsigned idx, const C4Pho& ancestor); 
    std::string desc() const ; 
#endif
}; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <sstream>
#include <iomanip>

/**
C4GS::Make
-----------

**/

inline C4GS C4GS::Make( int _index, int _photons, int _offset, int _gentype ) // static
{
    C4GS gs = { _index, _photons, _offset, _gentype } ; 
    return gs ; 
}


/**
C4GS::MakePho
-------------

ancestor.isDefined:false
    the more common case, when generating primary optical 
    photons via the Cerenkov or Scintillation processes 

    HMM: "ancestor" should more correctly be called "reemissionAncestorPhoton"

ancestor.isDefined:true
    when a photon undergoes reemission the ancestor is the parent photon

**/

inline C4Pho C4GS::MakePho(unsigned idx, const C4Pho& ancestor)
{
    return ancestor.isDefined() ? ancestor.make_nextgen() : C4Pho::MakePho(index, idx, offset + idx ) ; 
}

inline std::string C4GS::desc() const 
{
    std::stringstream ss ; 
    ss << "C4GS:"
       << " idx" << std::setw(4) << index 
       << " pho" << std::setw(6) << photons 
       << " off " << std::setw(6) << offset 
       << " typ " << std::setw(4) << gentype
       ;   
    std::string s = ss.str(); 
    return s ; 
}
#endif

