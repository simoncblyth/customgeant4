#pragma once
/**
MultiLayrStack.h : Thin/Thick Multi-layer Stack TMM "Transfer Matrix Method" A,R,T calculation 
=====================================================================================================

This was developed and tested as Layr.h at https://github.com/simoncblyth/j/blob/main/Layr/Layr.h

1. nothing JUNO specific here
2. header-only implementation


See Also tests and notes within https://github.com/simoncblyth/j/blob/main/Layr
----------------------------------------------------------------------------------

Layr.rst 
    notes/refs on TMM theory and CPU+GPU implementation 

LayrTest.{h,cc,cu,py,sh} 
    build, run cpu+gpu scans, plotting, comparisons float/double cpu/gpu std/thrust 

JPMT.h
    JUNO specific collection of PMT rindex and thickness into arrays 


Contents of Layr.h : (persisted array shapes)
----------------------------------------------

namespace Const 
    templated constexpr functions: zero, one, two, pi, twopi

template<typename T> struct Matx : (4,2)
    2x2 complex matrix  

template<typename T> struct Layr : (4,4,2)
    d:thickness, complex refractive index, angular and Fresnel coeff,  S+P Matx

    * d = zero : indicates thick (incoherent) layer 

template<typename F> struct ART_ : (4,4) 
    results  

template<typename T, int N> StackSpec :  (4,4) 
    4 sets of complex refractive index and thickness

template <typename T, int N> struct Stack : (constituents persisted separately) 
    N Layr stack : all calculations in ctor  

    Layr<T> ll[N] ;    (4,4,4,2)  (32,4)    when N=4                            
    Layr<T> comp ;     (1,4,4,2)  (8, 4)    composite for the N layers    (
    ART_<T>  art ;     (4,4)      (4, 4)
                                 --------
                                  (44,4)

    LAYR_METHOD Stack(T wl, T minus_cos_theta, const StackSpec4<T>& ss);

**/

#ifdef WITH_THRUST
#include <thrust/complex.h>
#else
#include <complex>
#include <cmath>
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <array>
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define LAYR_METHOD __host__ __device__ __forceinline__
#else
#    define LAYR_METHOD inline 
#endif


/**
Const
-------

The point of these Const functions is to plant the desired type of constant
into assembly code with no runtime transients of another type. Important for
avoiding expensive unintentional doubles in GPU code. The constexpr means that
the conversions and calculations happen at compile time, NOT runtime. 

**/

namespace Const
{
    template<typename T>  
    LAYR_METHOD constexpr T zero(){ return T(0.0) ; } 
 
    template<typename T>
    LAYR_METHOD constexpr T one() { return T(1.0) ; } 

    template<typename T>
    LAYR_METHOD constexpr T two() { return T(2.0) ; } 
    
    template<typename T>
    LAYR_METHOD constexpr T pi_() { return T(M_PI) ; } 

    template<typename T>
    LAYR_METHOD constexpr T twopi_() { return T(2.0*M_PI) ; } 
}


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    #ifdef WITH_THRUST
    template<typename T>
    inline std::ostream& operator<<(std::ostream& os, const thrust::complex<T>& z)  
    {
        os << "(" << std::setw(10) << std::fixed << std::setprecision(4) << z.real() 
           << " " << std::setw(10) << std::fixed << std::setprecision(4) << z.imag() << ")t" ; 
        return os; 
    }
    #else
    template<typename T>
    inline std::ostream& operator<<(std::ostream& os, const std::complex<T>& z)  
    {
        os << "(" << std::setw(10) << std::fixed << std::setprecision(4) << z.real() 
           << " " << std::setw(10) << std::fixed << std::setprecision(4) << z.imag() << ")s" ; 
        return os; 
    }
    #endif  // clarity is more important than brevity 
#endif

template<typename T>
struct Matx
{
#ifdef WITH_THRUST
    thrust::complex<T> M00, M01, M10, M11 ;   
#else
    std::complex<T>    M00, M01, M10, M11 ;       
#endif
    LAYR_METHOD void reset();             
    LAYR_METHOD void dot(const Matx<T>& other); 
};

template<typename T>
LAYR_METHOD void Matx<T>::reset()
{
    M00.real(1) ; M00.imag(0) ; // conversion from int 
    M01.real(0) ; M01.imag(0) ; 
    M10.real(0) ; M10.imag(0) ; 
    M11.real(1) ; M11.imag(0) ; 
}
/**

      | T00  T01  |  |  M00   M01 | 
      |           |  |            | 
      | T10  T11  |  |  M10   M11 | 

**/
template<typename T>
LAYR_METHOD void Matx<T>::dot(const Matx<T>& other)
{
#ifdef WITH_THRUST
    using thrust::complex ; 
#else
    using std::complex ; 
#endif
    complex<T> T00(M00) ; 
    complex<T> T01(M01) ; 
    complex<T> T10(M10) ; 
    complex<T> T11(M11) ; 

    M00 = T00*other.M00 + T01*other.M10;
    M01 = T00*other.M01 + T01*other.M11;
    M10 = T10*other.M00 + T11*other.M10;
    M11 = T10*other.M01 + T11*other.M11;
}

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Matx<T>& m)  
{
    os 
       << "| " << m.M00 << " " << m.M01 << " |" << std::endl 
       << "| " << m.M10 << " " << m.M11 << " |" << std::endl  
       ;
    return os; 
}
#endif


/**
Layr : (4,4,2) 
-----------------

The comp layers do not have the 0th (4,2) filled::

   assert np.all( e.f.comps[:,0] == 0 ) 

**/

template<typename T>
struct Layr
{
    // ---------------------------------------- 0th (4,2)
    T  d ;
    T  pad=0 ;
#ifdef WITH_THRUST 
    thrust::complex<T>  n, st, ct ; 
#else
    std::complex<T>     n, st, ct ;
#endif
    // ---------------------------------------- 1st (4,2)

#ifdef WITH_THRUST 
    thrust::complex<T>  rs, rp, ts, tp ;    
#else
    std::complex<T>     rs, rp, ts, tp ;    
#endif
    // ---------------------------------------- 2nd (4,2)
    Matx<T> S ;                             
    // ---------------------------------------- 3rd (4,2)
    Matx<T> P ;                               
    // ---------------------------------------- 

    LAYR_METHOD void reset(); 
    LAYR_METHOD void load4( const T* vals ); 
    LAYR_METHOD const T* cdata() const { return &d ; }
};

template<typename T>
LAYR_METHOD void Layr<T>::reset()
{
    S.reset(); 
    P.reset(); 
}

template<typename T>
LAYR_METHOD void Layr<T>::load4(const T* vals)
{
    d   = vals[0] ; 
    pad = vals[1] ; 
    n.real(vals[2]) ; 
    n.imag(vals[3]) ; 
}



#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Layr<T>& l)  
{
    os 
       << "Layr"
       << std::endl 
       << std::setw(4) << "n:" << l.n  
       << std::setw(4) << "d:" << std::fixed << std::setw(10) << std::setprecision(4) << l.d  
       << std::endl 
       << std::setw(4) << "st:" << l.st 
       << std::setw(4) << "ct:" << l.ct
       << std::endl 
       << std::setw(4) << "rs:" << l.rs 
       << std::setw(4) << "rp:" << l.rp
       << std::endl 
       << std::setw(4) << "ts:" << l.ts 
       << std::setw(4) << "tp:" << l.tp
       << std::endl 
       << "S" 
       << std::endl 
       << l.S 
       << std::endl 
       << "P"
       << std::endl 
       << l.P
       << std::endl 
       ;
    return os; 
}
#endif

template<typename F>
struct ART_
{   
    F R_s;     // R_s = a.arts[:,0,0]
    F R_p;     // R_p = a.arts[:,0,1]
    F T_s;     // T_s = a.arts[:,0,2]
    F T_p;     // T_p = a.arts[:,0,3]

    F A_s;     // A_s = a.arts[:,1,0]
    F A_p;     // A_p = a.arts[:,1,1]
    F R_av;    // R_av   = a.arts[:,1,2]
    F T_av;    // T_av   = a.arts[:,1,3]

    F A_av;    // A_av   = a.arts[:,2,0]
    F A_R_T_av ;  // A_R_T_av = a.arts[:,2,1] 
    F wl ;     // wl  = a.arts[:,2,2]
    F mct ;    // mct  = a.arts[:,2,3]   

    F A ;       // A = a.arts[:,3,0]
    F R ;       // R = a.arts[:,3,1]
    F T ;       // T = a.arts[:,3,2]
    F S ;       // S = a.arts[:,3,3]     S_pol vs P_pol power fraction 

    LAYR_METHOD const F* cdata() const { return &R_s ; }
    // persisted into shape (4,4) 
};

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template<typename F>
inline std::ostream& operator<<(std::ostream& os, const ART_<F>& art )  
{
    os 
        << "ART_" << std::endl 
        << " R_s " << std::fixed << std::setw(10) << std::setprecision(4) << art.R_s 
        << " R_p " << std::fixed << std::setw(10) << std::setprecision(4) << art.R_p 
        << std::endl 
        << " T_s " << std::fixed << std::setw(10) << std::setprecision(4) << art.T_s 
        << " T_p " << std::fixed << std::setw(10) << std::setprecision(4) << art.T_p 
        << std::endl 
        << " A_s " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_s 
        << " A_p " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_p 
        << std::endl 
        << " R_av " << std::fixed << std::setw(10) << std::setprecision(4) << art.R_av   
        << " T_av " << std::fixed << std::setw(10) << std::setprecision(4) << art.T_av   
        << " A_av " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_av  
        << " A_R_T_av " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_R_T_av 
        << std::endl 
        << " R " << std::fixed << std::setw(10) << std::setprecision(4) << art.R
        << " T " << std::fixed << std::setw(10) << std::setprecision(4) << art.T   
        << " A " << std::fixed << std::setw(10) << std::setprecision(4) << art.A  
        << " S " << std::fixed << std::setw(10) << std::setprecision(4) << art.S 
        << std::endl 
        << " wl  " << std::fixed << std::setw(10) << std::setprecision(4) << art.wl  << std::endl 
        << " mct " << std::fixed << std::setw(10) << std::setprecision(4) << art.mct << std::endl 
        ;
    return os; 
}
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
namespace sys
{
    template<typename F>
    inline std::vector<F>* getenvvec(const char* ekey, const char* fallback = nullptr, char delim=',')
    {
        char* _line = getenv(ekey);
        const char* line = _line ? _line : fallback ; 
        if(line == nullptr) return nullptr ; 

        std::stringstream ss; 
        ss.str(line);
        std::string s;

        std::vector<F>* vec = new std::vector<F>() ; 

        while (std::getline(ss, s, delim)) 
        {   
            std::istringstream iss(s);
            F f ; 
            iss >> f ; 
            vec->push_back(f) ; 
        }   
        return vec ; 
    }

    template<typename F, unsigned long N>
    inline F max_diff( const std::array<F,N>& a, const std::array<F,N>& b )
    {
        F mx = 0. ; 
        for(unsigned long i=0 ; i < N ; i++) 
        {
            F df = std::abs(a[i]-b[i]) ; 
            if(df > mx ) mx = df ; 
        } 
        return mx ; 
    }

    template<typename F, unsigned long N>
    inline std::string desc_diff( const std::array<F,N>& a, const std::array<F,N>& b )
    {
        std::stringstream ss ; 
        for(unsigned long i=0 ; i < N ; i++) 
        {
            ss 
                << " i " << std::setw(2) << i   
                << " a " << std::setw(10) << std::scientific << a[i]
                << " b " << std::setw(10) << std::scientific << b[i]
                << " a - b  " << std::setw(10) << std::scientific << ( a[i] - b[i] )
                << std::endl 
                ;   
        }
        std::string str = ss.str(); 
        return str ; 
    }

}
#endif


template<typename F>
struct LayrSpec
{
    F nr, ni, d, pad  ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void serialize(std::array<F,4>& a) const ; 
    static int EGet(LayrSpec<F>& ls, int idx); 
#endif
};

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template<typename F>
LAYR_METHOD void LayrSpec<F>::serialize(std::array<F,4>& a) const 
{
    a[0] = nr ; 
    a[1] = ni ; 
    a[2] = d ; 
    a[3] = pad  ;  
}

template<typename F>
LAYR_METHOD int LayrSpec<F>::EGet(LayrSpec<F>& ls, int idx)
{
    std::stringstream ss ; 
    ss << "L" << idx ; 
    std::string ekey = ss.str(); 
    std::vector<F>* vls = sys::getenvvec<F>(ekey.c_str()) ; 
    if(vls == nullptr) return 0 ; 
    const F zero(0) ; 
    ls.nr = vls->size() > 0u ? (*vls)[0] : zero ; 
    ls.ni = vls->size() > 1u ? (*vls)[1] : zero ; 
    ls.d  = vls->size() > 2u ? (*vls)[2] : zero ; 
    ls.pad = zero ; 
    return 1 ; 
}

template<typename F>
LAYR_METHOD std::ostream& operator<<(std::ostream& os, const LayrSpec<F>& ls )  
{
    os 
        << "LayrSpec<" << ( sizeof(F) == 8 ? "double" : "float" ) << "> "  
        << std::fixed << std::setw(10) << std::setprecision(4) << ls.nr << " "
        << std::fixed << std::setw(10) << std::setprecision(4) << ls.ni << " ; "
        << std::fixed << std::setw(10) << std::setprecision(4) << ls.d  << ")"
        << std::endl 
        ;
    return os ; 
}
#endif


template<typename F, int N>  
struct StackSpec
{
    LayrSpec<F> ls[N] ; 
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    LAYR_METHOD void eget() ; 
    LAYR_METHOD F* data() const ; 
    LAYR_METHOD const F* cdata() const ; 
    LAYR_METHOD void serialize(std::array<F,N*4>& a) const ; 
    LAYR_METHOD void import(const std::array<F,N*4>& a) ;  
    LAYR_METHOD bool is_equal( const StackSpec<F,N>& other ) const ; 
    LAYR_METHOD std::string desc_compare( const StackSpec<F,N>& other ) const  ; 
#endif

}; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template<typename F, int N>
LAYR_METHOD void StackSpec<F,N>::eget()  
{
    int count = 0 ; 
    for(int i=0 ; i < N ; i++) count += LayrSpec<F>::EGet(ls[i], i); 
    assert( count == N ) ; 
}

template<typename F, int N>
LAYR_METHOD F* StackSpec<F,N>::data() const 
{
    return (F*)&(ls[0].nr) ; 
}

template<typename F, int N>
LAYR_METHOD const F* StackSpec<F,N>::cdata() const 
{
    return (F*)&(ls[0].nr) ; 
}

template<typename F, int N>
LAYR_METHOD void StackSpec<F,N>::serialize(std::array<F,N*4>& a) const 
{
    for(int i=0 ; i < N ; i++)
    {
        std::array<F,4> ls_i ; 
        ls[i].serialize(ls_i) ; 
        for(int j=0 ; j < 4 ; j++) a[i*4+j] = ls_i[j] ;   
    }
}

template<typename F, int N>
LAYR_METHOD void StackSpec<F,N>::import(const std::array<F,N*4>& a) 
{
    memcpy( data(), a.data(), a.size()*sizeof(F) );  
}

template<typename F, int N>
LAYR_METHOD bool StackSpec<F,N>::is_equal( const StackSpec<F,N>& other ) const 
{
    std::array<F, N*4> a ; 
    serialize(a) ; 

    std::array<F, N*4> b ; 
    other.serialize(b); 

    return a == b ; 
}


template<typename F, int N>
LAYR_METHOD std::string StackSpec<F,N>::desc_compare( const StackSpec<F,N>& other ) const 
{
    std::array<F, N*4> a ; 
    serialize(a) ; 

    std::array<F, N*4> b ; 
    other.serialize(b); 

    F mx = sys::max_diff(a, b) ; 

    std::stringstream ss ; 
    ss << "StackSpec<" << ( sizeof(F) == 8 ? "double" : "float" ) << "," << N << ">::desc_compare " ;  
    ss << " is_equal " << ( is_equal(other) ? "YES" : "NO"  ) ; 
    ss << " max_diff " << std::setw(10) << std::scientific << mx ; 
    ss << std::endl  ; 

    for(int i=0 ; i < 4*N ; i++) ss 
        << ( i % 4 == 0  ? "\n" : " " ) 
        << ( a[i] == b[i] ? " " : "[" ) 
        << std::fixed << std::setw(10) << std::setprecision(4) << a[i] << " " 
        << std::fixed << std::setw(10) << std::setprecision(4) << b[i] << " " 
        << ( a[i] == b[i] ? " " : "]" ) 
        ;

    std::string str = ss.str(); 
    return str ; 
}







template<typename F>
LAYR_METHOD std::ostream& operator<<(std::ostream& os, const std::array<F,16>& aa )
{
    // curiously fails to template match with generic int N  template param
    os << std::endl ; 
    for(int i=0 ; i < 16 ; i++) os 
        << ( i % 4 == 0 ? '\n' : ' ' ) 
        << std::setw(10) << std::fixed << std::setprecision(4) << aa[i] 
        ;   
    os << std::endl ; 
    return os ; 
}

template<typename F, int N>
LAYR_METHOD std::ostream& operator<<(std::ostream& os, const StackSpec<F,N>& ss )  
{
    os 
        << std::endl 
        << "StackSpec<" 
        << ( sizeof(F) == 8 ? "double" : "float" ) 
        << "," << N
        << ">"  
        << std::endl ;

    for(int i=0 ; i < N ; i++) os << ss.ls[i] ; 
    return os ; 
}

#endif


/**
Stack
-------

**/

template <typename F, int N>
struct Stack
{
    Layr<F> ll[N] ; //  (4,4,4,2)     when N=4  
    Layr<F> comp ;  //  (1,4,4,2)     composite for the N layers 
    ART_<F>  art ;  //  (4,4) 

    LAYR_METHOD void zero();
    LAYR_METHOD Stack();
    LAYR_METHOD Stack(F wl, F minus_cos_theta, F dot_pol_cross_mom_nrm, const StackSpec<F,N>& ss);
    LAYR_METHOD void calc(F wl, F minus_cos_theta, F dot_pol_cross_mom_nrm, const StackSpec<F,N>& ss);

    LAYR_METHOD const F* cdata() const { return ll[0].cdata() ; }

};

template<typename F, int N>
LAYR_METHOD void Stack<F,N>::zero()
{
    art = {} ; 
    comp = {} ; 
    for(int i=0 ; i < N ; i++) ll[i] = {} ; 
}


/**
Stack::Stack
---------------

Caution that StackSpec contains refractive indices that depend on wavelength, 
so the wavelength dependency enters twice. 

*minus_cos_theta* which is "dot(photon_momentum,outward_surface_normal)" 
is used as a more physical "angle" parameter, it corresponds to -cos(aoi)

1. minus_cos_theta = -1.f at normal incidence against surface_normal, inwards going 
2. minus_cos_theta = +1.f at normal incidence with the surface_normal, outwards going  
3. minus_cos_theta =  0.f at glancing incidence (90 deg AOI) : potential for math blowouts here 
4. sign of dot product indicates when must flip the stack of parameters
5. angle scan plots can then use aoi 0->180 deg, which is -cos(aoi) -1->1, there 
   should be continuity across the flip otherwise something very wrong

**/

template<typename F, int N>
LAYR_METHOD Stack<F,N>::Stack()
{
    zero(); 
}

/**
Stack::Stack ctor : does TMM A,R,T calc
------------------------------------------

1. Layr populate : ll[0..N-1].n ll[0..N-1].d by copying in from StackSpec

   * HMM: could copying from stackspec be avoided ? Not easily as copying into complex rindex  

2. l0.ct l0.st from minus_cos_theta 

**/

template<typename F, int N>
LAYR_METHOD Stack<F,N>::Stack(F wl, F minus_cos_theta, F dot_pol_cross_mom_nrm, const StackSpec<F,N>& ss ) 
{
    calc(wl, minus_cos_theta, dot_pol_cross_mom_nrm, ss ); 
}

template<typename F, int N>
void LAYR_METHOD Stack<F,N>::calc(F wl, F minus_cos_theta, F dot_pol_cross_mom_nrm, const StackSpec<F,N>& ss )
{
#ifdef WITH_THRUST
    using thrust::complex ; 
    using thrust::norm ; 
    using thrust::conj ; 
    using thrust::exp ; 
    using thrust::sqrt ; 
    using thrust::sin ; 
    using thrust::cos ; 
#else
    using std::complex ; 
    using std::norm ; 
    using std::conj ; 
    using std::exp ; 
    using std::sqrt ; 
    using std::sin ; 
    using std::cos ; 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    assert( N >= 2); 
#endif

    const F zero(Const::zero<F>()) ; 
    const F one(Const::one<F>()) ; 
    const F two(Const::two<F>()) ; 
    const F twopi_(Const::twopi_<F>()) ; 

    for(int i=0 ; i < N ; i++)
    {
        int j = minus_cos_theta < zero ? i : N - 1 - i ;  
        //  minus_cos_theta < zero  : against normal : ordinary stack  : j = i 
        //  minus_cos_theta >= zero : with normal    : backwards stack : j from end 

        ll[j].n.real(ss.ls[i].nr) ; 
        ll[j].n.imag(ss.ls[i].ni) ; 
        ll[j].d = ss.ls[i].d ; 
    }

    // NB stack flipping based on sign of minus_cos_theta ensures consistent layer ordering
    // 
    // ll[0]   is always "top"     : start layer : incident
    // ll[N-1] is always "bottom"  : end   layer : transmitted
    //

    art.wl = wl ; 
    art.mct = minus_cos_theta ; 

    const complex<F> zOne(one,zero); 
    const complex<F> zI(zero,one); 
    const complex<F> mct(minus_cos_theta);  // simpler for everything to be complex

    // Snell : set st,ct of all layers (depending on indices(hence wl) and incident angle) 
    Layr<F>& l0 = ll[0] ; 
    l0.ct = minus_cos_theta < zero  ? -mct : mct ; 
    //
    //  flip picks +ve ct that constrains the angle to first quadrant 
    //  this works as : cos(pi-theta) = -cos(theta)
    //  without flip, the ART values are non-physical : always outside 0->1 for mct > 0 angle > 90  
    //

    const F stst = one - minus_cos_theta*minus_cos_theta ; 
    l0.st.real(sqrt(stst)) ;  

    const F E_s2 = stst > zero ? (dot_pol_cross_mom_nrm*dot_pol_cross_mom_nrm)/stst : zero ;
    // E_s2 is the S_polarization vs P_polarization power fraction  
    // E_s2 matches E1_perp*E1_perp see sysrap/tests/stmm_vs_sboundary_test.cc 


    for(int idx=1 ; idx < N ; idx++)  // for N=2 idx only 1, sets only ll[1] 
    {
        Layr<F>& l = ll[idx] ; 
        l.st = l0.n * l0.st / l.n  ; 
        l.ct = sqrt( zOne - l.st*l.st );
    }     

    // Fresnel : set rs/rp/ts/tp for N-1 interfaces between N layers
    // (depending on indices(hence wl) and layer ct) 
    // HMM: last layer unset, perhaps zero it ?
    for(int idx=0 ; idx < N-1 ; idx++)
    {
        // cf OpticalSystem::Calculate_rt  
        // https://en.wikipedia.org/wiki/Fresnel_equations
        Layr<F>& i = ll[idx] ; 
        const Layr<F>& j = ll[idx+1] ;  

        i.rs = (i.n*i.ct - j.n*j.ct)/(i.n*i.ct+j.n*j.ct) ;  // r_s eoe[12] see g4op-eoe
        i.rp = (j.n*i.ct - i.n*j.ct)/(j.n*i.ct+i.n*j.ct) ;  // r_p eoe[7]
        i.ts = (two*i.n*i.ct)/(i.n*i.ct + j.n*j.ct) ;       // t_s eoe[9]
        i.tp = (two*i.n*i.ct)/(j.n*i.ct + i.n*j.ct) ;       // t_p eoe[8]
    }

    // populate transfer matrices for both thick and thin layers  
    // for N=2 only one interface

    ll[0].reset();    // ll[0].S ll[0].P matrices set to identity 

    for(int idx=1 ; idx < N ; idx++)
    {
        const Layr<F>& i = ll[idx-1] ;            
        Layr<F>& j       = ll[idx] ;          
        // looking at (i,j) pairs of layers 

        complex<F> tmp_s = one/i.ts ; 
        complex<F> tmp_p = one/i.tp ;   
        // at glancing incidence ts, tp approach zero : blowing up tmp_s tmp_p
        // which causes the S and P matrices to blow up yielding infinities at mct zero
        //
        // thick layers indicated with d = 0. 
        // thin layers have thickness presumably comparable to art.wl (WITH SAME LENGTH UNIT: nm)
        complex<F> delta         = j.d == zero ? zero : twopi_*j.n*j.d*j.ct/art.wl ; 
        complex<F> exp_neg_delta = j.d == zero ? one  : exp(-zI*delta) ; 
        complex<F> exp_pos_delta = j.d == zero ? one  : exp( zI*delta) ; 

        j.S.M00 = tmp_s*exp_neg_delta      ; j.S.M01 = tmp_s*i.rs*exp_pos_delta ; 
        j.S.M10 = tmp_s*i.rs*exp_neg_delta ; j.S.M11 =      tmp_s*exp_pos_delta ; 

        j.P.M00 = tmp_p*exp_neg_delta      ; j.P.M01 = tmp_p*i.rp*exp_pos_delta ; 
        j.P.M10 = tmp_p*i.rp*exp_neg_delta ; j.P.M11 =      tmp_p*exp_pos_delta ; 

        // NB: for thin layers the transfer matrices combine interface and propagation between them 
    }

    /*
    For d=0 (thick layer) 


    S     |  1/ts    rs/ts |    
          |                |
          |  rs/ts   1/ts  |     


    P     |  1/tp    rp/tp |    
          |                |
          |  rp/tp   1/tp  |     

    */



    // product of the transfer matrices
    comp.d = zero ; 
    comp.st = zero ; 
    comp.ct = zero ; 
    comp.S.reset(); 
    comp.P.reset(); 

    for(int idx=0 ; idx < N ; idx++) // TODO: start from idx=1 as ll[0].S ll[0].P always identity
    {
        const Layr<F>& l = ll[idx] ; 
        comp.S.dot(l.S) ; 
        comp.P.dot(l.P) ; 
    }
    // at glancing incidence the matrix from 
    // one of the layers has infinities, which 
    // yields nan in the matrix product 
    // and yields nan for all the below Fresnel coeffs 
    //
    // extract amplitude factors from the composite matrix
    comp.rs = comp.S.M10/comp.S.M00 ; 
    comp.rp = comp.P.M10/comp.P.M00 ; 
    comp.ts = one/comp.S.M00 ; 
    comp.tp = one/comp.P.M00 ; 

    Layr<F>& t = ll[0] ; 
    Layr<F>& b = ll[N-1] ; 

    // getting from amplitude to power relations for relectance R (same material and angle) and tranmittance T
    //  https://www.brown.edu/research/labs/mittleman/sites/brown.edu.research.labs.mittleman/files/uploads/lecture13_0.pdf

    complex<F> _T_s = (b.n*b.ct)/(t.n*t.ct)*norm(comp.ts) ;  
    complex<F> _T_p = (conj(b.n)*b.ct)/(conj(t.n)*t.ct)*norm(comp.tp) ; 
    // _T_p top and bot layers usually with real index ? so the conj above just noise ?

    art.R_s = norm(comp.rs) ; 
    art.R_p = norm(comp.rp) ; 
    art.T_s = _T_s.real() ; 
    art.T_p = _T_p.real() ; 

    // absorption factor by subtracting reflection and transmission
    art.A_s = one-art.R_s-art.T_s;
    art.A_p = one-art.R_p-art.T_p;

    // average of S and P 
    art.A_av   = (art.A_s+art.A_p)/two ;
    art.R_av   = (art.R_s+art.R_p)/two ;
    art.T_av   = (art.T_s+art.T_p)/two ;
    art.A_R_T_av = art.A_av + art.R_av + art.T_av ;  

    const F& S_frac = E_s2 ;
    const F P_frac = one - S_frac ;  

    // combination of S and P appropriate for (minus_cos_theta, dot_pol_cross_mom_nrm )
    art.A = S_frac*art.A_s + P_frac*art.A_p ; 
    art.R = S_frac*art.R_s + P_frac*art.R_p ; 
    art.T = S_frac*art.T_s + P_frac*art.T_p ;  
    art.S = S_frac ;
}

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template <typename F, int N>
inline std::ostream& operator<<(std::ostream& os, const Stack<F,N>& stk )  
{
    os << "Stack"
       << "<" 
       << ( sizeof(F) == 8 ? "double" : "float" )
       << ","
       << N 
       << ">" 
       << std::endl
       ; 
    for(int idx=0 ; idx < N ; idx++) os << "idx " << idx << std::endl << stk.ll[idx] ; 
    os << "comp" 
       << std::endl 
       << stk.comp 
       << std::endl 
       << stk.art 
       << std::endl 
       ; 
    return os; 
}
#endif

