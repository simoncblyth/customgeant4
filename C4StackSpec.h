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
    if( count != N ) std::cerr 
         << "StackSpec::eget"
         << " count " << count
         << " N " << N 
         << " : MISSING CONFIG : eget requires envvars such as L0=1,0,1 L1=1.5,0,0 etc.. for each layer "
         << std::endl
          ; 
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

