#pragma once

#ifdef WITH_THRUST
#include <thrust/complex.h>
#else
#include <complex>
#include <cmath>
#endif

#include "stdio.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define COMPLEX_TEST_METHOD __host__ __device__ __forceinline__
#else
#    define COMPLEX_TEST_METHOD inline 
#endif


struct complex_Test
{
   COMPLEX_TEST_METHOD static void check(float x, float y); 
};

COMPLEX_TEST_METHOD void complex_Test::check(float x, float y)
{
#ifdef WITH_THRUST
    using thrust::complex ; 
    using thrust::sqrt ; 
#else
    using std::complex ; 
    using std::sqrt ; 
#endif
    complex<float> z(x, y); 
    complex<float> s = sqrt(z) ; 

    const float stst = 0.5f ;
    const float st = sqrt(stst) ; 


    printf("//complex_Test::check z(%10.3f,%10.3f) s(%10.3f,%10.3f) st %10.3f \n",
         z.real(), z.imag(), s.real(), s.imag(), st );   
}
