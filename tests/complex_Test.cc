/**
complex_Test.cc 
================

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    #include "NP.hh"
    #ifdef WITH_THRUST
        #include "SU.hh"
        #include <cuda_runtime.h>
    #endif
#endif


extern void complex_Test_launch(int width, int height);

int main(int argc, char** argv)
{
#ifdef WITH_THRUST
    std::cout << "WITH_THRUST" << std::endl ; 
    complex_Test_launch(10, 10) ; 
    cudaDeviceSynchronize();
#else
    std::cout << "not-WITH_THRUST" << std::endl ; 
#endif    

    return 0 ; 
}
