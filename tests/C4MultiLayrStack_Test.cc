
#include "C4MultiLayrStack.h"

template<typename F>
void test_stack()
{
    F ss[16] = 
      {
        1.482,     0,      0,   0,
        1.920,     0,  36.49,   0,
        2.429, 1.366,  21.13,   0,
        1.000,     0,      0,   0  
      }; 

    F wl = 440. ; 
    F minus_cos_theta = -1. ; 
    F dot_pol_cross_mom_nrm = 0. ; 
    
    Stack<F,4> stack  ; 
    stack.calc( wl, minus_cos_theta, dot_pol_cross_mom_nrm, ss, 16u ); 

    //std::cout << stack ; 
    std::cout << stack.art ; 
}

int main(int argc, char** argv)
{
    test_stack<double>(); 
    //test_stack<float>(); 
    return 0 ; 
}


