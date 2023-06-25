#include "C4StackSpec.h"

int main(int argc, char** argv)
{
    StackSpec<float,4> ss ; 
    ss.eget(); 

    std::cout << ss << std::endl ; 

    return 0 ; 
}
