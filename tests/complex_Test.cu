#include "stdio.h"
#include <cassert>
#include "complex_Test.h"

#define API  __attribute__ ((visibility ("default")))

__global__ void complex_Test_kernel(int width, int height)
{ 
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    float x = float(ix)/float(width) ; 
    float y = float(iy)/float(height) ; 

    complex_Test::check(x, y); 
} 

void ConfigureLaunch2D( dim3& numBlocks, dim3& threadsPerBlock, int width, int height ) // static
{
    threadsPerBlock.x = 16 ; 
    threadsPerBlock.y = 16 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 
}

void complex_Test_launch(int width, int height)
{
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    ConfigureLaunch2D(numBlocks, threadsPerBlock, width, height );  
    complex_Test_kernel<<<numBlocks,threadsPerBlock>>>(width, height) ; 
}

