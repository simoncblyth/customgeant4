// name=V_test ; gcc $name.cc -std=c++11 -lstdc++ -I. -o /tmp/$name && /tmp/$name

#include <cassert>
#include <iostream>
#include <cmath>

#include "V.h"


template<typename F>
void test_dot()
{
    std::array<F,3> a = {{ 1., 2., 3. }} ; 
    std::array<F,3> b = {{ 10.,20., 30. }} ; 
 
    F a_dot_b = V::dot<F>( a, b ); 
    assert( a_dot_b == 10. + 40. + 90. ) ; 
}


template<typename F>
void test_cross_(const std::array<F,3>& c0, const std::array<F,3>& a, const std::array<F,3>& b )
{
    enum { X, Y, Z } ; 


    std::array<F,3> c = {{0., 0., 0. }} ;  
    V::cross<F>( c, a, b ); 

    std::cout << " a  " << V::desc<F>(a) << std::endl ; 
    std::cout << " b  " << V::desc<F>(b) << std::endl ; 
    std::cout << " c  " << V::desc<F>(c) << std::endl ; 
    std::cout << " c0 " << V::desc<F>(c0) << std::endl ; 

    assert( c[X] ==  (a[Y]*b[Z] - a[Z]*b[Y]) ) ; 
    assert( c[Y] == -(a[X]*b[Z] - a[Z]*b[X]) ) ; 
    assert( c[Z] ==  (a[X]*b[Y] - a[Y]*b[X]) ) ; 

    assert( c[X] == c0[X] ); 
    assert( c[Y] == c0[Y] ); 
    assert( c[Z] == c0[Z] ); 
}

/**

Right hand basis

     Z
     |  Y
     | /
     |/
     +----- X

**/

template<typename F>
void test_cross()
{
    {
        std::array<F,3> a = {{ 1., 2., 3. }} ; 
        std::array<F,3> b = {{ 10.,20., 30. }} ; 
        std::array<F,3> c = {{ 0., 0., 0. }} ;    // all zero because a and b are collinear
        test_cross_<F>(c, a, b ); 
    }
    {
        // X ^ Y = Z 
        std::array<F,3> a = {{ 1., 0., 0. }} ; 
        std::array<F,3> b = {{ 0., 1., 0. }} ; 
        std::array<F,3> c = {{ 0., 0., 1. }} ;  
        test_cross_<F>(c, a, b ); 
    }
    {
        // Y ^ X = -Z 
        std::array<F,3> a = {{ 0., 1.,  0. }} ; 
        std::array<F,3> b = {{ 1., 0.,  0. }} ; 
        std::array<F,3> c = {{ 0., 0., -1. }} ;  
        test_cross_<F>(c, a, b ); 
    }
}


/**
test_rotateUz
----------------

Consider mom is some direction, say +Z::

   (0, 0, 1)

There is a circle of vectors that are perpendicular 
to that mom, all in the XY plane, and with dot product zero::

   ( cos(phi), sin(phi), 0 )    phi 0->2pi 

**/

template<typename F>
void test_rotateUz()
{

    const F zero(0.); 
    std::array<F,3> u  = {{ 0., 0.,  0. }} ; 
    std::array<F,3> u0 = {{ 1., 0., -1. }} ; 
    V::normalize<F>(u, u0) ; 

    std::cout << " u0 " << V::desc<F>(u0) << std::endl ; 
    std::cout << " u  " << V::desc<F>(u)  << std::endl ; 

    int N = 360 ; 
    for(int i=0 ; i <= N ; i++)
    {   
        F phi = 2.*M_PI*F(i)/F(N) ; 

        std::array<F,3> d0 = {{ cos(phi), sin(phi), zero }} ; 
        // d0: ring of vectors in XY plane, "around" the +Z direction 

        std::array<F,3> d1(d0) ; 
        V::rotateUz(d1,u); 

        // d1: rotated XY ring of vectors to point in direction u 
        // So all the d1 are perpendicular to u 

        F chk = V::dot<F>(d1,u) ; 
        F len = V::length<F>(d1) ; 
        F expect_len = 1. ; 
        F diff_len = std::abs(len - expect_len ); 

        std::cout 
            << std::setw(2) << i 
            << " d0 " << V::desc<F>(d0)  
            << " d1 " << V::desc<F>(d1) 
            << " V::dot<F>(d1,u)  " << std::scientific << chk 
            << " len " << std::scientific << len
            << std::endl 
            ;    

         assert( chk < 1.e-6 );  
         assert( diff_len < 1.e-6 ); 
    }   
}


template<typename F>
void test_make_transverse_(const std::array<F,3>& mom0 )
{
    std::array<F,3> mom(mom0) ; 
    V::normalize(mom, mom0) ; 

    std::cout << "test_make_transverse_ " << std::endl ; 
    std::cout << " mom0 " << V::desc<F>( mom0 ) << std::endl ; 
    std::cout << " mom  " << V::desc<F>( mom  ) << std::endl ; 
    std::array<F,3> pol = {0.,0.,0.} ; 

    int N = 360 ; 
    for(int i=0 ; i <= N ; i++)
    {   
        F frac_twopi = F(i)/F(N) ;  
        V::make_transverse<F>( pol, mom, frac_twopi ); 
        F dot_pol_mom = V::dot<F>( pol, mom ); 

        std::cout 
            << std::setw(4) << i 
            << " "
            << " pol " << V::desc<F>(pol) 
            << " dot_pol_mom " << std::scientific << dot_pol_mom
            << std::endl 
            ; 
        assert( dot_pol_mom < 1e-7 ); 
    }
}
  
template<typename F>
void test_make_transverse()
{
    {
        std::array<F,3> mom = {0.,0.,1.} ; 
        test_make_transverse_<F>( mom ); 
    }
    {
        std::array<F,3> mom = {1.,0.,-1.} ; 
        test_make_transverse_<F>( mom ); 
    }
}

int main()
{
    /*
    test_dot<float>();  
    test_cross<float>();  
    test_rotateUz<float>();  
    */

    test_make_transverse<float>(); 

    return 0 ; 
}
