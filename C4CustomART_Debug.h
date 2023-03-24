#pragma once

#include <array>

struct C4CustomART_Debug
{
    static constexpr const int N = 12 ; 

    double A ; 
    double R ; 
    double T ; 
    double _qe ; 

    double An ; 
    double Rn ; 
    double Tn ; 
    double escape_fac ; 

    double minus_cos_theta ; 
    double wavelength_nm ; 
    double pmtid ; 
    double spare ; 

    void serialize( std::array<double, 16>& a );  
    const double* data() const ; 
};

inline void C4CustomART_Debug::serialize( std::array<double, 16>& a )
{
    double* ptr = &A ; 
    for(int i=0 ; i < 16 ; i++ ) a[i] = i < N ? ptr[i] : 0. ;  
}

inline const double* C4CustomART_Debug::data() const { return &A ; }









