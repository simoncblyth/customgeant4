#pragma once

#include <array>

struct C4IPMTAccessor
{
    virtual int    get_num_lpmt() const = 0 ; 
    virtual double get_pmtid_qe( int pmtid, double energy ) const = 0 ;
    virtual double get_qescale( int pmtid ) const = 0 ;
    virtual int    get_pmtcat( int pmtid  ) const = 0 ;  
    virtual void   get_stackspec( std::array<double, 16>& ss, int pmtcat, double energy_eV ) const = 0 ; 
    virtual const char* get_typename() const = 0 ; 

};


