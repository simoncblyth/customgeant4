#pragma once

#include <cstdlib>

struct C4Sys
{
    static int getenvint(const char* ekey, int fallback);
    static bool getenvbool(const char* ekey); 
};

inline int C4Sys::getenvint(const char* ekey, int fallback)
{
    char* val = getenv(ekey);
    return val ? std::atoi(val) : fallback ; 
}

inline bool C4Sys::getenvbool( const char* ekey )
{
    char* val = getenv(ekey);
    bool ival = val ? true : false ;
    return ival ; 
}



