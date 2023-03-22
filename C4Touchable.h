#pragma once
/**
S4Touchable.h
================

The touchable history is from inside to outside, so "deeper" corresponds to wider volumes.

This was developed within opticks, see::

    opticks/u4/U4Touchable.h
    opticks/u4/tests/U4TouchableTest.cc

**/

#include <string>
#include <sstream>
#include <iomanip>

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4TouchableHistory.hh"
#include "G4VSolid.hh"

struct S4Touchable
{
    static int ReplicaNumber(const G4VTouchable* touch) ; 
    static int ReplicaDepth(const G4VTouchable* touch) ; 
    static int TouchDepth(const G4VTouchable* touch ); 
    static bool HasMoreThanOneDaughterWithName( const G4LogicalVolume* lv, const char* name); 

    static std::string Brief(const G4VTouchable* touch ); 
    static std::string Desc(const G4VTouchable* touch, int so_wid=20, int pv_wid=20);
};

inline int S4Touchable::ReplicaNumber(const G4VTouchable* touch)  // static 
{
    int d = ReplicaDepth(touch);
    return d > -1 ? touch->GetReplicaNumber(d) : -1  ;
}

/**
S4Touchable::ReplicaDepth
---------------------------

Starting from touch depth look outwards at (volume, mother_volume) 
pairs checking for a depth at which the mother_volume has more than one 
daughter with the same name as the volume. This means the volume has
at least one same named sibling making this the replica depth. 
When no such depth is found return -1. 

**/

inline int S4Touchable::ReplicaDepth(const G4VTouchable* touch)   // static
{
    int nd = touch->GetHistoryDepth();
    int t = TouchDepth(touch); 
    assert( t > -1 && t < nd ); 

    for (int d=t ; d < nd-1; ++d ) 
    {   
        const G4VPhysicalVolume* dpv = touch->GetVolume(d);
        const G4VPhysicalVolume* mpv = touch->GetVolume(d+1);

        const G4LogicalVolume* dlv = dpv->GetLogicalVolume();
        const G4LogicalVolume* mlv = mpv->GetLogicalVolume();

        bool hierarchy = dpv->GetMotherLogical() == mlv ; 
        assert(hierarchy); 

        const char* dlv_name = dlv->GetName().c_str() ; 
        if(HasMoreThanOneDaughterWithName(mlv, dlv_name)) return d ; 
    }
    return -1 ;
}

/**
S4Touchable::TouchDepth
-------------------------

Depth of touch volume, -1 if not found. 

**/

inline int S4Touchable::TouchDepth(const G4VTouchable* touch ) // static
{
    const G4VPhysicalVolume* tpv = touch->GetVolume() ;
    int t = -1 ; 
    for(int i=0 ; i < touch->GetHistoryDepth() ; i++) 
    {   
        const G4VPhysicalVolume* ipv = touch->GetVolume(i) ; 
        if(ipv == tpv) 
        {
            t = i ;  
            break ; 
        }
    } 
    return t ; 
}

inline bool S4Touchable::HasMoreThanOneDaughterWithName( const G4LogicalVolume* lv, const char* name)  // static
{
    int num_dau = lv->GetNoDaughters();
    if(num_dau <= 1) return false ; 
    int count = 0;  
    for (int k=0; k < num_dau ; ++k)   
    {
        const G4VPhysicalVolume* kpv = lv->GetDaughter(k) ;
        const G4LogicalVolume*   klv = kpv->GetLogicalVolume() ;
        const char* klv_name = klv->GetName().c_str() ;
        if(strcmp(name, klv_name)==0) count += 1 ;
        if(count > 1) return true ;
    }
    return false ; 
}

inline std::string S4Touchable::Brief(const G4VTouchable* touch )
{
    std::stringstream ss ; 
    ss << "S4Touchable::Brief"
       << " HistoryDepth " << std::setw(2) <<  touch->GetHistoryDepth()
       << " TouchDepth " << std::setw(2) << TouchDepth(touch)
       << " ReplicaDepth " << std::setw(2) << ReplicaDepth(touch)
       << " ReplicaNumber " << std::setw(6) << ReplicaNumber(touch)
       ; 
    return ss.str(); 
}
inline std::string S4Touchable::Desc(const G4VTouchable* touch, int so_wid, int pv_wid )
{
    int history_depth = touch->GetHistoryDepth();
    int touch_depth = TouchDepth(touch); 
    int replica_depth = ReplicaDepth(touch); 
    int replica_number = ReplicaNumber(touch); 

    std::stringstream ss ; 
    ss << "S4Touchable::Desc"
       << " HistoryDepth " << std::setw(2) << history_depth 
       << " TouchDepth " << std::setw(2) << touch_depth 
       << " ReplicaDepth " << std::setw(2) << replica_depth
       << " ReplicaNumber " << std::setw(6) << replica_number 
       << std::endl 
       ; 

    for(int i=0 ; i< history_depth ; i++) 
    { 
        G4VPhysicalVolume* pv = touch->GetVolume(i); 
        G4LogicalVolume* lv = pv->GetLogicalVolume();
        G4int nd = lv->GetNoDaughters();
        G4VSolid* so = touch->GetSolid(i); 
        G4int cp = touch->GetReplicaNumber(i); 

        ss << " i " << std::setw(2) << i 
           << " cp " << std::setw(6)  << cp
           << " nd " << std::setw(6)  << nd
           << " so " << std::setw(so_wid) << so->GetName()
           << " pv " << std::setw(pv_wid) << pv->GetName()
           << std::endl ; 
    }   
    std::string s = ss.str(); 
    return s ; 
}


