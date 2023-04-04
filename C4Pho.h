#pragma once
/**
C4Pho.h (from opticks/sysrap/spho.h) : photon labelling used by genstep collection
=======================================================================================

isSameLineage 
    does not require the same reemission generation

isIdentical
    requires isSameLineage and same reemission generation

NB spho lacks gentype, to get that must reference corresponding sgs struct using the gs index 

NB having reemission generations larger than zero DOES NOT mean the 
photon originally came from scintillaton.
For example in a case where no photons are coming from scint, 
reemission of initially Cerenkov photons may still happen, 
resulting in potentially multiple reemission generations.   

**/

#include <string>
#include <array>

struct C4Pho_uchar4 { unsigned char x,y,z,w ; }; 

union C4Pho_uuc4 { 
    unsigned     u   ;   
    C4Pho_uchar4 uc4 ; 
};  

struct C4Pho
{
    static constexpr const int N = 4 ;

    int gs ; // 0-based genstep index within the event
    int ix ; // 0-based photon index within the genstep
    int id ; // 0-based photon identity index within the event 

    C4Pho_uchar4 uc4 ;  
    // uc4.x : gen : 0-based reemission index incremented at each reemission 
    // uc4.y : eph : eg junoSD_PMT_v2::ProcessHits eph enumeration 
    // uc4.z : ext : unused
    // uc4.w : flg : photon point flag TO/BT/BR/SC/AB/SD/SR/... etc 

    unsigned uc4packed() const ; 

    int gen() const ; 
    int eph() const ; 
    int ext() const ; 
    int flg() const ; 

    void set_gen(int gn) ; 
    void set_eph(int ep) ; 
    void set_ext(int ex) ; 
    void set_flg(int fg) ; 
 
    static C4Pho MakePho(int gs_, int ix_, int id_ );   
    static C4Pho Fabricate(int track_id); 
    static C4Pho Placeholder() ; 

    bool isSameLineage(const C4Pho& other) const { return gs == other.gs && ix == other.ix && id == other.id ; }
    bool isIdentical(const C4Pho& other) const { return isSameLineage(other) && uc4.x == other.uc4.x ; }

    bool isPlaceholder() const { return gs == -1 ; }
    bool isDefined() const {     return gs != -1 ; }

    C4Pho make_nextgen() const ; // formerly make_reemit 
    std::string desc() const ;

    const int* cdata() const ;
    int* data();
    void serialize( std::array<int, 4>& a ) const ; 
    void load( const std::array<int, 4>& a );  
};


#include <cassert>
#include <sstream>
#include <iomanip>

inline unsigned C4Pho::uc4packed() const
{
    C4Pho_uuc4 uuc4 ; 
    uuc4.uc4 = uc4 ; 
    return uuc4.u ;    
}

inline int C4Pho::gen() const { return int(uc4.x); }
inline int C4Pho::eph() const { return int(uc4.y); }
inline int C4Pho::ext() const { return int(uc4.z); }
inline int C4Pho::flg() const { return int(uc4.w); }

inline void C4Pho::set_gen(int gn) { uc4.x = (unsigned char)(gn) ; }
inline void C4Pho::set_eph(int ep) { uc4.y = (unsigned char)(ep) ; }
inline void C4Pho::set_ext(int ex) { uc4.z = (unsigned char)(ex) ; }
inline void C4Pho::set_flg(int fg) { uc4.w = (unsigned char)(fg) ; }


inline C4Pho C4Pho::MakePho(int gs_, int ix_, int id_) // static
{
    C4Pho ph = {gs_, ix_, id_, {0,0,0,0} } ; 
    return ph ;   
}
/**
C4Pho::Fabricate
---------------

*Fabricate* is not normally used, as C+S photons are always 
labelled at generation by U4::GenPhotonEnd

However as a workaround for torch/input photons that lack labels
this method is used from U4Recorder::PreUserTrackingAction_Optical
to provide a standin label based only on a 0-based track_id. 

**/
inline C4Pho C4Pho::Fabricate(int track_id) // static
{
    assert( track_id >= 0 ); 
    C4Pho fab = {0, track_id, track_id, {0,0,0,0} };
    return fab ;
}
inline C4Pho C4Pho::Placeholder() // static
{
    C4Pho inv = {-1, -1, -1, {0,0,0,0} };
    return inv ;
}
inline C4Pho C4Pho::make_nextgen() const
{
    // C4Pho nextgen = {gs, ix, id, gn+1 } ;  THIS WOULD SCRUB THE REST OF THE uc4
    C4Pho nextgen = *this ; 
    nextgen.uc4.x += 1 ; 
    return nextgen ;
}

inline std::string C4Pho::desc() const
{
    std::stringstream ss ;
    ss << "C4Pho" ;
    if(isPlaceholder())  
    {
        ss << " isPlaceholder " ; 
    }
    else 
    {
        ss << " (gs:ix:id:gn " 
           << std::setw(3) << gs 
           << std::setw(4) << ix 
           << std::setw(5) << id 
           << "[" 
           << std::setw(3) << int(uc4.x) << ","
           << std::setw(3) << int(uc4.y) << ","
           << std::setw(3) << int(uc4.z) << ","
           << std::setw(3) << int(uc4.w) 
           << "]"
           << ")"
           ;
    }
    std::string s = ss.str();
    return s ;
}




inline int* C4Pho::data()
{
    return &gs ;
}
inline const int* C4Pho::cdata() const
{
    return &gs ;
}
inline void C4Pho::serialize( std::array<int, 4>& a ) const
{
    assert( a.size() == N );
    const int* ptr = cdata() ;
    for(int i=0 ; i < N ; i++ ) a[i] = ptr[i] ;
}
inline void C4Pho::load( const std::array<int, 4>& a )
{
    assert( a.size() == N );
    int* ptr = data() ;
    for(int i=0 ; i < N ; i++ ) ptr[i] = a[i] ;
}




inline std::ostream& operator<<(std::ostream& os, const C4Pho& p)
{
    os << p.desc() ;   
    return os; 
}


