#pragma once
/**
C4Track.h
==========

Aim of C4Track is to keep the details of track labelling 
(C4Pho, C4TrackInfo) mostly hidden behind a higher level interface.  
To simplify usage and provide an interface with a higher
probablility of being able to stay unchanged.  

**/

#include "C4TrackInfo.h"
#include "C4Pho.h"
#include "G4OpticalPhoton.hh"

struct C4Track
{
    enum { GS, IX, ID } ; 
    static int  GetLabelVA(const G4Track* track, int i ); 
    static int  GetLabelGS(const G4Track* track ); 
    static int  GetLabelIX(const G4Track* track ); 
    static int  GetLabelID(const G4Track* track ); 

    enum { GEN, EPH, EXT, FLG } ;  
    static int  GetLabelValue( const G4Track* track, int i ); 
    static void SetLabelValue( const G4Track* track, int i, int value ); 
    static void IncrementLabelValue(const G4Track* track, int i ); 

    static int  GetLabelGen( const G4Track* track ); 
    static int  GetLabelEph( const G4Track* track ); 
    static int  GetLabelExt( const G4Track* track ); 
    static int  GetLabelFlg( const G4Track* track ); 

    static void SetLabelGen( const G4Track* track, int gen ); 
    static void SetLabelEph( const G4Track* track, int eph ); 
    static void SetLabelExt( const G4Track* track, int ext ); 
    static void SetLabelFlg( const G4Track* track, int flg ); 

    static void IncrementLabelGen( const G4Track* track ); 
    static void IncrementLabelEph( const G4Track* track ); 
    static void IncrementLabelExt( const G4Track* track ); 
    static void IncrementLabelFlg( const G4Track* track ); 

    static std::string Desc( const G4Track* track );  

    // below methods are mainly for low level testing 
    static void SetLabel(       G4Track* track, int  gs, int  ix, int  id, int  gen, int  eph, int  ext, int  flg ); 
    static bool GetLabel( const G4Track* track, int& gs, int& ix, int& id, int& gen, int& eph, int& ext, int& flg ); 
    static G4Track* MakePhoton(); 
};


inline int C4Track::GetLabelVA( const G4Track* track, int i )
{
    const C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track); 
    return label ? label->get_va(i) : -1 ; 
}
inline int C4Track::GetLabelGS( const G4Track* track ){ return GetLabelVA(track, GS) ; }
inline int C4Track::GetLabelIX( const G4Track* track ){ return GetLabelVA(track, IX) ; }
inline int C4Track::GetLabelID( const G4Track* track ){ return GetLabelVA(track, ID) ; }



inline int C4Track::GetLabelValue(const G4Track* track, int i )
{
    const C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track); 
    return label ? label->value(i) : -1 ; 
}
inline void C4Track::SetLabelValue(const G4Track* track, int i, int val )
{
    C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track); 
    if(label == nullptr) return ; 
    label->set_value(i, val ); 
}
inline void C4Track::IncrementLabelValue(const G4Track* track, int i )
{
    C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track); 
    if(label == nullptr) return ; 
    label->increment_value(i); 
}

inline int C4Track::GetLabelGen(const G4Track* track){ return GetLabelValue(track, GEN) ; }
inline int C4Track::GetLabelEph(const G4Track* track){ return GetLabelValue(track, EPH) ; }
inline int C4Track::GetLabelExt(const G4Track* track){ return GetLabelValue(track, EXT) ; }
inline int C4Track::GetLabelFlg(const G4Track* track){ return GetLabelValue(track, FLG) ; }

inline void C4Track::SetLabelGen(const G4Track* track, int gen ){ SetLabelValue(track, GEN, gen ) ; }
inline void C4Track::SetLabelEph(const G4Track* track, int eph ){ SetLabelValue(track, EPH, eph ) ; }
inline void C4Track::SetLabelExt(const G4Track* track, int ext ){ SetLabelValue(track, EXT, ext ) ; }
inline void C4Track::SetLabelFlg(const G4Track* track, int flg ){ SetLabelValue(track, FLG, flg ) ; }

inline void C4Track::IncrementLabelGen(const G4Track* track){ IncrementLabelValue(track, GEN ); }
inline void C4Track::IncrementLabelEph(const G4Track* track){ IncrementLabelValue(track, EPH ); }
inline void C4Track::IncrementLabelExt(const G4Track* track){ IncrementLabelValue(track, EXT ); }
inline void C4Track::IncrementLabelFlg(const G4Track* track){ IncrementLabelValue(track, FLG ); }






inline std::string C4Track::Desc( const G4Track* track )
{
    C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track); 
    std::stringstream ss ; 
    ss << "C4Track::Desc " ; 
    ss << ( label ? label->desc() : "-" ) ; 
    std::string str = ss.str(); 
    return str ; 
}

inline void C4Track::SetLabel( G4Track* track, int  gs, int  ix, int  id, int  gen, int  eph, int  ext, int  flg )
{
    C4Pho pho = {gs, ix, id, {(unsigned char)gen, (unsigned char)eph, (unsigned char)ext, (unsigned char)flg} } ;  
    C4TrackInfo<C4Pho>::Set(track, pho );  
}
inline bool C4Track::GetLabel( const G4Track* track, int& gs, int& ix, int& id, int& gen, int& eph, int& ext, int& flg )
{
    C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track) ; 
    if(label == nullptr) return false ; 

    gs  = label->gs ;
    ix  = label->ix ; 
    id  = label->id ; 
    gen = label->gen() ; 
    eph = label->eph() ; 
    ext = label->ext() ; 
    flg = label->flg() ;  

    return true ; 
}
inline G4Track* C4Track::MakePhoton()
{
    G4ParticleMomentum momentum(0., 0., 1.); 
    G4DynamicParticle* particle = new G4DynamicParticle(G4OpticalPhoton::Definition(),momentum);
    particle->SetPolarization(0., 1., 0. );  
    G4double time(0.); 
    G4ThreeVector position(0., 0., 0.); 
    G4Track* track = new G4Track(particle,time,position);
    return track ; 
}



