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
    static int  GetLabelId(   const G4Track* track ); 
    static int  GetLabelFlag( const G4Track* track ); 
    static void SetLabelFlag( const G4Track* track, int flag ); 
    static std::string Desc( const G4Track* track );  

    // below methods are mainly for low level testing 
    static void SetLabel(       G4Track* track, int  gs, int  ix, int  id, int  gen, int  eph, int  ext, int  flg ); 
    static bool GetLabel( const G4Track* track, int& gs, int& ix, int& id, int& gen, int& eph, int& ext, int& flg ); 
    static G4Track* MakePhoton(); 
};


inline int C4Track::GetLabelId( const G4Track* track )
{
    const C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track); 
    return label ? label->id : -1 ; 
}
inline int C4Track::GetLabelFlag(const G4Track* track )
{
    const C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track); 
    return label->flg() ; 
}
inline void C4Track::SetLabelFlag(const G4Track* track, int flag )
{
    C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track); 
    label->set_flg(flag); 
}

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



