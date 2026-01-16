#pragma once
/**
C4TrackInfo.h (from opticks/sysrap/STrackInfo.h)
===================================================

Required methods for T::

   std::string desc() const ;
   T T::Placeholder() ;
   T T::Fabricate(int id) ;

**/

#include <string>
#include "G4Track.hh"
#include "G4VUserTrackInformation.hh"
#include "C4Pho.h"

struct C4TrackInfo : public G4VUserTrackInformation
{
    C4Pho label  ;

    C4TrackInfo(const C4Pho& label);
    std::string desc() const ;

    static C4TrackInfo* GetTrackInfo_UNDEFINED(const G4Track* track);
    static C4TrackInfo* GetTrackInfo(const G4Track* track);
    static bool Exists(const G4Track* track);
    static C4Pho  Get(   const G4Track* track);   // by value
    static C4Pho* GetRef(const G4Track* track);   // by reference, allowing inplace changes
    static std::string Desc(const G4Track* track);

    static void Set(G4Track* track, const C4Pho& label );
};

inline C4TrackInfo::C4TrackInfo(const C4Pho& _label )
    :
    G4VUserTrackInformation("C4TrackInfo"),
    label(_label)
{
}

inline std::string C4TrackInfo::desc() const
{
    std::stringstream ss ;
    ss << *pType << " " << label.desc() ;
    std::string str = ss.str();
    return str ;
}

/**
C4TrackInfo::GetTrackInfo_UNDEFINED
------------------------------------

With U4PhotonInfo the ancestor of STrackInfo was using dynamic_cast
without issue. After moving to the templated STrackInfo the
dynamic cast always giving nullptr. So switched to static_cast.

THIS MEANS THAT THIS STRUCT SHOULD NOT BE USED UNLESS CAN BE
SURE THAT NO OTHER G4VUserTrackInformation SUB-CLASS IS IN USE
AS STATIC CASTING APPLIED TO SOME OTHER TYPE WILL LEAD TO UNDEFINED
ACTIONS

**/

inline C4TrackInfo* C4TrackInfo::GetTrackInfo_UNDEFINED(const G4Track* track) // static, label by value
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    C4TrackInfo* trackinfo = ui ? static_cast<C4TrackInfo*>(ui) : nullptr ;
    return trackinfo ;
}

inline C4TrackInfo* C4TrackInfo::GetTrackInfo(const G4Track* track) // static, label by value
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    C4TrackInfo* trackinfo = ui ? dynamic_cast<C4TrackInfo*>(ui) : nullptr ;
    return trackinfo ;
}




inline bool C4TrackInfo::Exists(const G4Track* track) // static
{
    C4TrackInfo* trackinfo = GetTrackInfo(track);
    return trackinfo != nullptr ;
}

inline C4Pho C4TrackInfo::Get(const G4Track* track) // static, label by value
{
    C4TrackInfo* trackinfo = GetTrackInfo(track);
    return trackinfo ? trackinfo->label : C4Pho::Placeholder() ;
}

inline C4Pho* C4TrackInfo::GetRef(const G4Track* track) // static, label reference
{
    C4TrackInfo* trackinfo = GetTrackInfo(track);
    return trackinfo ? &(trackinfo->label) : nullptr ;
}

inline std::string C4TrackInfo::Desc(const G4Track* track)
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;

    C4TrackInfo* trackinfo = GetTrackInfo(track);
    C4TrackInfo* trackinfo_undefined = GetTrackInfo_UNDEFINED(track);

    std::stringstream ss ;
    ss << "C4TrackInfo::Desc"
       << std::endl
       << " track " << track
       << " track.GetUserInformation " << ui
       << std::endl
       << " trackinfo " << trackinfo
       << " trackinfo_undefined " << trackinfo_undefined
       << std::endl
       << " trackinfo.desc " << ( trackinfo ? trackinfo->desc() : "-" )
       << std::endl
       << " trackinfo_undefined.desc " << ( trackinfo_undefined ? trackinfo_undefined->desc() : "-" )
       ;
    std::string str = ss.str();
    return str ;
}


inline void C4TrackInfo::Set(G4Track* track, const C4Pho& _label )  // static
{
    C4Pho* label = GetRef(track);
    if(label == nullptr)
    {
        track->SetUserInformation(new C4TrackInfo(_label));
    }
    else
    {
        *label = _label ;
    }
}

