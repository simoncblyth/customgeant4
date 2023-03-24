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

template<typename T>
struct C4TrackInfo : public G4VUserTrackInformation
{
    T label  ; 

    C4TrackInfo(const T& label); 
    std::string desc() const ; 

    static C4TrackInfo<T>* GetTrackInfo(const G4Track* track); 
    static C4TrackInfo<T>* GetTrackInfo_dynamic(const G4Track* track); 
    static bool Exists(const G4Track* track); 
    static T  Get(   const G4Track* track);   // by value 
    static T* GetRef(const G4Track* track);   // by reference, allowing inplace changes
    static std::string Desc(const G4Track* track); 

    static void Set(G4Track* track, const T& label ); 
};

template<typename T>
inline C4TrackInfo<T>::C4TrackInfo(const T& _label )
    :   
    G4VUserTrackInformation("C4TrackInfo"),
    label(_label)
{
}
 
template<typename T>
inline std::string C4TrackInfo<T>::desc() const 
{
    std::stringstream ss ; 
    ss << *pType << " " << label.desc() ; 
    std::string str = ss.str(); 
    return str ; 
}

/**
C4TrackInfo::GetTrackInfo
--------------------------

With U4PhotonInfo the ancestor of STrackInfo was using dynamic_cast 
without issue. After moving to the templated STrackInfo the 
dynamic cast always giving nullptr. So switched to static_cast. 

**/

template<typename T>
inline C4TrackInfo<T>* C4TrackInfo<T>::GetTrackInfo(const G4Track* track) // static, label by value 
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    C4TrackInfo<T>* trackinfo = ui ? static_cast<C4TrackInfo<T>*>(ui) : nullptr ;
    return trackinfo ; 
}

template<typename T>
inline C4TrackInfo<T>* C4TrackInfo<T>::GetTrackInfo_dynamic(const G4Track* track) // static, label by value 
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;
    C4TrackInfo<T>* trackinfo = ui ? dynamic_cast<C4TrackInfo<T>*>(ui) : nullptr ;
    return trackinfo ; 
}




template<typename T>
inline bool C4TrackInfo<T>::Exists(const G4Track* track) // static
{
    C4TrackInfo<T>* trackinfo = GetTrackInfo(track); 
    return trackinfo != nullptr ; 
}

template<typename T>
inline T C4TrackInfo<T>::Get(const G4Track* track) // static, label by value 
{
    C4TrackInfo<T>* trackinfo = GetTrackInfo(track); 
    return trackinfo ? trackinfo->label : T::Placeholder() ; 
}

template<typename T>
inline T* C4TrackInfo<T>::GetRef(const G4Track* track) // static, label reference 
{
    C4TrackInfo<T>* trackinfo = GetTrackInfo(track); 
    return trackinfo ? &(trackinfo->label) : nullptr ; 
}

template<typename T>
inline std::string C4TrackInfo<T>::Desc(const G4Track* track)
{
    G4VUserTrackInformation* ui = track->GetUserInformation() ;

    C4TrackInfo<T>* trackinfo = GetTrackInfo(track); 
    C4TrackInfo<T>* trackinfo_dyn = GetTrackInfo_dynamic(track); 

    std::stringstream ss ; 
    ss << "C4TrackInfo::Desc" 
       << std::endl 
       << " track " << track 
       << " track.GetUserInformation " << ui
       << std::endl 
       << " trackinfo " << trackinfo 
       << " trackinfo_dyn " << trackinfo_dyn 
       << std::endl 
       << " trackinfo.desc " << ( trackinfo ? trackinfo->desc() : "-" )
       << std::endl 
       << " trackinfo_dyn.desc " << ( trackinfo_dyn ? trackinfo_dyn->desc() : "-" )
       ; 
    std::string str = ss.str(); 
    return str ; 
}  


template<typename T>
inline void C4TrackInfo<T>::Set(G4Track* track, const T& _label )  // static 
{
    T* label = GetRef(track); 
    if(label == nullptr)
    {
        track->SetUserInformation(new C4TrackInfo<T>(_label)); 
    }
    else
    {
        *label = _label ; 
    }
}

