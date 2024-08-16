#include <cassert>
#include <iostream>

#include "G4OpticalPhoton.hh"
#include "G4ParticleMomentum.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4VParticleChange.hh"
#include "G4NavigationHistory.hh"
#include "G4TouchableHistory.hh"
#include "G4TouchableHandle.hh"

#include "G4OpBoundaryProcess.hh"
#include "C4OpBoundaryProcess.hh"
#include "C4IPMTAccessor.h"

struct DummyPMTAccessor : public C4IPMTAccessor
{
    static constexpr const char* NAME = "DummyPMTAccessor" ; 

    int get_num_lpmt() const { return 0 ; }
    double get_pmtid_qe( int pmtid, double energy ) const { return 1. ; }   
    double get_qescale( int pmtid ) const { return 1. ; } 
    int    get_pmtcat( int pmtid  ) const { return 0 ; }
    void   get_stackspec( std::array<double, 16>& ss, int pmtcat, double energy_eV ) const { } ; 
    const char* get_typename() const { return NAME ; }

    double get_pmtid_qe_angular( int pmtid, double energy, double lposcost, double minus_cos_theta ) const { return 1. ; }   

    int  get_implementation_version() const ;
    void set_implementation_version(int);

};


inline int DummyPMTAccessor::get_implementation_version() const
{
    return 1 ; 
}

inline void DummyPMTAccessor::set_implementation_version(int v)
{
}





int main()
{
    C4IPMTAccessor* accessor = new DummyPMTAccessor ; 
    C4OpBoundaryProcess* proc = new C4OpBoundaryProcess(accessor) ;   

    std::cout << " accessor  " << std::hex << accessor << std::dec << std::endl ; 
    std::cout << " proc      " << std::hex << proc << std::dec << std::endl ; 

    assert( proc ); 


    G4Material* material1 = nullptr ; 
    G4Material* material2 = nullptr ; 


    G4ThreeVector mom(0,0,1) ; 
    G4ThreeVector pol(0,1,0) ; 

    G4double en = 1.*CLHEP::MeV ; 
    G4ParticleMomentum momentum(mom.x(),mom.y(),mom.z()); 
    G4DynamicParticle* particle = new G4DynamicParticle(G4OpticalPhoton::Definition(),momentum);
    particle->SetPolarization(pol.x(), pol.y(), pol.z() );  
    particle->SetKineticEnergy(en); 

    G4ThreeVector position(0.f, 0.f, 0.f); 
    G4double time(0.); 

    G4Track* track = new G4Track(particle,time,position);
    G4StepPoint* pre = new G4StepPoint ; 
    G4StepPoint* post = new G4StepPoint ; 

    G4ThreeVector pre_position(0., 0., 0.);
    G4ThreeVector post_position(0., 0., 1.);
    pre->SetPosition(pre_position); 
    post->SetPosition(post_position); 

    G4VPhysicalVolume* prePV = nullptr ; 
    G4VPhysicalVolume* postPV = nullptr ; 

    G4NavigationHistory* pre_navHist = new G4NavigationHistory ; 
    pre_navHist->SetFirstEntry( prePV );  

    G4NavigationHistory* post_navHist = new G4NavigationHistory ;   
    post_navHist->SetFirstEntry( postPV );  

    G4TouchableHistory* pre_touchHist = new G4TouchableHistory(*pre_navHist);
    G4TouchableHistory* post_touchHist = new G4TouchableHistory(*post_navHist);

    G4TouchableHandle pre_touchable(pre_touchHist);
    G4TouchableHandle post_touchable(post_touchHist);

    pre->SetTouchableHandle(pre_touchable);
    post->SetTouchableHandle(post_touchable);

    const G4StepStatus postStepStatus = fGeomBoundary ;
    post->SetStepStatus( postStepStatus );

    // G4Track::GetMaterial comes from current step preStepPoint 
    pre->SetMaterial(material1);    // HUH: why not const 
    post->SetMaterial(material2);

    G4double step_length = 1. ;

    G4Step* step = new G4Step ;
    step->SetPreStepPoint(pre);
    step->SetPostStepPoint(post);
    step->SetStepLength(step_length);
    track->SetStepLength(step_length);
    track->SetStep(step);

    G4VParticleChange* change = proc->PostStepDoIt(*track, *step); 
    assert( change ); 

    return 0 ; 
}


