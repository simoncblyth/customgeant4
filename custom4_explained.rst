Custom4 Explained
=====================





C4OpBoundaryProcess : Custom Boundary Process contructor (ctor)
-----------------------------------------------------------------

The ctor of C4OpBoundaryProcess is instructive to explain what Custom4 does.
All information about the PMTs comes from the accessor object, a pointer 
to which is passed in the first argument. Typically defaults of the other args 
are used.   


customgeant4/C4OpBoundaryProcess.hh::

    117 
    118         C4OpBoundaryProcess(
    119                                      const C4IPMTAccessor* accessor,
    120                                      const G4String& processName = "OpBoundary",
    121                                      G4ProcessType type = fOptical);


customgeant4/C4OpBoundaryProcess.cc::


     119 C4OpBoundaryProcess::C4OpBoundaryProcess(
     120                                                const C4IPMTAccessor* accessor,
     121                                                const G4String& processName,
     122                                                G4ProcessType type)
     123              :
     124              G4VDiscreteProcess(processName, type),
     125              m_custom_status('U'),
     126              m_custom_art(new C4CustomART(
     127                                         accessor,
     128                                         theAbsorption,
     129                                         theReflectivity,
     130                                         theTransmittance,
     131                                         theEfficiency,
     132                                         theGlobalPoint,
     133                                         OldMomentum,
     134                                         OldPolarization,
     135                                         theRecoveredNormal,
     136                                         thePhotonMomentum
     137                                        ))
     138 {



C4IPMTAccessor : Protocol for accessing PMT information including PHC, ARC thickness and refractive index
-----------------------------------------------------------------------------------------------------------

Note that C4IPMTAccessor is pure virtual. It defines the interface with which PMT information can be accessed. 

     01 #pragma once
      2 
      3 #include <array>
      4 
      5 struct C4IPMTAccessor
      6 {
      7     virtual int    get_num_lpmt() const = 0 ;
      8     virtual double get_pmtid_qe( int pmtid, double energy ) const = 0 ;
      9     virtual double get_qescale( int pmtid ) const = 0 ;
     10     virtual int    get_pmtcat( int pmtid  ) const = 0 ;
     11     virtual void   get_stackspec( std::array<double, 16>& ss, int pmtcat, double energy_eV ) const = 0 ;
     12     virtual const char* get_typename() const = 0 ;
     13 
     14 };
     15 


How junosw uses custom4 : instanciation of accessor and custom boundary process
--------------------------------------------------------------------------------

The custom boundary process is instanciated within JUNOSW by 
DsPhysConsOptical which sets up all processes relevant to optical 
photons::

    Simulation/DetSimV2/PhysiSim/include/DsPhysConsOptical.h
    Simulation/DetSimV2/PhysiSim/src/DsPhysConsOptical.cc


DsPhysConsOptical::CreateCustomG4OpBoundaryProcess::

    375 #include "IPMTSimParamSvc/IPMTSimParamSvc.h"
    376 #include "PMTSimParamSvc/PMTSimParamData.h"
    377 #include "PMTSimParamSvc/PMTAccessor.h"
    378 
    379 C4OpBoundaryProcess* DsPhysConsOptical::CreateCustomG4OpBoundaryProcess()
    380 {
    381     SniperPtr<IPMTSimParamSvc> psps_ptr(*getParent(), "PMTSimParamSvc");
    382 
    383     if(psps_ptr.invalid())
    384     {
    385         std::cout << "invalid" << std::endl ;
    386         return nullptr ;
    387     }
    388 
    389     IPMTSimParamSvc* ipsps = psps_ptr.data();
    390     PMTSimParamData* pspd = ipsps->getPMTSimParamData() ;
    391 
    392     C4IPMTAccessor* accessor = new PMTAccessor(pspd) ;
    393     C4OpBoundaryProcess* boundproc = new C4OpBoundaryProcess(accessor) ;
    394     std::cout << "DsPhysConsOptical::CreateCustomG4OpBoundaryProcess" << std::endl ;
    395 
    396     return boundproc ;
    397 }


The JUNOSW implementation of the C4IPMTAccessor protocol is done by PMTAccessor.h 

Simulation/SimSvc/PMTSimParamSvc/PMTSimParamSvc/PMTAccessor.h::

    223 inline void PMTAccessor::get_stackspec( std::array<double, 16>& ss, int pmtcat, double energy_eV ) const
    224 {
    225     double energy = energy_eV*CLHEP::eV ;
    226 
    227     ss.fill(0.);
    228 
    229     ss[4*0+0] = PyrexRINDEX ? PyrexRINDEX->Value(energy) : 0. ;
    230 
    231     ss[4*1+0] = data->get_pmtcat_prop(       pmtcat, "ARC_RINDEX" , energy );
    232     ss[4*1+1] = data->get_pmtcat_prop(       pmtcat, "ARC_KINDEX" , energy );
    233     ss[4*1+2] = data->get_pmtcat_const_prop( pmtcat, "ARC_THICKNESS" )/CLHEP::nm ;
    234 
    235     ss[4*2+0] = data->get_pmtcat_prop(       pmtcat, "PHC_RINDEX" , energy );
    236     ss[4*2+1] = data->get_pmtcat_prop(       pmtcat, "PHC_KINDEX" , energy );
    237     ss[4*2+2] = data->get_pmtcat_const_prop( pmtcat, "PHC_THICKNESS" )/CLHEP::nm ;
    238 
    239     ss[4*3+0] = VacuumRINDEX ? VacuumRINDEX->Value(energy) : 1. ;
    240 }




When the custom boundary calculation is used 
----------------------------------------------

* only for boundary intersects onto the Z>0 upper portion of volumes with 
  optical surfaces named beginning with '@'



C4OpBoundaryProcess.cc::

     504             //[OpticalSurface.mpt.CustomPrefix
     505             if( OpticalSurfaceName0 == '@' || OpticalSurfaceName0 == '#' )  // only customize specially named OpticalSurfaces 
     506             {
     507                 if( m_custom_art->local_z(aTrack) < 0. ) // lower hemi : No customization, standard boundary  
     508                 {
     509                     m_custom_status = 'Z' ;
     510                 }
     511                 else if( OpticalSurfaceName0 == '@') //  upper hemi with name starting @ : MultiFilm ART transmit thru into PMT
     512                 {
     513                     m_custom_status = 'Y' ;
     514 
     515 #ifdef C4_DEBUG_PIDX
     516                     m_custom_art->dump = m_track_dump ;
     517 #endif
     518                     m_custom_art->doIt(aTrack, aStep) ;
     519 
     520                     /**
     521                     m_custom_art calculates 3-way probabilities (A,R,T) that sum to 1. 
     522                     and looks up theEfficiency appropriate for the PMT 
     523                     
     524                     BUT: as DielectricDielectric is expecting a 2-way *theTransmittance* probability 
     525                     m_custom_art leaves theAbsorption as A and rescales the others to create 2-way probs::
     526 
     527                          ( theAbsorption, theReflectivity, theTransmittance ) =  ( A, R/(1-A), T/(1-A) )
     528 
     529                     **/
     530 
     531 
     532                     type = dielectric_dielectric ;
     533                     theModel = glisur ;
     534                     theFinish = polished ;  // to make Rindex2 get picked up below, plus use theGlobalNormal as theFacetNormal 
     535 
     536                     // ACTUALLY : ITS SIMPLER TO TREAT m_custom_status:Y as kinda another type 
     537                     // in the big type switch below to avoid depending on the jungle
     538 
     539                 }
     540                 else if( OpticalSurfaceName0 == '#' ) // upper hemi with name starting # : Traditional Detection at photocathode
     541                 {
     542                     m_custom_status = '-' ;
     543 
     544                     type = dielectric_metal ;
     545                     theModel = glisur ;
     546                     theReflectivity = 0. ;
     547                     theTransmittance = 0. ;
     548                     theEfficiency = 1. ;
     549                 }
     550             }
     551             //]OpticalSurface.mpt.CustomPrefix



What the custom boundary calc does
------------------------------------

* custom boundry calc changes its reference ctor arguments "the{Absorption,Reflectivity,Transmittance,Efficiency}" aka {A,R,T,E}
* so it changes the proportions of photons absorbed/reflected/transmitted as well as setting theEfficiency 


C4CustomART.h::

    107     C4CustomART(
    108         const C4IPMTAccessor* accessor,
    109         G4double& theAbsorption,
    110         G4double& theReflectivity,
    111         G4double& theTransmittance,
    112         G4double& theEfficiency,
    113         const G4ThreeVector& theGlobalPoint,
    114         const G4ThreeVector& OldMomentum,
    115         const G4ThreeVector& OldPolarization,
    116         const G4ThreeVector& theRecoveredNormal,
    117         const G4double& thePhotonMomentum
    118     );



::

    /**
    C4CustomART::doIt
    ------------------
    
    Dot product "mct"::

       G4double minus_cos_theta = OldMomentum*theRecoveredNormal 
    
    theRecoveredNormal points outwards, away from PMT boundary 

      



          ingoing  
          mct < 0 
               V     
                \  :        :
                 \ :        :
                  \:        :    Pyrex 
             ------+--------------------- ARC
             ------+--------+------------ PHC 
                           /     Vacuum
                          /
                         /
                        ^
                      outgoing  
                      mct > 0 



    
    0. calculate "mct" (minus_cos_theta( from track direction and the outward normal  
    1. find pmtid via G4Track geometry lookup with C4Touchable::VolumeIdentifier 
    2. C4IPMTAccessor::get_pmtcat lookup type of PMT from pmtid 
    3. C4IPMTAccessor::get_pmtid_qe lookup PMT specific QE 
    4. artifically set QE to zero for outgoing photons : mct > 0
    5. C4IPMTAccessor::get_stackspec lookup thicknesses and refractive indices for 4 layers

       * note accessor argument is based on pmtcat and energy (NOT pmtid)

    **/



HOW TO IMPROVE:

* move decision about backwards zeroing into accessor ? 
* generalize for angle dependent impl

"get_pmtid_qe_angular(pmtid,energy,local_theta_poi,minus_cos_theta_aoi)":

* local_theta ("position of incidence")
* minus_cos_theta ("angle of incidence") 

* HMM: thats confusing the theta are not the same




::

    293 inline void C4CustomART::doIt(const G4Track& aTrack, const G4Step& )
    294 {
    295     G4double zero = 0. ;
    296     G4double minus_one = -1. ;
    297     G4double minus_cos_theta = OldMomentum*theRecoveredNormal ;
    298     G4double dot_pol_cross_mom_nrm = OldPolarization*OldMomentum.cross(theRecoveredNormal) ;
    299 
    300     G4double energy = thePhotonMomentum ;
    301     G4double wavelength = CLHEP::twopi*CLHEP::hbarc/energy ;
    302     G4double energy_eV = energy/CLHEP::eV ;
    303     G4double wavelength_nm = wavelength/CLHEP::nm ;
    304 
    305     int pmtid = C4Touchable::VolumeIdentifier(&aTrack, true );
    306     int pmtcat = accessor->get_pmtcat( pmtid ) ;
    307     double _qe = minus_cos_theta > 0. ? 0.0 : accessor->get_pmtid_qe( pmtid, energy ) ;  // energy_eV ?
    308     // following the old junoPMTOpticalModel with "backwards" _qe always zero 
    309 
    310     std::array<double,16> a_spec ;
    311     accessor->get_stackspec(a_spec, pmtcat, energy_eV );
    312 
    313     const double* ss = a_spec.data() ;
    314 
    315     Stack<double,4> stack ;
    316 
    317     theEfficiency = zero ;
    318     if( minus_cos_theta < zero ) // only ingoing photons 
    319     {
    320         stack.calc( wavelength_nm, minus_one, zero, ss, 16u );
    321         theEfficiency = _qe/stack.art.A ;    // aka escape_fac
    322 
    323         bool expect = theEfficiency <= 1. ;
    324         if(!expect) std::cerr
    325             << "C4CustomART::doIt"
    326             << " FATAL "
    327             << " ERR: theEfficiency > 1. : " << theEfficiency
    328             << " _qe " << _qe
    329             << " stack.art.A (aka An) " << stack.art.A
    330             << std::endl
    331             ;
    332         assert( expect );




