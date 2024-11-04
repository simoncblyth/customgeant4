Custom4 Explained
=====================

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



C4IPMTAccessor : Protocol for accessing PMT information including efficiency, PHC, ARC thickness and refractive index
-----------------------------------------------------------------------------------------------------------------------

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



What the custom boundary calc C4CustomART::doIt does
--------------------------------------------------------


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



* custom boundry calc changes the reference ctor arguments "the{Absorption,Reflectivity,Transmittance,Efficiency}" aka {A,R,T,E}

The results of that are:

* via {A,R,T} : change proportions of photons absorbed/reflected/transmitted 
* via {E} : change proportion of "absorbed" photons classified as Detect vs just Absorb 



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



HMM: how to change for angular 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    double _qe = accessor->get_pmtid_qe_angular( pmtid, energy, lposcost, minus_ ) 



What are the consequences of changing theEfficiency and where are they felt ?
---------------------------------------------------------------------------------

* consequence is the proportions with which theStatus gets set to Absorption vs Detection
* felt for custom handled boundary intersects



C4OpBoundaryProcess.cc::

     603     // SCB treat m_custom_status:Y as a kinda boundary type 
     604     // in order to provide  : Absorption-or-Detection/FresnelReflect/FresnelRefract
     605     if( m_custom_status == 'Y' )
     606     {
     607         G4double rand = G4UniformRand();
     608 
     609         if ( rand < theAbsorption )
     610         {
     611             DoAbsorption();   // theStatus is set to Detection/Absorption depending on a random and theEfficiency  
     612         }
     613         else


C4OpBoundaryProcess.hh::

    323 inline
    324 void C4OpBoundaryProcess::DoAbsorption()
    325 {
    326               theStatus = Absorption;
    327 
    328               if ( G4BooleanRand(theEfficiency) ) {
    329 
    330                  // EnergyDeposited =/= 0 means: photon has been detected
    331                  theStatus = Detection;
    332                  aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
    333               }
    334               else {
    335                  aParticleChange.ProposeLocalEnergyDeposit(0.0);
    336               }
    337 
    338               NewMomentum = OldMomentum;
    339               NewPolarization = OldPolarization;
    340 
    341 //              aParticleChange.ProposeEnergy(0.0);
    342               aParticleChange.ProposeTrackStatus(fStopAndKill);
    343 }



How an equivalent calc is done on GPU within Opticks ?
----------------------------------------------------------

::

    1731 inline QSIM_METHOD int qsim::propagate_at_surface_CustomART(unsigned& flag, curandStateXORWOW& rng, sctx& ctx) const
    1732 {
    1733 
    1734     const sphoton& p = ctx.p ;
    1735     const float3* normal = (float3*)&ctx.prd->q0.f.x ;  // geometrical outwards normal 
    1736     int lpmtid = ctx.prd->identity() - 1 ;  // identity comes from optixInstance.instanceId where 0 means not-a-sensor  
    1737     //int lpmtid = p.identity ; 
    1738 
    1739     float minus_cos_theta = dot(p.mom, *normal);
    1740     float dot_pol_cross_mom_nrm = dot(p.pol,cross(p.mom,*normal)) ;
    1759     if(lpmtid < 0 )
    1760     {
    1761         flag = NAN_ABORT ;
    1766         return BREAK ;
    1767     }
    1769 
    1770     float ARTE[4] ;
    1771     if(lpmtid > -1) pmt->get_lpmtid_ARTE(ARTE, lpmtid, p.wavelength, minus_cos_theta, dot_pol_cross_mom_nrm );
    1772 
    ....
    1780     const float& theAbsorption = ARTE[0];
    1781     //const float& theReflectivity = ARTE[1]; 
    1782     const float& theTransmittance = ARTE[2];
    1783     const float& theEfficiency = ARTE[3];
    1784 
    1785     float u_theAbsorption = curand_uniform(&rng);
    1786     int action = u_theAbsorption < theAbsorption  ? BREAK : CONTINUE ;
    ....
    1795     if( action == BREAK )
    1796     {
    1797         float u_theEfficiency = curand_uniform(&rng) ;
    1798         flag = u_theEfficiency < theEfficiency ? SURFACE_DETECT : SURFACE_ABSORB ;
    1799     }
    1800     else
    1801     {
    1802         propagate_at_boundary( flag, rng, ctx, theTransmittance  );
    1803     }
    1804     return action ;
    1805 }




Q: add lposcost arg to pmt->get_lpmtid_ARTE ?
A: its available from ctx.prd 


::

    291 /**
    292 qpmt::get_lpmtid_ARTE
    293 -----------------------
    294 
    295 lpmtid and polarization customized TMM calc of::
    296 
    297    theAbsorption
    298    theReflectivity
    299    theTransmittance
    300    theEfficiency 
    301 
    302 **/
    303 
    304 template<typename F>
    305 inline QPMT_METHOD void qpmt<F>::get_lpmtid_ARTE(
    306     F* arte4,
    307     int lpmtid,
    308     F wavelength_nm,
    309     F minus_cos_theta,
    310     F dot_pol_cross_mom_nrm ) const
    311 {
    312     const F energy_eV = hc_eVnm/wavelength_nm ;
    313 
    314     F spec[16] ;
    315     get_lpmtid_stackspec( spec, lpmtid, energy_eV );
    316 
    317     const F* ss = spec ;
    318     const F& _qe = spec[15] ;

    ///     HMM : THIS IS GPU ONLY METH ?

    319 
    320 #ifdef MOCK_CURAND_DEBUG
    321     printf("//qpmt::get_lpmtid_ARTE lpmtid %d energy_eV %7.3f _qe %7.3f \n", lpmtid, energy_eV, _qe );
    322 #endif
    323 
    324 
    325     Stack<F,4> stack ;
    326 
    327     if( minus_cos_theta < zero )
    328     {   
    329         stack.calc(wavelength_nm, -one, zero, ss, 16u );
    330         arte4[3] = _qe/stack.art.A ;
    331 
    332 #ifdef MOCK_CURAND_DEBUG
    333         printf("//qpmt::get_lpmtid_ARTE stack.art.A %7.3f _qe/stack.art.A %7.3f \n", stack.art.A, arte4[3] );
    334 #endif
    335     }
    336     else
    337     {
    338         arte4[3] = zero ;
    339     }
    340 
    341     stack.calc(wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm, ss, 16u );
    342 
    343     const F& A = stack.art.A ;
    344     const F& R = stack.art.R ;
    345     const F& T = stack.art.T ;




::

    052 template<typename F>
     53 struct qpmt
     54 {
     55     enum { L0, L1, L2, L3 } ;
     56 
     57     static constexpr const F hc_eVnm = 1239.84198433200208455673  ;
     58     static constexpr const F zero = 0. ;
     59     static constexpr const F one = 1. ;
     60     // constexpr should mean any double conversions happen at compile time ?
     61 
     62     qprop<F>* rindex_prop ;
     63     qprop<F>* qeshape_prop ;
     64 
     65     F*        thickness ;
     66     F*        lcqs ;
     67     int*      i_lcqs ;  // int* "view" of lcqs memory
     68 


    156 template<typename F>
    157 inline QPMT_METHOD void qpmt<F>::get_lpmtid_stackspec( F* spec, int lpmtid, F energy_eV ) const
    158 {
    159     const int& lpmtcat = i_lcqs[lpmtid*2+0] ;
    160     // printf("//qpmt::get_lpmtid_stackspec lpmtid %d lpmtcat %d \n", lpmtid, lpmtcat );  
    161 
    162     const F& qe_scale = lcqs[lpmtid*2+1] ;
    163     const F qe_shape = qeshape_prop->interpolate( lpmtcat, energy_eV ) ;
    164     const F qe = qe_scale*qe_shape ;
    165 
    166     spec[0*4+3] = lpmtcat ;
    167     spec[1*4+3] = qe_scale ;
    168     spec[2*4+3] = qe_shape ;
    169     spec[3*4+3] = qe ;
    170 
    171     get_lpmtcat_stackspec( spec, lpmtcat, energy_eV );
    172 }



QPMT handles uploading, eg::

    130 template<typename T>
    131 inline void QPMT<T>::init_lcqs()
    132 {
    133     LOG(LEVEL)
    134        << " src_lcqs " << ( src_lcqs ? src_lcqs->sstr() : "-" )
    135        << " lcqs " << ( lcqs ? lcqs->sstr() : "-" )
    136        ;
    137 
    138     const char* label = "QPMT::init_lcqs/d_lcqs" ;
    139 
    140 #if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    141     T* d_lcqs = lcqs ? const_cast<T*>(lcqs->cvalues<T>()) : nullptr ;
    142 #else
    143     T* d_lcqs = lcqs ? QU::UploadArray<T>(lcqs->cvalues<T>(), lcqs->num_values(), label) : nullptr ;
    144 #endif
    145 
    146     pmt->lcqs = d_lcqs ;
    147     pmt->i_lcqs = (int*)d_lcqs ;   // HMM: would cause issues with T=double  
    148 }


It is booted from an NPFold::

     31 template<typename T>
     32 struct QUDARAP_API QPMT
     33 {
     34     static const plog::Severity LEVEL ;
     35     static const QPMT<T>*    INSTANCE ;
     36     static const QPMT<T>*    Get();
     37 
     38     static std::string Desc();
     39 
     40     const char* ExecutableName ;
     41 
     42     const NP* src_rindex ;    // (NUM_PMTCAT, NUM_LAYER, NUM_PROP, NEN, 2:[energy,value] )
     43     const NP* src_thickness ; // (NUM_PMTCAT, NUM_LAYER, 1:value )  
     44     const NP* src_qeshape ;   // (NUM_PMTCAT, NEN_SAMPLES~44, 2:[energy,value] )
     45     const NP* src_lcqs ;      // (NUM_LPMT, 2:[cat,qescale])
     46 
     47     const NP* rindex3 ;       // (NUM_PMTCAT*NUM_LAYER*NUM_PROP,  NEN, 2:[energy,value] )
     48     const NP* rindex ;
     49     const QProp<T>* rindex_prop ;
     50 
     51     const NP* qeshape ;
     52     const QProp<T>* qeshape_prop ;
     53 
     54     const NP* thickness ;
     55     const NP* lcqs ;
     56     const int* i_lcqs ;  // CPU side lpmtid -> lpmtcat 0/1/2
     57 
     58     qpmt<T>* pmt ;
     59     qpmt<T>* d_pmt ;
     60 
     61     // .h 
     62     QPMT(const NPFold* pf);
     63 



With the fold coming from SSim::

     100 void QSim::UploadComponents( const SSim* ssim  )
     101 {
     ...
     179     const NPFold* spmt_f = ssim->get_spmt_f() ;
     180     QPMT<float>* qpmt = spmt_f ? new QPMT<float>(spmt_f) : nullptr ;
     181     LOG_IF(LEVEL, qpmt == nullptr )
     182         << " NO QPMT instance "
     183         << " spmt_f " << ( spmt_f ? "YES" : "NO " )
     184         << " qpmt " << ( qpmt ? "YES" : "NO " )
     185         ;
     186 



1. raw jpmt NPFold (direct serializations of PMT data) used to boot SPMT instance 
2. SPMT instance summarizes PMT info into just whats needed GPU side
3. Serialized SPMT used to create QPMT instance

::

    055     static constexpr const char* JPMT_RELP = "extra/jpmt" ;


    310 const NPFold* SSim::get_jpmt() const
    311 {
    312     const NPFold* f = top ? top->find_subfold(JPMT_RELP) : nullptr ;
    313     return f ;
    314 }
    315 const SPMT* SSim::get_spmt() const
    316 {
    317     const NPFold* jpmt = get_jpmt();
    318     return jpmt ? new SPMT(jpmt) : nullptr ;
    319 }
    320 const NPFold* SSim::get_spmt_f() const
    321 {
    322     const SPMT* spmt = get_spmt() ;
    323     const NPFold* spmt_f = spmt ? spmt->serialize() : nullptr ;
    324     return spmt_f ;
    325 }


::

    epsilon:jpmt blyth$ pwd
    /Users/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim/extra/jpmt
    epsilon:jpmt blyth$ l
    total 8
    0 -rw-rw-r--   1 blyth  staff    0 Nov 28  2023 NPFold_names.txt
    8 -rw-rw-r--   1 blyth  staff   40 Nov 28  2023 NPFold_index.txt
    0 drwxr-xr-x   7 blyth  staff  224 Nov 26  2023 .
    0 drwxr-xr-x   5 blyth  staff  160 Nov 26  2023 ..
    0 drwxr-xr-x  20 blyth  staff  640 Nov 26  2023 PMTSimParamData
    0 drwxr-xr-x   6 blyth  staff  192 Nov 26  2023 PMT_RINDEX
    0 drwxr-xr-x   5 blyth  staff  160 Nov 26  2023 PMTParamData
    epsilon:jpmt blyth$ 

    epsilon:jpmt blyth$ find . 
    .
    ./PMTSimParamData
    ./PMTSimParamData/lpmtData.npy
    ./PMTSimParamData/pmtTotal.npy
    ./PMTSimParamData/MPT
    ./PMTSimParamData/MPT/000
    ./PMTSimParamData/MPT/000/PHC_KINDEX.npy
    ./PMTSimParamData/MPT/000/PHC_RINDEX.npy
    ./PMTSimParamData/MPT/000/ARC_KINDEX.npy
    ./PMTSimParamData/MPT/000/NPFold_index.txt
    ./PMTSimParamData/MPT/000/ARC_RINDEX.npy
    ./PMTSimParamData/MPT/000/NPFold_names.txt
    ./PMTSimParamData/MPT/001
    ./PMTSimParamData/MPT/001/PHC_KINDEX.npy
    ./PMTSimParamData/MPT/001/PHC_RINDEX.npy
    ./PMTSimParamData/MPT/001/ARC_KINDEX.npy
    ./PMTSimParamData/MPT/001/NPFold_index.txt
    ./PMTSimParamData/MPT/001/ARC_RINDEX.npy
    ./PMTSimParamData/MPT/001/NPFold_names.txt
    ./PMTSimParamData/MPT/NPFold_index.txt
    ./PMTSimParamData/MPT/003
    ./PMTSimParamData/MPT/003/PHC_KINDEX.npy
    ./PMTSimParamData/MPT/003/PHC_RINDEX.npy
    ./PMTSimParamData/MPT/003/ARC_KINDEX.npy
    ./PMTSimParamData/MPT/003/NPFold_index.txt
    ./PMTSimParamData/MPT/003/ARC_RINDEX.npy
    ./PMTSimParamData/MPT/003/NPFold_names.txt
    ./PMTSimParamData/MPT/NPFold_names.txt
    ./PMTSimParamData/pmtCat.npy
    ./PMTSimParamData/QEshape
    ./PMTSimParamData/QEshape/QEshape_NNVT_HiQE.npy
    ./PMTSimParamData/QEshape/QEshape_WP_PMT.npy
    ./PMTSimParamData/QEshape/QEshape_R12860.npy
    ./PMTSimParamData/QEshape/QEshape_NNVT.npy
    ./PMTSimParamData/QEshape/QEshape_HZC.npy
    ./PMTSimParamData/QEshape/NPFold_index.txt
    ./PMTSimParamData/QEshape/NPFold_names.txt
    ./PMTSimParamData/pmtCatVec.npy
    ./PMTSimParamData/pmtCatName_names.txt
    ./PMTSimParamData/CONST
    ./PMTSimParamData/CONST/001_names.txt
    ./PMTSimParamData/CONST/000_names.txt
    ./PMTSimParamData/CONST/003.npy
    ./PMTSimParamData/CONST/000.npy
    ./PMTSimParamData/CONST/001.npy
    ./PMTSimParamData/CONST/003_names.txt
    ./PMTSimParamData/CONST/NPFold_index.txt
    ./PMTSimParamData/CONST/NPFold_names.txt
    ./PMTSimParamData/pmtID.npy
    ./PMTSimParamData/NPFold_index.txt
    ./PMTSimParamData/spmtData_meta.txt
    ./PMTSimParamData/spmtData.npy
    ./PMTSimParamData/pmtTotal_names.txt
    ./PMTSimParamData/qeScale.npy
    ./PMTSimParamData/lpmtCat_meta.txt
    ./PMTSimParamData/lpmtCat.npy
    ./PMTSimParamData/NPFold_names.txt
    ./PMTSimParamData/pmtCatName.npy
    ./NPFold_index.txt
    ./NPFold_names.txt
    ./PMTParamData
    ./PMTParamData/pmtCat.npy
    ./PMTParamData/NPFold_index.txt
    ./PMTParamData/NPFold_names.txt
    ./PMT_RINDEX
    ./PMT_RINDEX/NPFold_index.txt
    ./PMT_RINDEX/PyrexRINDEX.npy
    ./PMT_RINDEX/NPFold_names.txt
    ./PMT_RINDEX/VacuumRINDEX.npy
    epsilon:jpmt blyth$ 



Hmm CPU side duplication?::

     863 inline void SPMT::get_ARTE(
     864     SPMTData& pd,
     865     int   lpmtid,
     866     float wavelength_nm,
     867     float minus_cos_theta,
     868     float dot_pol_cross_mom_nrm ) const
     869 {
     870     const float energy_eV = hc_eVnm/wavelength_nm ;
     871     get_lpmtid_stackspec(pd.spec, lpmtid, energy_eV);
     872 
     873     const float* ss = pd.spec.cdata() ;
     874     const float& _qe = ss[15] ;
     875 
     876     pd.args.x = lpmtid ;
     877     pd.args.y = energy_eV ;
     878     pd.args.z = minus_cos_theta ;
     879     pd.args.w = dot_pol_cross_mom_nrm ;
     880 
     881     if( minus_cos_theta < 0.f ) // only ingoing photons 
     882     {
     883         pd.stack.calc(wavelength_nm, -1.f, 0.f, ss, 16u );
     884         pd.ARTE.w = _qe/pd.stack.art.A ;  // aka theEfficiency and escape_fac, no mct dep 
     885 
     886         pd.extra.x = 1.f - (pd.stack.art.T_av + pd.stack.art.R_av ) ;  // old An
     887         pd.extra.y = pd.stack.art.A_av ;
     888         pd.extra.z = pd.stack.art.A   ;
     889         pd.extra.w = pd.stack.art.A_s ;
     890     }
     891     else
     892     {
     893         pd.ARTE.w = 0.f ;
     894     }
     895 
     896     pd.stack.calc(wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm, ss, 16u );
     897 

