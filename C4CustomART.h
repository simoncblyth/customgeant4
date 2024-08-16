#pragma once
/**
C4CustomART : Used by C4OpBoundaryProcess for Custom calc of A/R/T coeffs
==================================================================================

* CustomART is instanciated by the CustomG4OpBoundaryProcess ctor
* aims to provides customization with minimal code change to CustomG4OpBoundaryProcess 
* aims to make few detector specific assumptions by moving such specifics into 
  the PMTAccessor which is accessed via IPMTAccessor protocol 

What detector/Geant4 geometry specific assumptions are made ?
---------------------------------------------------------------

1. ART customized for surfaces with names starting with '@' on local_z > 0. 
2. pmtid obtained from Geant4 volume ReplicaNumber
 

Headers used by CustomART.h were developed and tested elsewhere
------------------------------------------------------------------

MultiLayrStack.h 
    Developed as Layr.h at https://github.com/simoncblyth/j/blob/main/Layr/Layr.h

S4Touchable
    Developed as U4Touchable.h at https://bitbucket.org/simoncblyth/opticks/src/master/u4/U4Touchable.h 

Overview
----------

CustomART::doIt only sets the below within the host CustomG4OpBoundaryProcess, 
via reference ctor arguments::

    theTransmittance
    theReflectivity
    theEfficiency 


Is 2-layer (Pyrex,Vacuum) polarization direction calc applicable to 4-layer (Pyrex,ARC,PHC,Vacuum) situation ? 
-----------------------------------------------------------------------------------------------------------------

The Geant4 calculation of polarization direction for a simple
boundary between two layers is based on continuity of E and B fields
in S and P directions at the boundary, essentially Maxwells Eqn boundary conditions.
Exactly the same thing yields Snells law::

    n1 sin t1  = n2 sin t2 

My thinking on this is that Snell's law with a bunch of layers would be::

    n1 sin t1 = n2 sin t2 =  n3 sin t3 =  n4 sin t4 

So the boundary conditions from 1->4 where n1 and n4 are real still gives::

    n1 sin t1 = n4 sin t4

Even when n2,t2,n3,t3 are complex.

So by analogy that makes me think that the 2-layer polarization calculation 
between layers 1 and 4 (as done by G4OpBoundaryProcess) 
should still be valid even when there is a stack of extra layers 
inbetween layer 1 and 4. 

Essentially the stack calculation changes A,R,T so it changes
how much things happen : but it doesnt change what happens. 
So the two-layer polarization calculation from first and last layer 
should still be valid to the situation of the stack.

Do you agree with this argument ? 

**/

#include "G4ThreeVector.hh"

#include "C4IPMTAccessor.h"
#include "C4MultiLayrStack.h"  
#include "C4Touchable.h"

#ifdef C4_DEBUG
#include "C4CustomART_Debug.h"
#endif

struct C4CustomART
{
    bool   dump ; 
    int    count ; 
    const C4IPMTAccessor* accessor ; 
    int    implementation_version ;    

    G4double& theAbsorption ;
    G4double& theReflectivity ;
    G4double& theTransmittance ;
    G4double& theEfficiency ;

    const G4ThreeVector& theGlobalPoint ; 
    const G4ThreeVector& OldMomentum ; 
    const G4ThreeVector& OldPolarization ; 
    const G4ThreeVector& theRecoveredNormal ; 
    const G4double& thePhotonMomentum ; 

    G4ThreeVector localPoint ; 
#ifdef C4_DEBUG
    C4CustomART_Debug dbg ;  
#endif

    C4CustomART(
        const C4IPMTAccessor* accessor, 
        G4double& theAbsorption,
        G4double& theReflectivity,
        G4double& theTransmittance,
        G4double& theEfficiency,
        const G4ThreeVector& theGlobalPoint,  
        const G4ThreeVector& OldMomentum,  
        const G4ThreeVector& OldPolarization,
        const G4ThreeVector& theRecoveredNormal,
        const G4double& thePhotonMomentum
    );  

    void   update_local_position( const G4Track& aTrack ); 
    double local_z() const ;
    double local_cost() const ;
    void doIt(const G4Track& aTrack, const G4Step& ); 
    std::string desc() const ; 

}; 

inline C4CustomART::C4CustomART(
    const C4IPMTAccessor* accessor_, 
          G4double& theAbsorption_,
          G4double& theReflectivity_,
          G4double& theTransmittance_,
          G4double& theEfficiency_,
    const G4ThreeVector& theGlobalPoint_,
    const G4ThreeVector& OldMomentum_,
    const G4ThreeVector& OldPolarization_,
    const G4ThreeVector& theRecoveredNormal_,
    const G4double&      thePhotonMomentum_
    )
    :
    dump(false),
    count(0),
    accessor(accessor_),
    implementation_version(accessor->get_implementation_version()),
    theAbsorption(theAbsorption_),
    theReflectivity(theReflectivity_),
    theTransmittance(theTransmittance_),
    theEfficiency(theEfficiency_),
    theGlobalPoint(theGlobalPoint_),
    OldMomentum(OldMomentum_),
    OldPolarization(OldPolarization_),
    theRecoveredNormal(theRecoveredNormal_),
    thePhotonMomentum(thePhotonMomentum_),
    localPoint(0.,0.,0.)
{
}

/**
C4CustomART::update_local_position
-----------------------------------

Call *update_local_position* once for each track before using 
other methods such as *local_z*

Typically this is done prior to *doIt* to check the 
hemisphere to decide if *doIt* needs to be called.  

**/


inline void C4CustomART::update_local_position( const G4Track& aTrack )
{
    const G4AffineTransform& transform = aTrack.GetTouchable()->GetHistory()->GetTopTransform();
    localPoint = transform.TransformPoint(theGlobalPoint);
}



/**
C4CustomART::local_z
-------------------------

**/


inline double C4CustomART::local_z() const
{
    return localPoint.z() ; 
}

/**
C4CustomART::local_cost (aka lposcost)
-------------------------------------------

Q:What is lposcost for ?  

A:Preparing for doing this on GPU, as lposcost is available there already but zlocal is not, 
  so want to check the sign of lposcost is following that of zlocal. It looks 
  like it should:: 

    157 inline double Hep3Vector::cosTheta() const {
    158   double ptot = mag();
    159   return ptot == 0.0 ? 1.0 : dz/ptot;
    160 }

**/

inline double C4CustomART::local_cost() const
{
    return localPoint.cosTheta() ; 
}



/**
C4CustomART::doIt
--------------------

Called from CustomG4OpBoundaryProcess::PostStepDoIt only for optical surfaces
with names starting with the special prefix character '@'  

NB stack is flipped for minus_cos_theta > 0. so:

* stack.ll[0] always incident side
* stack.ll[3] always transmission side 

Why theEfficiency is _qe/An ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider factorization of _qe obtained from *get_pmtid_qe* as function of pmtid and energy::

     _qe = An * escape_fac  

An
    Normal incidence absorption (from An = 1-Rn-Tn) 
    depending on aoi, wl, material properties of the stack including 
    complex refractive indices and thicknesses of PHC and ARC.
    (Fresnel calc, ie Maxwells EM boundary conditions) 

escape_fac
    fraction of photoelectrons that escape and travel from cathode to dynode/MCP and form signal,
    depending on fields inside the PMT vacuum, shape of dynodes/MCP etc. (does not depend on aoi).
    Within the simulation this escape_fac is taken to express the fraction 
    of absorbed photons that are detected. 

    Currently there appears be be an assumption that escape_fac does not depend on the position on the PMT. 
    Some position angle dependence (not same as aoi) is a possibility. 

See https://link.springer.com/article/10.1140/epjc/s10052-022-10288-y "A new optical model for photomultiplier tubes"

From that paper it seems the _qe seems is obtained from normal incidence measurements in LAB
(linear alkylbenzene) which has refractive index close to Pyrex, so there is very little reflection 
on LAB/Pyrex boundary and hence the calculated normal incidence absorption An from the 4 layer 
stack Pyrex/ARC/PHC/Vacuum at normal incidence can be compared with the measured _qe 
in order to provide a non-aoi dependent escape_fac factor expressing the fraction of 
absorbed photons that are regarded as being detected. 


3-way ART to 2-way RT probability rescaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
A,R,T are 3-way probabilities summing to 1. 
BUT CustomG4OpBoundaryProcess::DielectricDielectric expects  
*theTransmittance* to be a 2-way Transmit-or-Reflect probability. 
So the 3-way probabilities are scaled into 2-way ones, eg::

    theAbsorption = A ; 
    theReflectivity  = R/(1.-A) ; 
    theTransmittance = T/(1.-A)  ;   


    (A, R, T)   ( 0.5,  0.25,   0.25 )    

                        0.25        0.25
     ->( R, T)   (    --------  ,  -------- ) =  ( 0.5, 0.5 )
                       (1-0.5)     (1-0.5) ) 


E_s2 : how that corresponds to polarization power fraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     mom       nrm
         +--s--+
          \    |
           \   | 
     pol.   \  |  
             \ | 
              \|
     ----------0-------

OldMomentum.cross(theRecoveredNormal) 
    transverse direction, eg out the page 
    (OldMomentum, theRecoveredNoraml are normalized, 
    so magnitude will be sine of angle between mom and nrm) 

(OldPolarization*OldMomentum.cross(theRecoveredNormal)) 
    dot product between the OldPolarization and transverse direction
    is expressing the S polarization fraction
    (OldPolarization is normalized so the magnitude will be 
     cos(angle-between-pol-and-transverse)*sin(angle-between-mom-and-nrm)

    * hmm pulling out "pol_dot_mom_cross_nrm" argument would provide some splitting 
    * dot product with a cross product is the determinant of the three vectors, 
      thats the volume of the parallelopiped formed by the vectors 
      

stack.ll[0].st.real()
    thus is just sqrt(1.-mct*mct) so its "st" sine(angle-between-mom-and-normal)
   
    * mct is OldMomentum*theRecoveredNormal (both those are normalized) 
    * no need to involve stack or stackspec  

(OldPolarization*OldMomentum.cross(theRecoveredNormal))/stack.ll[0].st.real()
    division by "st" brings this to cos(angle-betweel-pol-and-tranverse)
    so it does represent the S polarization fraction 

**/

inline void C4CustomART::doIt(const G4Track& aTrack, const G4Step& )
{
    G4double zero = 0. ; 
    G4double minus_one = -1. ; 
    G4double minus_cos_theta = OldMomentum*theRecoveredNormal ; 
    G4double dot_pol_cross_mom_nrm = OldPolarization*OldMomentum.cross(theRecoveredNormal) ; 

    G4double energy = thePhotonMomentum ; 
    G4double wavelength = CLHEP::twopi*CLHEP::hbarc/energy ;
    G4double energy_eV = energy/CLHEP::eV ;
    G4double wavelength_nm = wavelength/CLHEP::nm ; 

    int pmtid = C4Touchable::VolumeIdentifier(&aTrack, true ); 
    int pmtcat = accessor->get_pmtcat( pmtid ) ; 

    std::array<double,16> a_spec ; 
    accessor->get_stackspec(a_spec, pmtcat, energy_eV ); 
    const double* ss = a_spec.data() ; 

    Stack<double,4> stack ; 
    theEfficiency = zero ;
    double _qe = zero ; 
    double lposcost = local_cost();  

    if(implementation_version == 0 )
    {
        _qe = minus_cos_theta > 0. ? 0.0 : accessor->get_pmtid_qe( pmtid, energy ) ;  // energy_eV ?
        // following the old junoPMTOpticalModel with "backwards" _qe always zero 
        if( minus_cos_theta < zero ) 
        {
            // normal incidence calc only needed for ingoing photons as _qe is fixed to zero for outgoing  
            stack.calc( wavelength_nm, minus_one, zero, ss, 16u );  // normal incidence calc 
            theEfficiency = _qe/stack.art.A ;                       // aka escape_fac
        }
    }
    else
    {
        // allowing non-zero "backwards" _qe means must do norml incidence calc every time as need theEfficiency
        _qe = accessor->get_pmtid_qe_angular( pmtid, energy, lposcost, minus_cos_theta ) ; 
        stack.calc( wavelength_nm, minus_one      , zero, ss, 16u );  // for normal incidence efficiency 
        theEfficiency = _qe/stack.art.A ;                       // aka escape_fac
    }

#ifdef C4_DEBUG
    dbg.An = stack.art.A ; 
    dbg.Rn = stack.art.R  ; 
    dbg.Tn = stack.art.T  ; 
    dbg.escape_fac = theEfficiency ; 
#endif
    stack.calc( wavelength_nm, minus_cos_theta, dot_pol_cross_mom_nrm, ss, 16u );  



    bool expect = theEfficiency <= 1. ; 
    if(!expect) std::cerr
        << "C4CustomART::doIt"
        << " implementation_version " << implementation_version
        << " FATAL "
        << " ERR: theEfficiency > 1. : " << theEfficiency
        << " _qe " << _qe
        << " lposcost " << lposcost
        << " stack.art.A (aka An) " << stack.art.A 
        << std::endl 
        ;
    assert( expect ); 

    const double& A = stack.art.A ; 
    const double& R = stack.art.R ; 
    const double& T = stack.art.T ; 

    theAbsorption = A ; 
    theReflectivity  = R/(1.-A) ; 
    theTransmittance = T/(1.-A)  ;   

    if(dump) std::cerr   
        << "C4CustomART::doIt"
        << " implementation_version " << implementation_version
        << std::endl 
        << " pmtid " << pmtid << std::endl 
        << " _qe                      : " << std::fixed << std::setw(10) << std::setprecision(4) << _qe  << std::endl 
        << " minus_cos_theta          : " << std::fixed << std::setw(10) << std::setprecision(4) << minus_cos_theta  << std::endl 
        << " dot_pol_cross_mom_nrm    : " << std::fixed << std::setw(10) << std::setprecision(4) << dot_pol_cross_mom_nrm  << std::endl 
        << " lposcost                 : " << std::fixed << std::setw(10) << std::setprecision(4) << lposcost << std::endl 
        << std::endl 
        << " stack " 
        << std::endl 
        << stack 
        << std::endl 
        << " theAbsorption    : " << std::fixed << std::setw(10) << std::setprecision(4) << theAbsorption  << std::endl 
        << " theReflectivity  : " << std::fixed << std::setw(10) << std::setprecision(4) << theReflectivity  << std::endl 
        << " theTransmittance : " << std::fixed << std::setw(10) << std::setprecision(4) << theTransmittance  << std::endl 
        << " theEfficiency    : " << std::fixed << std::setw(10) << std::setprecision(4) << theEfficiency  << std::endl 
        ;


#ifdef C4_DEBUG
    dbg.A = A ; 
    dbg.R = R ; 
    dbg.T = T ; 
    dbg._qe = _qe ; 

    dbg.minus_cos_theta = minus_cos_theta ; 
    dbg.wavelength_nm   = wavelength_nm ; 
    dbg.pmtid           = double(pmtid) ; 
    dbg.lposcost        = lposcost ; 
#endif

    count += 1 ; 
}



inline std::string C4CustomART::desc() const 
{
    std::stringstream ss ; 
    ss << "C4CustomART::desc"
       << " count " << std::setw(4) << count 
       << " theGlobalPoint " << theGlobalPoint 
       << " theRecoveredNormal " << theRecoveredNormal 
       << " theAbsorption " << std::fixed << std::setw(10) << std::setprecision(5) << theAbsorption
       << " theReflectivity " << std::fixed << std::setw(10) << std::setprecision(5) << theReflectivity
       << " theTransmittance " << std::fixed << std::setw(10) << std::setprecision(5) << theTransmittance
       << " theEfficiency " << std::fixed << std::setw(10) << std::setprecision(5) << theEfficiency
       ;
    std::string str = ss.str(); 
    return str ; 
}

