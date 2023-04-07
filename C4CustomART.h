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
    int    count ; 
    double zlocal ; 
#ifdef C4_DEBUG
    double lposcost ; 
#endif
    const C4IPMTAccessor* accessor ; 

    G4double& theAbsorption ;
    G4double& theReflectivity ;
    G4double& theTransmittance ;
    G4double& theEfficiency ;

    const G4ThreeVector& theGlobalPoint ; 
    const G4ThreeVector& OldMomentum ; 
    const G4ThreeVector& OldPolarization ; 
    const G4ThreeVector& theRecoveredNormal ; 
    const G4double& thePhotonMomentum ; 

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

    double local_z( const G4Track& aTrack ); 
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
    count(0),
    zlocal(-1.),
#ifdef C4_DEBUG
    lposcost(-2.),
#endif
    accessor(accessor_),
    theAbsorption(theAbsorption_),
    theReflectivity(theReflectivity_),
    theTransmittance(theTransmittance_),
    theEfficiency(theEfficiency_),
    theGlobalPoint(theGlobalPoint_),
    OldMomentum(OldMomentum_),
    OldPolarization(OldPolarization_),
    theRecoveredNormal(theRecoveredNormal_),
    thePhotonMomentum(thePhotonMomentum_)
{
}

/**
C4CustomART::local_z
-------------------------

Q:What is lposcost for ?  

A:Preparing for doing this on GPU, as lposcost is available there already but zlocal is not, 
  so want to check the sign of lposcost is following that of zlocal. It looks 
  like it should:: 

    157 inline double Hep3Vector::cosTheta() const {
    158   double ptot = mag();
    159   return ptot == 0.0 ? 1.0 : dz/ptot;
    160 }


**/

inline double C4CustomART::local_z( const G4Track& aTrack )
{
    const G4AffineTransform& transform = aTrack.GetTouchable()->GetHistory()->GetTopTransform();
    G4ThreeVector localPoint = transform.TransformPoint(theGlobalPoint);
    zlocal = localPoint.z() ; 
#ifdef C4_DEBUG
    lposcost = localPoint.cosTheta() ;  
#endif
    return zlocal  ; 
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

**/

inline void C4CustomART::doIt(const G4Track& aTrack, const G4Step& )
{
    G4double minus_cos_theta = OldMomentum*theRecoveredNormal ; 
    G4double energy = thePhotonMomentum ; 
    G4double wavelength = CLHEP::twopi*CLHEP::hbarc/energy ;
    G4double energy_eV = energy/CLHEP::eV ;
    G4double wavelength_nm = wavelength/CLHEP::nm ; 

    int pmtid = C4Touchable::VolumeIdentifier(&aTrack, true ); 
    int pmtcat = accessor->get_pmtcat( pmtid ) ; 
    double _qe = minus_cos_theta > 0. ? 0.0 : accessor->get_pmtid_qe( pmtid, energy ) ;  
    // following the old junoPMTOpticalModel with "backwards" _qe always zero 

    std::array<double,16> a_spec ; 
    accessor->get_stackspec(a_spec, pmtcat, energy_eV ); 
    StackSpec<double,4> spec ; 
    spec.import( a_spec ); 

    Stack<double,4> stack(wavelength_nm, minus_cos_theta, spec );  

    const double _si = stack.ll[0].st.real() ; 
    double E_s2 = _si > 0. ? (OldPolarization*OldMomentum.cross(theRecoveredNormal))/_si : 0. ; 
    E_s2 *= E_s2;      

    // E_s2 : S-vs-P power fraction : signs make no difference as squared
    // E_s2 matches E1_perp*E1_perp see sysrap/tests/stmm_vs_sboundary_test.cc 

    double one = 1.0 ; 
    double S = E_s2 ; 
    double P = one - S ; 

    double T = S*stack.art.T_s + P*stack.art.T_p ;  // matched with TransCoeff see sysrap/tests/stmm_vs_sboundary_test.cc
    double R = S*stack.art.R_s + P*stack.art.R_p ;
    double A = S*stack.art.A_s + P*stack.art.A_p ;  
    //double A1 = one - (T+R);  // note that A1 matches A 

    theAbsorption = A ; 
    theReflectivity  = R/(1.-A) ; 
    theTransmittance = T/(1.-A)  ;   

    // stackNormal is not flipped (as minus_cos_theta is fixed at -1.) presumably this is due to _qe definition
    Stack<double,4> stackNormal(wavelength_nm, -1. , spec ); 

    // at normal incidence S/P distinction is meaningless, and the values converge anyhow : so no polarization worries here
    //double An = stackNormal.art.A ; 
    double An = one - (stackNormal.art.T + stackNormal.art.R) ; 
    double escape_fac = _qe/An;   
    theEfficiency = escape_fac ; 

    bool expect = theEfficiency <= 1. ; 
    if(!expect) std::cerr
        << "CustomART::doIt"
        << " FATAL "
        << " ERR: theEfficiency > 1. : " << theEfficiency
        << " _qe " << _qe
        << " An " << An
        << std::endl 
        ;
    assert( expect ); 

#ifdef C4_DEBUG
    dbg.A = A ; 
    dbg.R = R ; 
    dbg.T = T ; 
    dbg._qe = _qe ; 

    dbg.An = An ; 
    dbg.Rn = stackNormal.art.R  ; 
    dbg.Tn = stackNormal.art.T  ; 
    dbg.escape_fac = escape_fac ; 

    dbg.minus_cos_theta = minus_cos_theta ; 
    dbg.wavelength_nm   = wavelength_nm ; 
    dbg.pmtid           = double(pmtid) ; 
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

