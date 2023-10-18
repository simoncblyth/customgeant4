Custom4 : Geant4 Customizations
=================================

This Custom4 mini-package was created to avoid circular dependency 
between opticks and junosw by splitting off classes/struct that 
depend only on Geant4 so as to allow high level communication 
between opticks and junosw in the "language" provided by Custom4. 

Classes/Structs
------------------

C4OpBoundaryProcess.hh
   modified G4OpBoundaryProcess with customized calculation 
   of absorption, reflection, transmissing coefficients
   using C4CustomART.h

C4CustomART.h
   integrates between the boundary process and the TMM calculation

C4MultiLayrStack.h
   TMM (transfer-matrix method) calculation of absorption, reflection and transmission 
   (ART) coefficients based on complex refractive indices and layer thicknesses 

C4IPMTAccessor.h
   pure virtual protocol interface for providing PMT information 
   including layer refractive indices and thicknesses to the boundary process 
    
C4CustomART_Debug.h
   debug struct with serialization to std::array 

C4CustomStatus.h
   notes on custom status char set by C4OpBoundaryProcess::PostStepDoIt

C4Pho.h
   photon label (equivalent to opticks sysrap/spho.h)  

C4TrackInfo.h
   utility for G4TrackInfo manipulations  

C4Touchable.h
   utility for G4Touchable manipulations 

C4Sys.h
   general static functions 



The custom status char is set by C4OpBoundaryProcess::PostStepDoIt

+------+-------------------------------------------------------------------------------+
| char |                                                                               |
+======+===============================================================================+
|  U   |  starting value set at initialization and at every step                       |
+------+-------------------------------------------------------------------------------+
|  Z   |  @/# OpticalSurface BUT local_z < 0 : so ordinary surface                     |         
+------+-------------------------------------------------------------------------------+
|  Y   |  @ OpticalSurface AND local_z > 0 : so C4CustomART::doIt runs                 |
+------+-------------------------------------------------------------------------------+
|  \-  |  # OpticalSurface AND local_z > 0 : so traditional detect at photocathode     |                
+------+-------------------------------------------------------------------------------+
|  X   |  NOT @/# OpticalSurface : so ordinary surface                                 | 
+------+-------------------------------------------------------------------------------+
|  \0  |  Uninitialized array content                                                  |
+------+-------------------------------------------------------------------------------+


References to Opticks and other related packages 
--------------------------------------------------

* https://simoncblyth.bitbucket.io/
* https://simoncblyth.github.io/


How to tag a Custom4 release
-------------------------------

::

    c4
    vi CMakeLists.txt      # bump the version 

    git commit ..            
    git add ..              
    git push ..            # update code on github  

    ./addtag.sh 
    ./addtag.sh | sh       # update tags on github     





How to update the Custom4 version used by junosw
--------------------------------------------------

See notes in $JUNOTOP/junoenv/packages/custom4.sh 



