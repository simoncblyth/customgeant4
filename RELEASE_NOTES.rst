Custom4 Release Notes
========================


0.1.9
------

* untested compilation fixes regarding the implementation version switching

0.1.8
-------

* minor change in `ART_` struct : switching [:,3,2] from A_av to dot_pol_cross_mom_nrm to assist testing
* remove obsolete `OLD_ART_` struct 
* add tests/C4MultiLayrStack_Test2.sh comparing results from pypi tmm package with this package : get near perfect match
* add tests/V.h minimal vector methods needed by C4MultiLayrStack_Test2.cc

0.1.7 
------

* no result changes from this release
* improve documentation of tagging and development workflows 
* util.sh script for tarball curl and scp to workstation 

0.1.6
-------

* fixes polarization S/P power fraction (E_s2) bug 
* adds debug dumping 

0.1.5
------

* adds versioning with CMake generated C4Version.h
* build.sh adopts separate CMAKE_INSTALL_PREFIX 
 

0.1.4 and prior
-----------------

Versioning was only fully setup in 0.1.5 thus to use versioning 
before that requires some manual kludging. 
For example adding a minimal C4Version.h into the appropriate include dir,
will allow macro version branching to work::

    echo "#define Custom4_VERSION_NUMBER 00104" > /usr/local/opticks_externals/custom4/0.1.4/lib/Custom4-0.1.4/C4Version.h


