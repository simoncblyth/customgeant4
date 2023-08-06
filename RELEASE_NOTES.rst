Custom4 Release Notes
========================


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


