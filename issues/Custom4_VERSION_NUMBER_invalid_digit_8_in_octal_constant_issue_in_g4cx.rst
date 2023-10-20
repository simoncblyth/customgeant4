Custom4_VERSION_NUMBER_invalid_digit_8_in_octal_constant_issue_in_g4cx
========================================================================


::

    === om-make-one : g4cx            /data/blyth/junotop/opticks/g4cx                             /data/blyth/junotop/ExternalLibs/opticks/head/build/g4cx     
    [ 13%] Building CXX object CMakeFiles/G4CX.dir/G4CXOpticks.cc.o
    [ 13%] Building CXX object CMakeFiles/G4CX.dir/G4CX_LOG.cc.o
    In file included from /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:33:
    /data/blyth/junotop/ExternalLibs/custom4/0.1.8/include/Custom4/C4Version.h:62:5: error: invalid digit "8" in octal constant
     #if Custom4_VERSION_NUMBER >= 00105
         ^~~~~~~~~~~~~~~~~~~~~~
    /data/blyth/junotop/ExternalLibs/custom4/0.1.8/include/Custom4/C4Version.h:64:7: error: invalid digit "8" in octal constant
     #elif Custom4_VERSION_NUMBER >= 00104
           ^~~~~~~~~~~~~~~~~~~~~~
    /data/blyth/junotop/ExternalLibs/custom4/0.1.8/include/Custom4/C4Version.h:71:5: error: invalid digit "8" in octal constant
     #if Custom4_VERSION_NUMBER > 00105
         ^~~~~~~~~~~~~~~~~~~~~~
    /data/blyth/junotop/ExternalLibs/custom4/0.1.8/include/Custom4/C4Version.h:73:7: error: invalid digit "8" in octal constant
     #elif Custom4_VERSION_NUMBER > 00104
           ^~~~~~~~~~~~~~~~~~~~~~
    make[2]: *** [CMakeFiles/G4CX.dir/G4CXOpticks.cc.o] Error 1



::

     32 #ifdef WITH_CUSTOM4
     33 #include "C4Version.h"
     34 #endif




