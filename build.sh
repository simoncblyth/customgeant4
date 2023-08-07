#!/bin/bash -l 
usage(){ cat << EOU
build.sh
===========

NB this works without any Opticks code or CMake machinery. 
It does however require CMAKE_PREFIX_PATH envvar to find Geant4,CLHEP eg::

    epsilon:customgeant4 blyth$ echo $CMAKE_PREFIX_PATH | tr ":" "\n"
    /usr/local/opticks_externals/g4_1042
    /usr/local/opticks_externals/clhep
    /usr/local/opticks_externals/xercesc
    /usr/local/opticks_externals/boost
    /usr/local/opticks
    /usr/local/opticks/externals
    /usr/local/optix
    epsilon:customgeant4 blyth$ 



Temporarily return to a tagged version
-----------------------------------------

::

    > git tag    # list tags 
    ...
    v0.1.2
    v0.1.3
    v0.1.4
        
    git checkout tags/v0.1.4 -b main



CMAKE_INSTALL_PREFIX
----------------------
 
Before release 0.1.5 this build script used the dirty approach of installing 
into the OPTICKS_PREFIX dir, eg /usr/local/opticks, making it 
difficult to work with multiple versions of Custom4. 

Since version 0.1.5 the install prefix can be controlled by 
CUSTOM4_PREFIX envvar (rather than OPTICKS_PREFIX) 
with default prefix value incorporating the version::

    ${OPTICKS_PREFIX}_externals/custom4/0.1.5      
    /usr/local/opticks_externals/custom4/0.1.5

This behaviour can be mimicked with prior versions of this
build.sh such as 0.1.4 by manually editing build.sh to 
use CUSTOM4_PREFIX rather than OPTICKS_PREFIX and 
setting that as shown below to the  appropriate versioned directory::

    CUSTOM4_PREFIX=/usr/local/opticks_externals/custom4/0.1.4 ./build.sh 



Opticks Clean build to pickup new Custom4
-------------------------------------------

Alternatively do a fully clean build of everything::
  
    o
    om-
    om-clean
    om-conf
    oo


JUNOSW clean build to pickup new Custom4
-------------------------------------------

And clean build junosw::

   jo
   cd build
   make clean
   cd ..
   ./build_Debug.sh 


HUH: Sometimes need nuclear option, to get rid of old library refs in CMake files::

   jo
   rm -rf build
   ./build_Debug.sh 


Location of reference installs on laptop
-------------------------------------------

::

    /usr/local/opticks_externals/custom4/0.1.6/include/Custom4

Workstation install dirs
-------------------------

::

    /data/blyth/junotop/ExternalLibs/custom4/0.1.6



EOU
}


REALDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
sdir=$(pwd)
name=$(basename $sdir) 

BASE=/tmp/$USER/$name
bdir=$BASE/build 

VERSION_MAJOR=$(perl -ne 'm,VERSION_MAJOR (\d*)\), && print $1' $REALDIR/CMakeLists.txt)
VERSION_MINOR=$(perl -ne 'm,VERSION_MINOR (\d*)\), && print $1' $REALDIR/CMakeLists.txt)
VERSION_PATCH=$(perl -ne 'm,VERSION_PATCH (\d*)\), && print $1' $REALDIR/CMakeLists.txt)
VERSION=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}


custom4_prefix=${OPTICKS_PREFIX}_externals/custom4/$VERSION
CUSTOM4_PREFIX=${CUSTOM4_PREFIX:-$custom4_prefix}

custom4_cmake_build_type=Release
#custom4_cmake_build_type=Debug
CUSTOM4_CMAKE_BUILD_TYPE=${CUSTOM4_CMAKE_BUILD_TYPE:-${custom4_cmake_build_type}}


defarg="info_install"
arg=${1:-$defarg}

vars="sdir name BASE bdir idir arg OPTICKS_PREFIX custom4_prefix CUSTOM4_PREFIX"
vars="$vars VERSION VERSION_MAJOR VERSION_MINOR VERSION_PATCH"


if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 


if [ "${arg/install}" != "$arg" ]; then 

    rm -rf $bdir
    mkdir -p $bdir  # bdir must be created, CMake populates it with Makefile and workings 
    cd $bdir 
    pwd 

    cmake $sdir -DCMAKE_BUILD_TYPE=$CUSTOM4_CMAKE_BUILD_TYPE -DCMAKE_INSTALL_PREFIX=$CUSTOM4_PREFIX
    make
    make install   
fi 




