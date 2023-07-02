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



Versioning in 0.1.4 and prior
----------------------------------

Versioning was only fully setup in 0.1.5 thus to use versioning 
before that requires some manual kludging. 
For example adding a minimal C4Version.h into the appropriate include dir,
will allow macro version branching to work::

    echo "#define Custom4_VERSION_NUMBER 00104" > /usr/local/opticks_externals/custom4/0.1.4/lib/Custom4-0.1.4/C4Version.h


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
---------------------

Previous versions of this script used the dirty approach of installing 
into the OPTICKS_PREFIX dir, eg /usr/local/opticks, making it 
difficult to work with multiple versions of Custom4. 

Since version 0.1.5 the install prefix can be controlled by 
CUSTOM4_PREFIX envvar (rather than OPTICKS_PREFIX) 
with default prefix value incorporating the version::

    ${OPTICKS_PREFIX}_externals/custom4/0.1.5      
    /usr/local/opticks_externals/custom4/0.1.5

This behaviour can be mimicked with prior versions of this
build.sh such as 0.1.4 by manually overriding OPTICKS_PREFIX
with the appropriate versioned directory::

    OPTICKS_PREFIX=/usr/local/opticks_externals/custom4/0.1.4 ./build.sh 


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

    cmake $sdir -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$CUSTOM4_PREFIX
    make
    make install   
fi 




