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

EOU
}


sdir=$(pwd)
name=$(basename $sdir) 

BASE=/tmp/$USER/$name
bdir=$BASE/build 
idir=$BASE/install

if [ -n "$OPTICKS_PREFIX" ]; then 
    idir=$OPTICKS_PREFIX
fi 

rm -rf $bdir
mkdir -p $bdir  # bdir must be created, CMake populates it with Makefile and workings 
cd $bdir 
pwd 

cmake $sdir -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$idir
make
make install   



