#!/bin/bash -l 

sdir=$(pwd)
name=$(basename $sdir) 

BASE=/tmp/$USER/$name
bdir=$BASE/build 
idir=$BASE/install

rm -rf $bdir
mkdir -p $bdir
cd $bdir

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/tmp/$USER/customgeant4/install

cmake $sdir -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$idir



