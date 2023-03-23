#!/bin/bash -l 
usage(){ cat << EOU
build.sh
==========

Need CMAKE_PREFIX_PATH setup to find Custom4 for example::


EOU
}


arg=$1

sdir=$(pwd)
name=$(basename $sdir) 

BASE=/tmp/$USER/$name
bdir=$BASE/build 
idir=$BASE/install

rm -rf $bdir
mkdir -p $bdir
cd $bdir

if [ "${arg/kludge}" != "$arg" ]; then
    export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/tmp/$USER/customgeant4/install
fi 

echo $CMAKE_PREFIX_PATH | tr ":" "\n"

cmake $sdir -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$idir



