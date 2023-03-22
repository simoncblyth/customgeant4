#!/bin/bash -l 

usage(){ cat << EOU
C4OpBoundaryProcessTest.sh
============================

Manual (without CMake) build as sanity check.

Currently segments as lacks geometry, nevertheless checking 
the build is useful.  

EOU
}

name=C4OpBoundaryProcessTest

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

clhep-
g4-

defarg="build_run"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
         ../C4OpBoundaryProcess.cc \
           -g -std=c++11 -lstdc++ \
           -I.. \
           -I$(clhep-prefix)/include \
           -I$(g4-prefix)/include/Geant4  \
           -L$(g4-prefix)/lib \
           -L$(clhep-prefix)/lib \
           -lG4global \
           -lG4geometry \
           -lG4particles \
           -lG4processes \
           -lG4materials \
           -lG4event \
           -lG4track \
           -lG4tracking \
           -lCLHEP \
           -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then

    case $(uname) in 
      Darwin) lldb__ $bin ;;
      Linux)  gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 




exit 0


