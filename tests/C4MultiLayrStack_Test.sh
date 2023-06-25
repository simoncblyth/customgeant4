#!/bin/bash -l 

name=C4MultiLayrStack_Test

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

defarg="build_run"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc -std=c++11 -lstdc++ -I.. -o $bin  
   [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
   $bin 
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 


exit 0 


