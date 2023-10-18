#!/bin/bash -l 

name=C4MultiLayrStack_Test2

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

defarg="build_run_ana"
arg=${1:-$defarg}
#opt=-DMOCK_CURAND_DEBUG
opt=""

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc -std=c++11 -lstdc++ -I$HOME/np $opt  -I.. -o $bin  
   [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
   $bin 
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $name.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3 
fi 

exit 0 


