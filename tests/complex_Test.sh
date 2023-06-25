#!/bin/bash -l 
usage(){ cat << EOU
complex_Test.sh
=============

SysRap lib only needed for WITH_THRUST GPU running 
by the NP.hh header is needed in both cases

EOU
}

name="complex_Test"

defarg=info_build_run_ana
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


opt="-std=c++11 -I. -I$CUDA_PREFIX/include -I$OPTICKS_PREFIX/include/SysRap"
c4opt="-I$HOME/customgeant4"
linkflags="-lstdc++"

WITH_THRUST=1     # comment for CPU only test

if [ -n "$WITH_THRUST" ]; then 
    opt="$opt -DWITH_THRUST"
    linkflags="$linkflags -L$OPTICKS_PREFIX/lib -lSysRap"
    echo $BASH_SOURCE : WITH_THRUST config  
else
    echo $BASH_SOURCE : not WITH_THRUST config 
fi

fold=/tmp/$USER/opticks/$name
export FOLD=$fold
bin=$FOLD/$name

vars="BASH_SOURCE name FOLD bin opt c4opt linkflags" 

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/clean}" != "$arg" ]; then
    rm -rf $FOLD
fi 

if [ "${arg/build}" != "$arg" ]; then

    mkdir -p $FOLD
    if [ "${opt/WITH_THRUST}" != "$opt" ]; then

        cmds=( "gcc  -c $name.cc $opt $c4opt -o $FOLD/${name}_cc.o"
               "nvcc -c $name.cu $opt        -o $FOLD/${name}_cu.o"
               "nvcc -o $bin $linkflags $FOLD/${name}_cc.o $FOLD/${name}_cu.o " 
               "rm $FOLD/${name}_cc.o $FOLD/${name}_cu.o "
            )
    else
        cmds=( "gcc   $name.cc $opt $c4opt $linkflags -o  $bin" )
    fi 

    for cmd in "${cmds[@]}"; do
        echo "$cmd"
        eval "$cmd" 
        [ $? -ne 0 ] && echo $BASH_SOURCE :  error with : $cmd  && exit 1
    done
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 

exit 0 

