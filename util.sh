#!/bin/bash -l 
usage(){ cat << EOU
util.sh
========

info
   dump variables
get
   download release tarball
scp 
   scp release tarball to destination

EOU
}

defarg="info"
arg=${1:-$defarg}

tag="v0.1.6"
TAG=${TAG:-$tag}

export FOLD=/tmp/Custom4
mkdir -p $FOLD
tgz=$TAG.tar.gz
url=https://github.com/simoncblyth/customgeant4/archive/refs/tags/$tgz

dst=P:/data/blyth/junotop/ExternalLibs/Build/
DST=${DST:-$dst}

vars="BASH_SOURCE FOLD arg TAG tgz url DST"

cd $FOLD

if [ "${arg/info}" != "$arg" ]; then 
    usage
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/get}" != "$arg" ]; then 
    if [ ! -f "$tgz" ]; then
        curl -L -O $url 
    else
        echo $BASH_SOURCE : tgz already downloaded 
    fi 
    ls -l $tgz
    tar ztvf $tgz
fi 

if [ "${arg/scp}" != "$arg" ]; then 
    if [ -f "$tgz" ]; then 
        scp $tgz $DST
    else
        echo $BASH_SOURCE : ERROR no tgz $tgz 
    fi
fi


