#!/bin/bash -l
usage(){ cat << EOU
build_into_junosw.sh
=====================

Usage
------

./build_into_junosw.sh info
   show usage and variable values

./build_into_junosw.sh put
   run rsync_put.sh syncing repo to workstation 

./build_into_junosw.sh build
   run build.sh installing Custom4 into junotop/ExternalLibs/custom4/\$VERSION 


Motivation
------------

Development via tagged releases and junoenv machinery on workstation 
is too painful, so take a dirty approach to install a Debug build
into junotop/ExternalLibs/custom4/\$VERSION where the VERSION will
typically be the untagged next version parsed from the CMakeLists.txt

Initial setup to use untagged "next" Custom4 VERSION on workstation
----------------------------------------------------------------------
 
1. laptop: "c4 ; ./rsync_put.sh" sync Custom4 repo from laptop to workstation
2. workstation: "c4 ; ./build_into_junosw.sh" build and install Custom4 into /data/blyth/junotop/ExternalLibs/custom4/VERSION
3. workstation: manually prep the VERSION scripts based on prior VERSION, eg::

    N[blyth@localhost custom4]$ pwd
    /data/blyth/junotop/ExternalLibs/custom4
    N[blyth@localhost custom4]$ cp 0.1.6/bashrc 0.1.7/
    N[blyth@localhost custom4]$ cp 0.1.6/tcshrc 0.1.7/
    N[blyth@localhost custom4]$ perl -pi -e 's/0.1.6/0.1.7/g' 0.1.7/bashrc
    N[blyth@localhost custom4]$ perl -pi -e 's/0.1.6/0.1.7/g' 0.1.7/tcshrc

4. workstation: "jt ; vi bashrc.sh" set the version of Custom4 in jre (juno-runtime-environment) to the next VERSION just installed 
5. exit session and reconnect 
6. clean build Opticks. Atually only need to rebuild from qudarap onwards, 
   but simpler to clean build everything::

    o
    om-
    om-clean
    om-conf
    oo

Updating Custom4 workflow
--------------------------

1. laptop: "c4 ; ./rsync_put.sh" sync Custom4 repo from laptop to workstation
2. workstation: "c4 ; ./build_into_junosw.sh" build and install Custom4 into /data/blyth/junotop/ExternalLibs/custom4/VERSION
3. workstation: update build opticks, "o ; oo" 

Note that because the Custom4 install directory stays the same
there is no need for clean building. This makes it fast to update Custom4. 

EOU
}

REALDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

VERSION_MAJOR=$(perl -ne 'm,VERSION_MAJOR (\d*)\), && print $1' $REALDIR/CMakeLists.txt)
VERSION_MINOR=$(perl -ne 'm,VERSION_MINOR (\d*)\), && print $1' $REALDIR/CMakeLists.txt)
VERSION_PATCH=$(perl -ne 'm,VERSION_PATCH (\d*)\), && print $1' $REALDIR/CMakeLists.txt)
VERSION=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}

custom4_prefix=/data/blyth/junotop/ExternalLibs/custom4/$VERSION
export CUSTOM4_PREFIX=${CUSTOM4_PREFIX:-$custom4_prefix}

vars="BASH_SOURCE REALDIR VERSION CUSTOM4_PREFIX"

case $(uname) in
   Darwin) defarg="info" ;;
   Linux)  defarg="info_build" ;; 
esac
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then
    usage 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi 

if [ "${arg/put}" != "$arg" ]; then
   $REALDIR/rsync_put.sh 
   [ $? -ne 0 ] && echo $BASH_SOURCE : put error && exit 1 
fi 

if [ "${arg/build}" != "$arg" ]; then
    #export CUSTOM4_CMAKE_BUILD_TYPE=Release
    export CUSTOM4_CMAKE_BUILD_TYPE=Debug
    $REALDIR/build.sh 
   [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 2
fi

exit 0


