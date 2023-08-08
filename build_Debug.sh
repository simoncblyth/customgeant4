#!/bin/bash -l 

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
export CUSTOM4_CMAKE_BUILD_TYPE=Debug

$SDIR/build.sh 

