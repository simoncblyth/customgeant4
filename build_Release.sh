#!/bin/bash -l 

cd $(dirname $(realpath $BASH_SOURCE))
export CUSTOM4_CMAKE_BUILD_TYPE=Release

./build.sh 

