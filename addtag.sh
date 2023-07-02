#!/bin/bash -l 

VERSION_MAJOR=$(perl -ne 'm,VERSION_MAJOR (\d*)\), && print $1' CMakeLists.txt)
VERSION_MINOR=$(perl -ne 'm,VERSION_MINOR (\d*)\), && print $1' CMakeLists.txt)
VERSION_PATCH=$(perl -ne 'm,VERSION_PATCH (\d*)\), && print $1' CMakeLists.txt)
VERSION_NUMBER=$(perl -ne 'm,VERSION_NUMBER (\d*)\), && print $1' CMakeLists.txt)

VERSION=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}

cat << EOC
## addtag.sh 
## NB before piping the below commands to the shell bump the version in CMakeLists.txt and 
## make sure to "git add" and "git push"  to avoid version inconsistency 

git tag -a v$VERSION -m "addtag.sh for VERSION $VERSION VERSION_NUMBER $VERSION_NUMBER extracted from CMakeLists.txt  "
git push --tags

## NB addtag.sh only emits to stdout : if satisfied with the commands then pipe to the shell 

EOC

