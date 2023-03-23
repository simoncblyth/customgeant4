#!/bin/bash -l 

VERSION_MAJOR=$(perl -ne 'm,VERSION_MAJOR (\d*)\), && print $1' CMakeLists.txt)
VERSION_MINOR=$(perl -ne 'm,VERSION_MINOR (\d*)\), && print $1' CMakeLists.txt)
VERSION_PATCH=$(perl -ne 'm,VERSION_PATCH (\d*)\), && print $1' CMakeLists.txt)

VERSION=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}

cat << EOC

git tag -a v$VERSION -m "addtag.sh for VERSION $VERSION extracted from CMakeLists.txt  "
git push --tags

## NB addtag.sh only emits to stdout : if satisfied with the commands then pipe to the shell 

EOC

