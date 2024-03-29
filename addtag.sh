#!/bin/bash -l 
usage(){ cat << EOU
addtag.sh 
===========

Workflow:

1. following C4 developments worthy of tagging, 
   get changes committed and pushed. Ensure that the 
   CMakeLists.txt VERSION is bumped compared to the prior tag, 
   ie it should be incremented from the last listed by "git tag"::

        epsilon:customgeant4 blyth$ git tag 
        ...
        v0.1.4
        v0.1.5

   Add an entry to the RELEASE_NOTES.rst for this version

2. run addtag.sh checking VERSION and VERSION_NUMBER are consistent

3. run addtag.sh again and pipe the commands to the shell 
   which will add the tag and propagate that up to github 

4. make a reference install of the just tagged version on laptop, before 
   making any further changes (will probably need to delete the installed
   pre-tagged version first)::

      ./build.sh info  # check VERSION and dirs 
      ./build.sh       # proceed 

5. after the reference install bump the CMakeLists.txt VERSION 
   such that subsequent installs go into the future versioned directory, 
   previewing the next version
 
6. on laptop edit ~/.opticks_config to pick the Custom4 version used by Opticks, eg::

       opticks-prepend-prefix /usr/local/opticks_externals/custom4/0.1.7 ## next tag    

   which will usually be the next version. 
   But sometimes it will be the prior tagged install or even an earlier install. 

7. on laptop edit je:packages/custom4.sh setting the default version to the 
   just tagged version then "put.sh | sh" it to workstation and install 
   the newly tagged version with::

        je ; bash junoenv libs all custom4 
        ## this downloads and installs the tagged tarball from github 

When the Great Firewall prevents download the typical thing that happens is get a 
hang for several minutes and a zero sized tarball. 
So delete that and from remote scp in the tarball.::

    curl -L -O https://github.com/simoncblyth/customgeant4/archive/refs/tags/v0.1.5.tar.gz
    scp v0.1.5.tar.gz P:/data/blyth/junotop/ExternalLibs/Build/

Can now use util.sh to do that after changing the tag in the script::

    ./util.sh info
    ./util.sh get
    ./util.sh scp

Then can rerun, which skips download as tarball already present::

    bash junoenv libs all custom4 

To pickup that version then::

    jt ; vi bashrc.sh   ## edit the custom4 source line version 

To check Opticks is picking up the updated Custom4 use::

    ~/opticks/examples/UseCustom4/go.sh 

EOU
}

VERSION_MAJOR=$(perl -ne 'm,VERSION_MAJOR (\d*)\), && print $1' CMakeLists.txt)
VERSION_MINOR=$(perl -ne 'm,VERSION_MINOR (\d*)\), && print $1' CMakeLists.txt)
VERSION_PATCH=$(perl -ne 'm,VERSION_PATCH (\d*)\), && print $1' CMakeLists.txt)
VERSION_NUMBER=$(perl -ne 'm,VERSION_NUMBER (\d*)\), && print $1' CMakeLists.txt)

VERSION=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}



cat << EOC
## addtag.sh 
##
## NB before piping the below commands to the shell make the following checks:
##
## 1. "git status" : check all changes are committed and pushed
##
EOC

echo "## git status --porcelain | sed -e 's/^/## /' "
git status --porcelain | sed -e 's/^/## /'

cat << EOC
##
## 2. "git tag" : ensure that the tag string extracted from the CMakeLists.txt is incremented
##    from the last one listed by "git tag"
## 
EOC

echo "## git tag | tail -10 | sed -e 's/^/## /' "
git tag | tail -10 | sed -e 's/^/## /' 

cat << EOC
##

git tag -a v$VERSION -m "addtag.sh for VERSION $VERSION VERSION_NUMBER $VERSION_NUMBER extracted from CMakeLists.txt  "
git push --tags

## NB addtag.sh only emits to stdout : if satisfied with the commands then pipe to the shell 
##

EOC

