#!/bin/bash


generate_bashrc()
{

   local custom4_prefix=/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/custom4/0.1.8
   local custom4_prefix_relative=\${JUNOTOP}/ExternalLibs/custom4/0.1.8

   local CUSTOM4_PREFIX=${CUSTOM4_PREFIX:-$custom4_prefix}
   #local CUSTOM4_PREFIX_RELATIVE=${CUSTOM4_PREFIX_RELATIVE:-$custom4_prefix_relative}
   local CUSTOM4_PREFIX_RELATIVE=${CUSTOM4_PREFIX}

   cat << EOH
## generated $(date) by $(realpath $BASH_SOURCE) $FUNCNAME

if [ -z "\${JUNOTOP}" ]; then
export JUNO_EXTLIB_custom4_HOME=$CUSTOM4_PREFIX
else
export JUNO_EXTLIB_custom4_HOME=$CUSTOM4_PREFIX_RELATIVE
fi
EOH

cat << \EOS
export PATH=${JUNO_EXTLIB_custom4_HOME}/bin:${PATH}
if [ -d ${JUNO_EXTLIB_custom4_HOME}/lib ];
then
export LD_LIBRARY_PATH=${JUNO_EXTLIB_custom4_HOME}/lib:${LD_LIBRARY_PATH}
fi
if [ -d ${JUNO_EXTLIB_custom4_HOME}/lib/pkgconfig ];
then
export PKG_CONFIG_PATH=${JUNO_EXTLIB_custom4_HOME}/lib/pkgconfig:${PKG_CONFIG_PATH}
fi
if [ -d ${JUNO_EXTLIB_custom4_HOME}/lib64 ];
then
export LD_LIBRARY_PATH=${JUNO_EXTLIB_custom4_HOME}/lib64:${LD_LIBRARY_PATH}
fi
if [ -d ${JUNO_EXTLIB_custom4_HOME}/lib64/pkgconfig ];
then
export PKG_CONFIG_PATH=${JUNO_EXTLIB_custom4_HOME}/lib64/pkgconfig:${PKG_CONFIG_PATH}
fi
export CPATH=${JUNO_EXTLIB_custom4_HOME}/include:${CPATH}
export MANPATH=${JUNO_EXTLIB_custom4_HOME}/share/man:${MANPATH}

# For CMake search path
export CMAKE_PREFIX_PATH=${JUNO_EXTLIB_custom4_HOME}:${CMAKE_PREFIX_PATH}

EOS

}

