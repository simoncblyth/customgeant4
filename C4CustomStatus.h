#pragma once
/**
C4CustomStatus.h
==================

The custom status char is set by C4OpBoundaryProcess::PostStepDoIt

+------+-------------------------------------------------------------------------------+
| char |                                                                               |
+======+===============================================================================+
|  U   |  starting value set at initialization and at every step                       |
+------+-------------------------------------------------------------------------------+
|  Z   |  @/# OpticalSurface BUT local_z < 0 : so ordinary surface                     |         
+------+-------------------------------------------------------------------------------+
|  Y   |  @ OpticalSurface AND local_z > 0 : so C4CustomART::doIt runs                 |
+------+-------------------------------------------------------------------------------+
|  -   |  # OpticalSurface AND local_z > 0 : so traditional detect at photocathode     |                
+------+-------------------------------------------------------------------------------+
|  X   |  NOT @/# OpticalSurface : so ordinary surface                                 | 
+------+-------------------------------------------------------------------------------+
|  \0  |  Uninitialized array content                                                  |
+------+-------------------------------------------------------------------------------+

::

   jxn
   ./ntds.sh ana

    In [13]: a_bop = a.f.aux.view(np.int32)[:,:,3,3]
    In [14]: b_bop = b.f.aux.view(np.int32)[:,:,3,3]

    In [15]: np.c_[np.unique( a_bop, return_counts=True )]
    Out[15]: 
    array([[    0, 26037],       ## uninit
           [   85,  5003],       ## U 
           [   88,   960]])      ## X

    In [16]: np.c_[np.unique( b_bop, return_counts=True )]
    Out[16]: 
    array([[    0, 25784],      ## uninit
           [   85,  4746],      ## U
           [   88,   270],      ## X
           [   89,  1073],      ## Y
           [   90,   127]])     ## Z

    In [18]: list(map(chr, [0, 85, 88, 89, 90]))
    Out[18]: [        '\x00', 'U','X','Y','Z']


**/
