angular_generalization
========================

* move decision about backwards qe zeroing into accessor ? 


new API : get_pmtid_qe_angular
-------------------------------

::

   "get_pmtid_qe_angular(pmtid,energy,lposcost,minus_cos_theta_aoi)":

* lposcost (cosine of polar angle of local PMT boundary intersect position, "local_z/radius")

  * 0->1  : z>0 forward hemisphere of PMT
  * -1->0 : z<0 backward hemisphere of PMT : this would indicate a bug as PMT not sensitive at -ve z   

* minus_cos_theta (mct: dot product of PMT intersection point outwards normal and photon momentum direction) 

  * minus_cos_theta > 0 : outwards going photons, with the normal
  * minus_cos_theta = 0 : grazing incidence, perpendicular to normal  
  * minus_cos_theta < 0 : inwards going photons, against the normal



where "get_pmtid_qe_angular" used ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`


Needed from C4CustomART::doIt 


how to handle the implementation version change ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Q:Can the different impls be encapsulated within the accessor ? 
A:Use with more args, but want version 0 to be current impl. 



how to get local_position_theta in c4 ?
------------------------------------------

::

    176 inline double C4CustomART::local_z( const G4Track& aTrack )
    177 {
    178     const G4AffineTransform& transform = aTrack.GetTouchable()->GetHistory()->GetTopTransform();
    179     G4ThreeVector localPoint = transform.TransformPoint(theGlobalPoint);
    180     zlocal = localPoint.z() ;
    181 #ifdef C4_DEBUG
    182     lposcost = localPoint.cosTheta() ;
    183 #endif
    184     return zlocal  ;
    185 }


local_position_theta (aka lposcost) in Opticks
-------------------------------------------------


CSGOptiX/CSGOptiX7.cu::

    579         const float3 P2 = mesh.vertex[ tri.z ];
    580         const float3 P = ( 1.0f-barys.x-barys.y)*P0 + barys.x*P1 + barys.y*P2;
    581 
    588         const float3 Ng = cross( P1-P0, P2-P0 );
    589         // local normal from cross product of vectors between vertices : HMM is winding order correct : TODO: check sense of normal
    590 
    591         const float3 N = normalize( optixTransformNormalFromObjectToWorldSpace( Ng ) );
    592 
    593         float t = optixGetRayTmax() ;
    594 
    601         float lposcost = normalize_z(P); // scuda.h 


    666 extern "C" __global__ void __intersection__is()
    667 {
    668 
    669 #if defined(DEBUG_PIDX)
    670     //printf("//__intersection__is\n"); 
    671 #endif
    672 
    673     HitGroupData* hg  = (HitGroupData*)optixGetSbtDataPointer();
    674     int nodeOffset = hg->prim.nodeOffset ;
    675 
    676     const CSGNode* node = params.node + nodeOffset ;  // root of tree
    677     const float4* plan = params.plan ;
    678     const qat4*   itra = params.itra ;
    679 
    680     const float  t_min = optixGetRayTmin() ;
    681     const float3 ray_origin = optixGetObjectRayOrigin();
    682     const float3 ray_direction = optixGetObjectRayDirection();
    683 
    684     float4 isect ; // .xyz normal .w distance 
    685     if(intersect_prim(isect, node, plan, itra, t_min , ray_origin, ray_direction ))
    686     {
    687         const float lposcost = normalize_z(ray_origin + isect.w*ray_direction ) ;  // scuda.h 
    688         const unsigned hitKind = 0u ;     // only up to 127:0x7f : could use to customize how attributes interpreted
    689         const unsigned boundary = node->boundary() ;  // all CSGNode in the tree for one CSGPrim tree have same boundary 
    690 
    691 #ifdef WITH_PRD
    692         if(optixReportIntersection( isect.w, hitKind))
    693         {
    694             quad2* prd = SOPTIX_getPRD<quad2>(); // access prd addr from RG program  
    695             prd->q0.f = isect ;  // .w:distance and .xyz:normal which starts as the local frame one 
    696             prd->set_boundary(boundary) ;
    697             prd->set_lposcost(lposcost);
    698         }


 
scuda.h::
           
     562 SUTIL_INLINE SUTIL_HOSTDEVICE float normalize_z(const float3& v)  // CLHEP ThreeVector calls this cosTheta 
     563 {
     564   return v.z / sqrtf(dot(v, v));
     565 }


