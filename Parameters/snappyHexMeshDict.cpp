castellatedMesh true;
snap true;
// Layer generation is hard-disabled in system/snappyHexMeshDict.

maxLocalCells 100000;
maxGlobalCells 3000000;
minRefinementCells 0;
maxLoadUnbalance 0.10;
nCellsBetweenLevels 3;

rotaryRegionSurfaceRefinementLevel (3 3);

propellerSurfaceRefinementLevel (4 4); // Uniform propeller surface refinement.

propellerRefinementRegionMode distance;
propellerRefinementRegionLevel ((0.0045 4) (0.0125 4));

rotaryRegionRefinementRegionMode inside;
rotaryRegionRefinementRegionLevel ((1E15 3));

innerCylinderRefinementRegionMode inside;
innerCylinderRefinementRegionLevel ((1E15 3));

outerCylinderRefinementRegionMode inside;
outerCylinderRefinementRegionLevel ((1E15 2));

acousticSurfaceRefinementRegionMode inside;
acousticSurfaceRefinementRegionLevel ((1E15 2));

nSmoothPatch 3;
tolerance 1.0;
nSolveIter 30;
nRelaxIter 20;
nFeatureSnapIter 10;
implicitFeatureSnap false;
explicitFeatureSnap true;
multiRegionFeatureSnap false;

maxNonOrtho 65;
maxBoundarySkewness 4;
maxInternalSkewness 4;
maxConcave 80;
minVol 1e-13;
mergeTolerance 1e-6;
nSmoothScale 4;

#include "snappyGeometry.generated"
