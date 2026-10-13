castellatedMesh true;
snap true;
addLayers false; // Set false to mesh without native snappy inflation layers.

maxLocalCells 100000;
maxGlobalCells 3000000;
minRefinementCells 0;
maxLoadUnbalance 0.10;
nCellsBetweenLevels 3;

rotaryRegionSurfaceRefinementLevel (3 3);

propellerSurfaceRefinementLevel (6 6); // Uniform propeller surface refinement.

propellerRefinementRegionMode distance;
propellerRefinementRegionLevel ((0.0045 6) (0.0125 5));

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

// Native snappy layer controls: applied to the propeller wall only.
// relativeSizes true: thicknesses are fractions of the local outer cell size.
// For metre-based thicknesses use relativeSizes false and adjust the values.
// Specify exactly two of expansionRatio, firstLayerThickness,
// finalLayerThickness and thickness (total stack thickness).
addLayersControls
{
    relativeSizes true;
    layers
    {
        propeller
        {
            nSurfaceLayers 1;
        }
    }
    expansionRatio 1.2;
    finalLayerThickness 2e-3;
    minThickness 2e-3;
    nGrow 0;
    featureAngle 60;
    slipFeatureAngle 30;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}

#include "snappyGeometry.generated"
