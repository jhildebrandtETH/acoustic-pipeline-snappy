# Blade trailing-edge blunt-cut tool

Independent STEP preprocessing utility. It does not launch Fluent or OpenFOAM,
change solver settings, or run as part of normal pipeline preprocessing.

## Run

From the repository root, using Python 3.12:

```powershell
conda activate of_pipeline_env
python -m pip install -r ansys_pipeline/geometry_tools/blunt_trailing_edge/requirements.txt
python ansys_pipeline/geometry_tools/blunt_trailing_edge/trim_step.py ansys_pipeline/geometry_tools/blunt_trailing_edge/settings.fused.json
```

Copy the JSON for another input. All file paths inside it are resolved relative
to the JSON file, regardless of the current working directory. Outputs default
to a separate directory; existing results are refused unless `overwrite` is
explicitly true. The input file is never overwritten. STEP import dimensions
are millimetres in Open CASCADE; STL diagnostic exports are also millimetres.

## What is changed

`setback_mm: 0.20` means **0.20 mm along the local section chord from the trailing
edge toward the leading edge**, not 0.20 mm final trailing-edge thickness.
These two quantities differ; the achieved closure thickness is reported.

At each span station, the two section extrema along the frame's chord-coordinate
axis define the chord direction. For the supplied APC orientation, the positive
blade's trailing edge is the minimum chord coordinate; the opposite blade's
sense is reversed. The chord-coordinate axis is rotation_axis cross
positive_blade_axis. This definition is explicit rather than an aerodynamic
edge-recognition heuristic: check the overlays before using any different model.

A piecewise ruled cutting solid follows those local cut planes. Its section
closure is straight, without a fillet. Only the **cutter** is lofted. The original
CAD is Boolean-subtracted: retained blade surfaces are not re-lofted, smoothed,
scaled or rotated. A coaxial cylindrical protected region prevents hub removal.
The cut ends where it meets that region, so a small root transition boundary is
expected. The protection covers the cylinder interior, not an arbitrary artistic
blade/hub face classification.

**Aligned CAD seams (schema 2):** `station_strategy: "cad_seams"` places the
cutter's cross-edges at existing constant-span CAD profile seams. This avoids
the stagger between an independent cutter grid and the blade's original face
divisions. No unrelated uniform or extra tip stations are inserted in this
mode. Existing CAD seam positions are measured, not fitted or moved. If those
stations are too sparse to meet the setback tolerance, the tool refuses the
result rather than silently adding unmatched edges.

Schema-1 JSON files default to the earlier `uniform` strategy for reproducibility.
That strategy retains the old uniform grid and extra tip stations. The new
example uses schema 2 and CAD seam alignment; the previous example is retained
as `settings.legacy-uniform.json`.

`hub.radius_mm: "auto"` selects the largest cylindrical CAD surface coaxial with
the configured axis and centre. The supplied source resolves to 10.16 mm. For
another geometry, check this value or provide the explicit hub radius. If a
rounded hub extends beyond that cylinder, specify a radius enclosing it. The
tool refuses invalid input solids, missing coaxial hub information, impossible
setbacks, invalid output, disconnected fragments or failed validation; it never
automatically heals or rounds input geometry.

Tracking sections uses a CAD tessellation with `section_deflection_mm` absolute
deflection (default 0.001 mm). This avoids incomplete exact plane/NURBS section
intersections observed in the source STEP. It defines the cutter only; the output
remains trimmed original CAD surfaces. `station_spacing_mm` controls the ruled
cutter resolution **only in uniform mode**; `validation_tolerance_mm` is a separate acceptance bound.
The tool rejects a result when independently checked sections miss that bound.
Changing the section resolution should be treated as changing tool settings.

## Outputs and safeguards

* `*_blunt0.200mm.step`: trimmed STEP geometry.
* `settings.resolved.json`: complete resolved settings.
* `validation.json`: tool/input/output hashes, CAD version, validity, solid
  counts, removed material, added-material and protected-hub checks, measured
  setback and closure thickness on both blades. In CAD seam mode it also lists
  source seam locations and verifies that short constant-span cross-edges are
  within 0.0001 mm of a native CAD station (including root/tip boundaries).
* `cutting-frames.json`: every cutter construction plane.
* `blade_+1_overlays.png` and `blade_-1_overlays.png`: section overlays in original
  coordinates, without fitted registration.
* `section-overlays.json`: raw before/after section polylines in millimetres.
* `original_analysis_mm.stl` / `trimmed_analysis_mm.stl`: controlled diagnostic
  tessellations; not substitutes for the STEP when meshing.

The pinned CAD version, input hash, resolved JSON and tool hash define a run.
The tool uses serial Boolean operations and records all construction sections.
STEP writer timestamps can differ between runs; geometric reproducibility is
checked through the invariants and measurements, not an assumption of identical
STEP header bytes. No named selections/SpaceClaim face labels are preserved by
this geometry-only export. Reassign them after import as required.

CAD validity and geometry fidelity do not demonstrate Fluent meshability or
OpenFOAM checkMesh quality. A 0.20 mm initial setback is deliberately modest;
increase it only after inspecting actual blunt thickness and meshing results.

Tests: `python -m unittest discover -s ansys_pipeline/geometry_tools/blunt_trailing_edge -p "test_*.py"`.
