"""Case matrix and overrides for the AMI dimension study."""

from itertools import product

from tools import update_parameter


STUDY_KEY = "ami_dimension_study"
BASE_SPHERE_DIAMETER = 2.5
REFINEMENT_SPHERE_DIAMETER = 2.1
BACKGROUND_CELLS = (18, 52, 18)


def validate_options(args):
    if args.mode != "AMI":
        raise ValueError("--ami_dimension_study requires --mode AMI")
    if args.study or any((args.study_file, args.study_parameter, args.study_values)):
        raise ValueError("--ami_dimension_study cannot be combined with --study options")
    if len(args.meshes) != 1 or len(args.rpms) != 1:
        raise ValueError("--ami_dimension_study requires exactly one STL and one RPM")
    if args.field_init != "off":
        raise ValueError("--ami_dimension_study requires --field-init off")
    if args.acoustic_surface == "permeable":
        if (args.acoustic_sphere_diameters is not None
                or args.acoustic_sphere_diameter not in (None, REFINEMENT_SPHERE_DIAMETER)):
            raise ValueError("--ami_dimension_study uses one 2.1D permeable sampling sphere")
        args.acoustic_sphere_diameter = REFINEMENT_SPHERE_DIAMETER


def case_variants():
    for radial_step, axial_step in product(range(4), repeat=2):
        yield f"_ami_dim_r{radial_step + 1}_a{axial_step + 1}", {
            "diameter_factor": 1.1 + (1.6 - 1.1) * radial_step / 3,
            "axial_fraction": axial_step / 3,
            "base_sphere_diameter_factor": BASE_SPHERE_DIAMETER,
            "refinement_sphere_diameter_factor": REFINEMENT_SPHERE_DIAMETER,
            "background_cells": list(BACKGROUND_CELLS),
            "inner_cylinder_level": 2,
        }


def apply_mesh_parameters(case, settings):
    parameters = case / "Parameters"
    cells = " ".join(map(str, settings["background_cells"]))
    overrides = (
        ("blockMeshDict.cpp", "blocks_resolution", f"({cells})"),
        ("snappyHexMeshDict.cpp", "innerCylinderRefinementRegionLevel",
         f"((1E15 {settings['inner_cylinder_level']}))"),
    )
    for filename, name, value in overrides:
        path = parameters / filename
        if not update_parameter(path, name, value, quiet=True):
            raise ValueError(f"Cannot set {name} in {path}")
