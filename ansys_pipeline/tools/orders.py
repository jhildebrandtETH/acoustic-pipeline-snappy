"""Validate options and create or resume a simulation order."""
from tools import SimulationOrderStore
from tools import create_simulation_order
from tools import ensure_scheduler_metadata
from tools import find_source_stls
from tools import load_simulation_order
from tools import parse_end_on
from tools import reactivate_failed_cases_for_resume
from tools.snappy_pipeline import safe_path
from tools.ansys_mesh import preflight, BACKEND, COMPATIBLE_BACKENDS, find_meshes, input_fingerprints
from tools.order_files import seed_current_stl, require_new_order
from tools import save_simulation_order
from tools import validate_acoustic_arguments


def prepare_simulation_order(args, parser):
    try:
        args.end_on, args.end_on_value = parse_end_on(args.end_on)
    except ValueError as error:
        parser.error(str(error))
    try:
        simulations_directory = safe_path(args.sim_dir)
    except ValueError as error:
        parser.error(str(error))
    if not args.resume:
        if getattr(args, 'slurm', False):
            from tools.slurm import validate_config
            validate_config(args)
        require_new_order(simulations_directory)
    source_meshes_directory = simulations_directory / "MESH"

    try:
        source_meshes = find_meshes(source_meshes_directory)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    # ----------------------------------------------------------------------
    # RESUME EXISTING ORDER
    # ----------------------------------------------------------------------
    if args.resume:
        if not simulations_directory.exists():
            parser.error(f"--sim-dir does not exist: {simulations_directory}")

        order_file = simulations_directory / "simulation_order.json"
        if not order_file.exists():
            parser.error(
                "--resume was used, but no simulation_order.json was found in "
                f"{simulations_directory}"
            )

        raw_order = load_simulation_order(simulations_directory)
        if bool(raw_order.get('slurm')) != bool(getattr(args, 'slurm', False)):
            parser.error('Resume must use the original execution mode (include --slurm for Slurm orders).')
        if raw_order.get("meshing_backend") not in COMPATIBLE_BACKENDS:
            parser.error(
                "This order was not created by the Ansys import pipeline; use a fresh order directory."
            )
        for stored_case in raw_order.get("cases", []):
            try:
                resolved_case = safe_path(simulations_directory / stored_case["folder"])
                if resolved_case.parent != simulations_directory:
                    raise ValueError(
                        "Stored case folder must be directly inside this order."
                    )
            except (ValueError, KeyError) as error:
                parser.error(str(error))
        legacy_order = "total_cores" not in raw_order

        if legacy_order and args.total_cores is None:
            parser.error(
                "Legacy simulation_order.json detected: its stored 'cores' "
                "value meant cores per case. Resume this order once with "
                "--total-cores <available cores> so the new scheduler can "
                "migrate it without guessing."
            )

        if (
            not legacy_order
            and args.total_cores is not None
            and int(args.total_cores) != int(raw_order["total_cores"])
        ):
            parser.error(
                "This simulation order already has a stored total-core budget "
                f"of {raw_order['total_cores']}. Changing the allocation during "
                "--resume is intentionally disabled because cases may already "
                "be decomposed with the stored core count."
            )

        if args.cores_per_case is not None and (
            int(args.cores_per_case) != int(raw_order.get("cores_per_case", 0))
            or any(int(case.get("allocated_cores", 0)) != args.cores_per_case
                   for case in raw_order.get("cases", []))
        ):
            parser.error("Changing --cores-per-case during --resume is disabled because cases may already be decomposed. Use the stored allocation or create a new order.")

        order = ensure_scheduler_metadata(
            raw_order,
            total_cores_override=args.total_cores if legacy_order else None,
        )

        args.mode = order["mode"]
        args.meshes = order["meshes"]
        args.rpms = order["rpms"]
        args.field_init = order["field_init"]
        args.study = order["study"]
        args.study_file = order["study_file"]
        args.study_parameter = order["study_parameter"]
        args.study_values = order["study_values"]
        args.total_cores = int(order["total_cores"])
        args.cores_per_case = int(order["cores_per_case"])
        args.mesh_only = order["mesh_only"]
        args.allow_bad_mesh = order["allow_bad_mesh"]
        args.turbulence = order["turbulence"]
        args.wall_functions = order.get("wall_functions", "legacy")
        args.aerodynamics_only = order.get("aerodynamics_only", False)
        args.end_on = order["end_on"]
        args.end_on_value = order.get("end_on_value")
        args.acoustic_surface = order.get("acoustic_surface")
        args.acoustic_sphere_diameter = order.get("acoustic_sphere_diameter")

        missing_source_meshes = [
            mesh for mesh in args.meshes if mesh not in source_meshes
        ]
        if missing_source_meshes:
            parser.error(
                "The following meshes from simulation_order.json are missing "
                "from MESH/: " + ", ".join(missing_source_meshes)
            )

        snapshot_root = simulations_directory / raw_order.get('preparation_snapshot', '')
        if raw_order.get('mesh_inputs') != input_fingerprints(source_meshes_directory, snapshot_root/'Parameters'):
            parser.error('MESH input files or per-mesh configuration changed; restore the saved inputs or create a new order.')

        try:
            preflight(args)
        except Exception as error:
            parser.error(str(error))
        if not getattr(args, 'slurm', False):
            save_simulation_order(simulations_directory, order)
        order_store = SimulationOrderStore(simulations_directory)
        if not getattr(args, 'slurm', False):
            reactivate_failed_cases_for_resume(order_store, simulations_directory)
        order = order_store.snapshot()

    # ----------------------------------------------------------------------
    # CREATE NEW ORDER
    # ----------------------------------------------------------------------
    else:
        args.meshes = list(source_meshes.keys())
        missing = []

        if args.rpms is None:
            missing.append("--rpms")
        if args.mode is None:
            missing.append("--mode")
        if args.total_cores is None:
            missing.append("--total-cores")
        if args.cores_per_case is None:
            missing.append("--cores-per-case")
        if args.turbulence is None:
            missing.append("--turbulence")
        if args.wall_functions is None:
            missing.append("--wall-functions")
        if not args.mesh_only and not args.aerodynamics_only and args.acoustic_surface is None:
            missing.append("--acoustic-surface")

        if missing:
            parser.error(
                "The following arguments are required for a new simulation run: "
                + ", ".join(missing)
            )

        if args.total_cores is not None and args.total_cores < 1:
            parser.error("--total-cores must be at least 1")
        if not 1 <= args.cores_per_case <= args.total_cores:
            parser.error("--cores-per-case must be between 1 and --total-cores")

        if len(set(args.rpms or [])) != len(args.rpms or []):
            parser.error("--rpms must not contain duplicate values")
        if any(rpm <= 0 for rpm in args.rpms):
            parser.error("--rpms must be positive")

        if args.field_init == "on" and args.rpms != sorted(args.rpms):
            parser.error(
                "--field-init on requires RPM values in ascending order because "
                "each case is initialized from the preceding RPM case."
            )

        from tools.templates import template_directory
        try:
            template_directory(args.mode, args.turbulence, args.wall_functions)
        except ValueError as error:
            parser.error(str(error))
        validate_acoustic_arguments(parser, args)

        if args.study:
            study_missing = []
            if args.study_file is None:
                study_missing.append("--study-file")
            if args.study_parameter is None:
                study_missing.append("--study-parameter")
            if args.study_values is None:
                study_missing.append("--study-values")

            if study_missing:
                parser.error(
                    "The following arguments are required when --study is set: "
                    + ", ".join(study_missing)
                )

            if len(args.meshes) != 1 or len(args.rpms) != 1:
                parser.error(
                    "When --study is set, MESH/ must contain exactly one mesh "
                    "and exactly one RPM must be provided."
                )

            if args.field_init == "on":
                parser.error(
                    "--field-init on is not supported together with --study. "
                    "Study cases are independent by design."
                )

        try:
            preflight(args)
        except Exception as error:
            parser.error(str(error))
        simulations_directory.mkdir(parents=True, exist_ok=True)
        create_simulation_order(
            args=args,
            simulations_directory=simulations_directory,
        )
        order_store = SimulationOrderStore(simulations_directory)
        order = order_store.snapshot()

    if getattr(args, 'slurm', False):
        from tools.slurm import configure_order
        configure_order(order_store, args)
        order = order_store.snapshot()
    return simulations_directory, source_meshes_directory, source_meshes, order_store, order
