# Pinned OpenFOAM 13 runtime

Default image: `acoustic-pipeline/openfoam:13-20260624`. Official Foundation OpenFOAM package: `openfoam13=20260624`. The prior workstation image contained `20260212`.

From the `ansys_pipeline` directory in Linux/WSL:

```bash
bash runtime/build.sh
```

The recipe pins the prior base image by digest, upgrades only the OpenFOAM package to the exact revision, and verifies the package version. It does not silently use a floating latest package. The old image remains available. The updated image was built locally; on another Docker host it must be built or loaded before running the pipeline. No image is published to a registry by this change.

`tools/runtime.py` provides the default used by local execution, preflight, default Slurm configuration and the opt-in NCC smoke test. Explicit `docker_image` settings in saved Slurm orders retain their recorded value; create a fresh order with the updated config to change those deliberately. New example Slurm config uses the new image. Runtime image selection is independent of saved `Parameters`/`CoreTemplates` snapshots.

## Revert the runtime only

Set `IMAGE` in `tools/runtime.py` back to `microfluidica/openfoam:13`, or use the exact prior base digest below for reproducible comparison:

```text
microfluidica/openfoam@sha256:d718c3c360f0113e61f8a2e7dcceb15b8f57b4c836d1ee379075bc833eef27b8
```

For Slurm, select the desired image explicitly in a new order's JSON config. Do not edit an active saved order's immutable settings. No other solver or mesh settings need be reverted. Pressure PCG/DIC, Poisson wall distance, disabled built-in yPlus and tip-label support remain separate changes; see [crash workarounds](../docs/runtime-crash-workarounds.md).

Updating a runtime is a diagnostic comparison, not a demonstrated fix for the heap corruption. The last crash was detected during an SST update, but the original corrupting operation was not identified. Compare the same checkpoint and 24-rank decomposition first; increasing ranks changes the decomposition as well as performance.

Sources: [official v13 patch releases](https://openfoam.org/news/v13-patch/), [official package upgrade instructions](https://openfoam.org/download/13-ubuntu/).
