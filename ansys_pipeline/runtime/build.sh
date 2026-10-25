#!/usr/bin/env bash
set -euo pipefail
directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
docker build --progress=plain -t acoustic-pipeline/openfoam:13-20260624 "$directory"
version=$(docker run --rm --entrypoint dpkg-query acoustic-pipeline/openfoam:13-20260624 -W '-f=${Version}' openfoam13)
test "$version" = 20260624
printf 'Verified OpenFOAM package %s\n' "$version"
