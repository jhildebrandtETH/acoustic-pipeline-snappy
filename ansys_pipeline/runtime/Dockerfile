FROM microfluidica/openfoam@sha256:d718c3c360f0113e61f8a2e7dcceb15b8f57b4c836d1ee379075bc833eef27b8

USER root
ARG OPENFOAM_PACKAGE_VERSION=20260624
RUN apt-get update \
 && apt-get install -y --no-install-recommends --only-upgrade "openfoam13=${OPENFOAM_PACKAGE_VERSION}" \
 && test "$(dpkg-query -W -f='${Version}' openfoam13)" = "${OPENFOAM_PACKAGE_VERSION}" \
 && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="Acoustic pipeline OpenFOAM 13" \
      org.opencontainers.image.version="13-20260624" \
      org.openfoam.package.version="20260624"

ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /simulation
