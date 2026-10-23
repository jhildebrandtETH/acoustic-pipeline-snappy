"""Prepare mesh-dependent fixed wall omega without changing wall treatment."""
from pathlib import Path

from tools.settings_report import read_dictionary
from tools.snappy_pipeline import docker_run, write


SOURCE = r'''
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "IOdictionary.H"
#include "IFstream.H"
#include "nearWallDist.H"
#include "wallFvPatch.H"
#include <cmath>

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::noParallel();
    #include "setRootCase.H"
    #include "createTime.H"
    runTime.setTime(instant(0, "0"), 0);
    #include "createMesh.H"

    IFstream controlStream(runTime.path()/"Parameters"/"wallResolvedDict");
    dictionary controls(controlStream);
    IOdictionary transport(IOobject("transportProperties", runTime.constant(),
        mesh, IOobject::MUST_READ, IOobject::NO_WRITE));
    const dimensionedScalar nu("nu", dimensionSet(0, 2, -1, 0, 0, 0, 0), transport);
    const scalar beta1 = controls.lookupOrDefault<scalar>("omegaBeta1", 0.075);
    const scalar coefficient = controls.lookupOrDefault<scalar>("omegaCoefficient", 60);
    if (!std::isfinite(nu.value()) || nu.value() <= 0
        || !std::isfinite(beta1) || beta1 <= 0
        || !std::isfinite(coefficient) || coefficient <= 0)
    {
        FatalErrorInFunction << "Wall omega requires positive finite nu, beta1 and coefficient"
            << exit(FatalError);
    }
    volScalarField omega(IOobject("omega", "0", mesh,
        IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    const label patchi = mesh.boundaryMesh().findIndex("propeller");
    if (patchi < 0 || !isA<wallFvPatch>(mesh.boundary()[patchi])
        || omega.boundaryField()[patchi].type() != "fixedValue")
    {
        FatalErrorInFunction << "Expected a fixedValue omega on the propeller wall"
            << exit(FatalError);
    }
    const scalarField& y = nearWallDist::New(mesh).y()[patchi];
    if (y.empty())
    {
        FatalErrorInFunction << "Empty propeller wall" << exit(FatalError);
    }
    scalarField values(y.size());
    forAll(y, facei)
    {
        if (!std::isfinite(y[facei]) || y[facei] <= 0)
        {
            FatalErrorInFunction << "Invalid wall distance on face " << facei
                << exit(FatalError);
        }
        values[facei] = coefficient*nu.value()/(beta1*sqr(y[facei]));
        if (!std::isfinite(values[facei]))
        {
            FatalErrorInFunction << "Nonfinite wall omega" << exit(FatalError);
        }
    }
    omega.boundaryFieldRef()[patchi] == values;
    IOstream::defaultPrecision(17);
    if (!omega.write())
    {
        FatalErrorInFunction << "Could not write 0/omega" << exit(FatalError);
    }
    Info<< "Wall distance min/max [m]: " << min(y) << " " << max(y) << nl
        << "Wall omega min/max [1/s]: " << min(values) << " " << max(values) << nl
        << "Prepared " << y.size() << " propeller faces using local wall distances" << nl;
    return 0;
}
'''


def wall_omega_command(case):
    """Stage a serial OpenFOAM utility, or skip unrelated/manual wall treatments."""
    case = Path(case)
    settings = case / 'Parameters/wallResolvedDict'
    field = case / '0/omega'
    if not settings.is_file() or not field.is_file():
        return None
    controls = read_dictionary(settings, case, {}, [])
    if str(controls.get('automaticOmega', 'false')).lower() not in ('true', 'yes', 'on', '1'):
        return None
    omega = read_dictionary(field, case, {}, [])
    if omega['boundaryField']['propeller']['type'] != 'fixedValue':
        return None
    directory = case / 'mesh/wallOmega'
    write(directory / 'calculateWallOmega.C', SOURCE)
    write(directory / 'Make/files', 'calculateWallOmega.C\n\nEXE = $(PWD)/calculateWallOmega\n')
    continuation = chr(92) + '\n    '
    write(directory / 'Make/options',
          'EXE_INC = ' + continuation + '-I$(LIB_SRC)/finiteVolume/lnInclude '
          + continuation + '-I$(LIB_SRC)/meshTools/lnInclude\n\n'
          'EXE_LIBS = ' + continuation + '-lfiniteVolume ' + continuation + '-lmeshTools\n')
    return ['bash', '-c', 'cd /simulation/mesh/wallOmega && wmake && '
            './calculateWallOmega -case /simulation']


def prepare_wall_omega(container, case, callback=None, live=False):
    """Calculate after NCC and field mapping, before decomposition; never on resume."""
    command = wall_omega_command(case)
    if command is None:
        return False
    docker_run(container, case, command, 'log.wallOmega', callback, live)
    return True
