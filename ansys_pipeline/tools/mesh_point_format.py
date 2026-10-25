"""Canonicalize explicit positive coordinate signs for Foundation mesh readers.

Only leading '+' signs on finite ASCII vector components are removed. Exponent
signs, negative signs, precision, and every other byte are preserved. Repairs
retain the original file and an audit record; unrelated syntax is not repaired.
"""
import json
from pathlib import Path
import re
import shutil
import tempfile

_NUMBER = rb"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_VECTOR = re.compile(rb"\s*\(\s*(" + _NUMBER + rb")\s+(" + _NUMBER
                     + rb")\s+(" + _NUMBER + rb")\s*\)\s*")
_PLUS = re.compile(rb"(?<=[(\s])\+(?=[\d.])")
_SCAN_BYTES = 4 * 1024 * 1024
_VALIDATED = {}


def _signature(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _candidate_lines(stream, first_line):
    """Scan in large reads; only split blocks containing a possible repair.

    Small buffered reads are expensive on Windows-mounted WSL files. Carry
    incomplete lines across block boundaries so validation stays byte exact.
    """
    pending = b''
    line_number = first_line
    while True:
        chunk = stream.read(_SCAN_BYTES)
        data = pending + chunk
        if chunk:
            end = data.rfind(b'\n') + 1
            complete, pending = data[:end], data[end:]
        else:
            complete, pending = data, b''
        if _PLUS.search(complete):
            for line in complete.splitlines(keepends=True):
                if _PLUS.search(line):
                    yield line_number, line
                line_number += 1
        else:
            line_number += complete.count(b'\n')
        if not chunk:
            break


def normalize_point_signs(path):
    """Return number of repaired point records; leave clean/binary files alone."""
    path = Path(path)
    key = str(path.resolve())
    before = _signature(path)
    if _VALIDATED.get(key) == before:
        return 0
    edits = {}
    header = bytearray()
    in_header = True
    with path.open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            if in_header:
                header.extend(line)
                if b'}' not in line:
                    if len(header) > 65536:
                        raise ValueError(f'Unrecognized points header: {path}')
                    continue
                in_header = False
                if re.search(rb'\bformat\s+binary\s*;', header):
                    if _signature(path) == before:
                        _VALIDATED[key] = before
                    return 0
                if not re.search(rb'\bformat\s+ascii\s*;', header):
                    raise ValueError(f'Unrecognized points format: {path}')
                break
        else:
            raise ValueError(f'Unrecognized points header: {path}')
        for line_number, line in _candidate_lines(stream, line_number + 1):
            if not _VECTOR.fullmatch(line):
                raise ValueError(f'Refusing to repair non-vector syntax: {path}:{line_number}')
            edits[line_number] = (line, _PLUS.sub(b'', line))
    if not edits:
        if _signature(path) == before:
            _VALIDATED[key] = before
        return 0

    # Keep the exact offending bytes for diagnosis, including across repeat runs.
    with tempfile.NamedTemporaryFile(prefix=path.name + '.before-sign-fix-',
                                     dir=path.parent, delete=False) as original:
        backup = Path(original.name)
    shutil.copy2(path, backup)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + '.normalized-',
                                         dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            with backup.open('rb') as stream:
                for line_number, line in enumerate(stream, 1):
                    output.write(edits.get(line_number, (line, line))[1])
        shutil.copymode(path, temporary)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    record = {'path': str(path), 'backup': str(backup), 'repairs': [
        {'line': n, 'before': before.decode().strip(), 'after': after.decode().strip()}
        for n, (before, after) in edits.items()
    ]}
    with path.with_name('point-sign-repairs.jsonl').open('a') as audit:
        audit.write(json.dumps(record) + '\n')
    _VALIDATED[key] = _signature(path)
    return len(edits)
