"""Read NDPS galaxy/halo catalogs (pickle or ASCII) downloaded from the NDPS web service.

Single-file module that needs only the standard library and NumPy, so it can be
copied next to your script instead of installed.

A catalog job produces, per snapshot:
    galaxy_NNNN.{pkl,txt}      structured array of object properties (halo_ for halos)
    galaxy_pid_NNNN.{pkl,txt}  member particle IDs (only when requested)
    snapinf_NNNN.txt           snapshot metadata, one ``key = <json>`` per line
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np

__all__ = ['read', 'read_catalog', 'read_members', 'read_snapinfo']
__version__ = '0.1.0'

_LABEL = re.compile(r'^(.*?)(?:\[(\d+(?:,\d+)*)\])?$')


class _NumpyUnpickler(pickle.Unpickler):
    """Unpickler that only rebuilds NumPy arrays, so untrusted files cannot run code."""
    _allowed = {
        ('numpy', 'ndarray'), ('numpy', 'dtype'),
        ('numpy.core.multiarray', '_reconstruct'), ('numpy.core.multiarray', 'scalar'),
        ('numpy._core.multiarray', '_reconstruct'), ('numpy._core.multiarray', 'scalar'),
        # Pickle protocol 5 (what gsq_v2.py writes) rebuilds arrays from raw buffers.
        ('numpy.core.numeric', '_frombuffer'), ('numpy._core.numeric', '_frombuffer'),
    }

    def find_class(self, module, name):
        if (module, name) in self._allowed:
            # NumPy 2 pickles name numpy._core; NumPy 1 only has numpy.core.
            if module.startswith('numpy._core') and not hasattr(np, '_core'):
                module = 'numpy.core' + module[len('numpy._core'):]
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'Refusing to load {module}.{name} from catalog pickle')


def _load_pickle(filename, trusted):
    with open(filename, 'rb') as stream:
        if trusted:
            return pickle.load(stream)
        return _NumpyUnpickler(stream).load()


def _is_pickle(filename):
    suffix = Path(filename).suffix.lower()
    if suffix in ('.pkl', '.pickle'):
        return True
    if suffix == '.txt':
        return False
    raise ValueError(f'Cannot tell catalog format from extension: {filename}')


def _convert(tokens):
    """Convert one column of text tokens to int64, float64, or str."""
    values = np.asarray(tokens, dtype=str)
    for dtype in (np.int64, np.float64):
        try:
            return values.astype(dtype)
        except ValueError:
            pass
    return np.array([json.loads(token) if token.startswith('"') else token
                     for token in tokens], dtype=str)


def _read_catalog_ascii(filename):
    with open(filename, encoding='ascii') as stream:
        header = stream.readline()
        if not header.startswith('#'):
            raise ValueError(f'{filename}: missing "# column..." header line')
        labels = header[1:].strip().split('\t') if header[1:].strip() else []
        rows = [line.rstrip('\n').split('\t') for line in stream if line.strip()]
    for number, row in enumerate(rows, start=2):
        if len(row) != len(labels):
            raise ValueError(f'{filename}:{number}: expected {len(labels)} columns, got {len(row)}')

    # Group flattened vector columns such as ConFrac_M[0], ConFrac_M[1] by field name.
    fields = {}
    for column, label in enumerate(labels):
        name, index = _LABEL.match(label).groups()
        index = tuple(int(i) for i in index.split(',')) if index else ()
        fields.setdefault(name, []).append((index, column))

    columns = list(zip(*rows)) if rows else [()] * len(labels)
    nrow = len(rows)
    arrays, formats = {}, []
    for name, entries in fields.items():
        shape = tuple(max(index[d] for index, _ in entries) + 1
                      for d in range(len(entries[0][0])))
        converted = [_convert(columns[column]) for _, column in entries]
        dtype = np.result_type(*converted) if converted else np.float64
        if nrow == 0:
            dtype = np.dtype(np.float64)
        data = np.zeros((nrow,) + shape, dtype=dtype)
        for (index, _), values in zip(entries, converted):
            data[(slice(None),) + index] = values
        arrays[name] = data
        formats.append((name, data.dtype, shape))

    catalog = np.empty(nrow, dtype=formats)
    for name, data in arrays.items():
        catalog[name] = data
    return catalog


def _read_members_ascii(filename):
    ids, counts, pids = [], [], []
    with open(filename, encoding='ascii') as stream:
        for number, line in enumerate(stream, start=1):
            if line.startswith('#') or not line.strip():
                continue
            values = np.array(line.split(), dtype=np.int64)
            if values.size < 2 or values.size - 2 != values[1]:
                raise ValueError(f'{filename}:{number}: Npart does not match the number of particle IDs')
            ids.append(values[0])
            counts.append(values[1])
            pids.append(values[2:])
    members = np.empty(len(ids), dtype=[('ID', np.int64), ('Npart', np.int64), ('P_ID', object)])
    members['ID'] = ids
    members['Npart'] = counts
    for i, particles in enumerate(pids):
        members['P_ID'][i] = particles
    return members


def read_catalog(filename, trusted=False):
    """Read galaxy_/halo_NNNN.{pkl,txt} into a NumPy structured array (one row per object).

    Vector properties keep their shape, e.g. ``catalog['SFR'][:, 3]``. ASCII files
    store no dtypes, so integer columns come back as int64 and the rest as float64.
    Set ``trusted=True`` to allow arbitrary pickle content from a source you trust.
    """
    if _is_pickle(filename):
        return _load_pickle(filename, trusted)
    return _read_catalog_ascii(filename)


def read_members(filename, trusted=False):
    """Read galaxy_/halo_pid_NNNN.{pkl,txt} into a structured array with ID, Npart, P_ID.

    ``P_ID`` holds one integer array of particle IDs per object.
    """
    if _is_pickle(filename):
        return _load_pickle(filename, trusted)
    return _read_members_ascii(filename)


def read_snapinfo(filename):
    """Read snapinf_NNNN.txt into a dict (values decoded from JSON)."""
    info = {}
    with open(filename, encoding='ascii') as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            key, separator, value = line.partition(' = ')
            if not separator:
                raise ValueError(f'{filename}:{number}: expected "key = value"')
            info[key.strip()] = json.loads(value)
    return info


def read(filename, trusted=False):
    """Read any NDPS catalog output, choosing the reader from the file name."""
    name = Path(filename).name
    if name.startswith('snapinf_'):
        return read_snapinfo(filename)
    if re.match(r'(galaxy|halo)_pid_', name):
        return read_members(filename, trusted)
    if re.match(r'(galaxy|halo)_', name):
        return read_catalog(filename, trusted)
    raise ValueError(f'Unrecognized NDPS catalog file name: {name}')
