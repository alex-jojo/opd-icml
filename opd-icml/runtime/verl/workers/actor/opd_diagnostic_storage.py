"""Lossless symmetric-moment archives; independent of CUDA, Ray and verl."""
from pathlib import Path
import zipfile

import numpy as np


def compact_moment(arrays):
    result = dict(arrays)
    matrix = np.asarray(result['second'])
    # Preserve unusual/asymmetric input verbatim rather than silently averaging it.
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.array_equal(matrix, matrix.T):
        return result
    n = len(matrix)
    packed = np.empty(n * (n + 1) // 2, dtype=matrix.dtype)
    offset = 0
    for row in range(n):
        width = n - row
        packed[offset:offset+width] = matrix[row, row:]
        offset += width
    del result['second']
    result.update(second_upper=packed, second_dimension=n, storage_schema='symmetric-upper-v1')
    return result


def load_moment(path):
    """Read both legacy dense and new lossless triangular NPZ files."""
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if 'second_upper' not in arrays:
        return arrays
    if str(arrays.get('storage_schema')) != 'symmetric-upper-v1':
        raise ValueError('Unknown symmetric archive schema')
    n = int(arrays.pop('second_dimension'))
    packed = arrays.pop('second_upper')
    if n < 0 or packed.ndim != 1 or len(packed) != n * (n+1)//2:
        raise ValueError('Invalid triangular matrix size')
    matrix = np.empty((n, n), dtype=packed.dtype)
    offset = 0
    for row in range(n):
        width = n - row
        matrix[row, row:] = packed[offset:offset+width]
        matrix[row:, row] = packed[offset:offset+width]
        offset += width
    arrays['second'] = matrix
    return arrays


def write_moment(path, arrays):
    """Atomic, low-compression archive; caller handles distributed rank ownership."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, value in compact_moment(arrays).items():
                with archive.open(name + '.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asarray(value), allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
