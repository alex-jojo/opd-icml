"""Leave space for other jobs while writing Base results and result archives."""
import errno
import io
import math
import os
from pathlib import Path
import shutil

GIB = 1024**3


def require_space(directory, additional=0):
    directory = Path(directory).expanduser().absolute()
    while not directory.exists():
        if directory == directory.parent:
            raise FileNotFoundError(f'Destination filesystem is unavailable: {directory}')
        directory = directory.parent
    reserve = float(os.environ.get('OPD_MIN_FREE_GIB', '5'))
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError('OPD_MIN_FREE_GIB must be finite and nonnegative')
    free = shutil.disk_usage(directory).free
    usable = free - math.ceil(reserve * GIB)
    if usable < additional:
        raise OSError(errno.ENOSPC, f'{directory}: {free/GIB:.2f} GiB free, '
                      f'need {additional/GIB:.2f} GiB plus {reserve:g} GiB reserve')
    return usable


class SpaceCheckedWriter(io.BufferedWriter):
    """Recheck the destination mount at least every 4 MiB of output."""
    def __init__(self, path):
        self.directory = Path(path).parent
        self.allowance = 0
        require_space(self.directory)
        super().__init__(io.FileIO(path, 'w'))

    def write(self, data):
        size = len(data)
        if size > self.allowance:
            self.flush()
            self.allowance = min(4 * 1024**2, require_space(self.directory, size))
        self.allowance -= size
        return super().write(data)
