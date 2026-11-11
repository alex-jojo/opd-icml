"""Project-relative public paths, anchored independently of a process's working directory."""
import os
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent


def project_absolute(value):
    path = Path(value).expanduser()
    if os.name == 'nt' and path.anchor and not path.drive:
        path = Path(ROOT.drive + str(path))
    elif not path.is_absolute():
        path = ROOT / path
    # Normalize .. without resolving model/cache symlinks into machine-specific paths.
    return Path(os.path.abspath(path))


def project_relative(value):
    try:
        return Path(os.path.relpath(project_absolute(value), ROOT)).as_posix()
    except ValueError:
        raise ValueError('Project paths must be on the same filesystem drive as the project') from None


def saved_path_matches(saved, current):
    """Accept portable paths and legacy absolute metadata for the same logical project path."""
    if not isinstance(saved, str) or not saved:
        return False
    expected = project_relative(current)
    normalized = str(saved).replace('\\', '/').rstrip('/')
    if normalized == expected:
        return True
    legacy_absolute = Path(saved).is_absolute() or PureWindowsPath(saved).is_absolute()
    if legacy_absolute and '..' not in Path(expected).parts and normalized.endswith('/' + expected):
        return True
    return project_absolute(saved).resolve() == project_absolute(current).resolve()
