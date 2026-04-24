from __future__ import annotations

import os
from importlib.resources import as_file, files
from pathlib import Path
from shutil import copyfile
from tempfile import gettempdir


_PathBase = type(Path())


class _ContextCompatiblePath(_PathBase):
    """Path subclass that also works with `with ...` for backward compatibility."""

    def __enter__(self) -> Path:
        return Path(self)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


def _default_example_dir() -> Path:
    candidates: list[Path] = []
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(Path(local_app_data) / "synthonor" / "examples")
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        candidates.append(Path(xdg_cache_home) / "synthonor" / "examples")
    candidates.append(Path.home() / ".cache" / "synthonor" / "examples")
    candidates.append(Path(gettempdir()) / "synthonor" / "examples")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    raise OSError("Unable to create a writable cache directory for synthonor example data")


def example_space_path() -> Path:
    """Return a stable writable path to the bundled example synthon TSV."""
    target_path = _default_example_dir() / "synthon_space_1M.tsv"
    if target_path.exists() and target_path.stat().st_size > 0:
        return _ContextCompatiblePath(target_path)

    resource = files("synthonor.data").joinpath("synthon_space_1M.tsv")
    with as_file(resource) as resolved_path:
        copyfile(Path(resolved_path), target_path)
    return _ContextCompatiblePath(target_path)
