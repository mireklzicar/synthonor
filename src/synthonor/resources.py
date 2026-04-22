from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path


@contextmanager
def example_space_path() -> Iterator[Path]:
    """Yield a real filesystem path for the bundled example synthon TSV."""
    resource = files("synthonor.data").joinpath("synthon_space_1M.tsv")
    with as_file(resource) as resolved_path:
        yield Path(resolved_path)
