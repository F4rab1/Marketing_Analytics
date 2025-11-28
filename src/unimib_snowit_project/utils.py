"""Utils"""

from __future__ import annotations

import re
import json
from warnings import warn

from pathlib import Path


# I/O Utilities

def get_root_dir() -> Path:
    """Get the path of the root dir of the package."""
    return Path(__file__).resolve().parent.parent.parent.absolute()


# Data Clean Utilities

def clean_str(string: str,
              case: str | None = None
              ) -> str | None:
    """Basic cleaning of string input."""
    assert isinstance(string, str)
    if case is not None:
        assert case in ['lower', 'upper']
    clean = string.strip()
    if clean == '':
        return None
    if case == 'lower':
        clean = clean.lower()
    elif case == 'upper':
        clean = clean.upper()
    return clean


def get_list_from_str(string: str,
                      has_list_brakets: bool = True,
                      sep: str = ','
                      ) -> list[str]:
    """Get a python list object from a string-formatted list."""
    assert isinstance(string, str)
    assert isinstance(has_list_brakets, bool)
    assert isinstance(sep, str)
    s = clean_str(string)
    if s is None:
        return []
    if has_list_brakets:
        s = re.sub(r'^\[', '', s)
        s = re.sub(r'\]$', '', s)
    return s.split(sep)


def get_dict_from_jsonstr(string: str) -> list[str]:
    """Get the python dict from a string-formatted json."""
    assert isinstance(string, str)
    try:
        json.loads(string)
    except json.JSONDecodeError:
        warn(f"Impossible to decode as JSON the string '{string}'")
