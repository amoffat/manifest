import io
from pathlib import Path
from typing import Any


def get_asset_data(arg_value: Any) -> io.BytesIO:
    """Get the data from an asset"""

    if isinstance(arg_value, io.BytesIO):
        return arg_value

    if isinstance(arg_value, io.BufferedReader):
        pos = arg_value.tell()
        bio = io.BytesIO(arg_value.read())
        arg_value.seek(pos)
        return bio

    if isinstance(arg_value, Path):
        return io.BytesIO(arg_value.read_bytes())

    raise TypeError(f"Unsupported asset type: {type(arg_value)}")
