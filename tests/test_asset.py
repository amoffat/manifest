from io import BytesIO
from pathlib import Path

from manifest.utils.asset import get_asset_data
from manifest.utils.types import is_asset


def test_open_file():
    with open(__file__, "rb") as h:
        assert is_asset(type(h))
        assert "test_open_file" in get_asset_data(h).getvalue().decode()


def test_bytesio():
    bio = BytesIO(b"abc")
    assert is_asset(type(bio))
    assert get_asset_data(bio).getvalue() == b"abc"


def test_path():
    path = Path(__file__)
    assert is_asset(type(path))
    assert "test_open_file" in get_asset_data(path).getvalue().decode()
