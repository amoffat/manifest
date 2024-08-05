import io
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Type, get_args, get_origin


def extract_type_registry(
    type_registry: dict[str, Any],
    obj: Type | UnionType,
):
    """Extract all enums and dataclasses, recursively, from a type, and store
    them in a registry keyed by name. This is used to rehydrate the types that
    we get back from the LLM."""

    if isinstance(obj, UnionType):
        for arg in obj.__args__:
            extract_type_registry(type_registry, arg)

    elif issubclass(obj, Enum):
        type_registry[obj.__name__] = obj

    elif is_dataclass(obj):
        type_registry[obj.__name__] = obj
        for field in fields(obj):
            extract_type_registry(type_registry, field.type)

    elif get_origin(obj) is list:
        item_type = get_args(obj)[0]
        extract_type_registry(type_registry, item_type)

    elif get_origin(obj) is dict:
        _, value_type = get_args(obj)
        extract_type_registry(type_registry, value_type)

    return type_registry


def is_asset(arg_type: Type) -> bool:
    """Check if a type is an uploadable asset"""
    try:
        if issubclass(arg_type, (io.BytesIO, io.BufferedReader, Path)):
            return True
    # Union types do not work in issubclass, but it's ok
    except TypeError:
        pass

    return False
