import io
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, ForwardRef, Type, get_args, get_origin


def resolve_forward_ref(obj: Any) -> Any:
    """Ensure that an object is a ForwardRef, if it's a string. This helps
    smooth over the difference between List['Node'] and list['Node']"""
    if isinstance(obj, str):
        return ForwardRef(obj)
    return obj


def extract_type_registry(
    *,
    registry: dict[str, Any],
    obj: Type | UnionType,
    caller_ns: dict[str, Any],
):
    """Extract all enums and dataclasses, recursively, from a type, and store
    them in a registry keyed by name. This is used to rehydrate the types that
    we get back from the LLM."""

    if isinstance(obj, UnionType):
        for arg in obj.__args__:
            extract_type_registry(
                registry=registry,
                obj=arg,
                caller_ns=caller_ns,
            )

    elif is_dataclass(obj):
        registry[obj.__name__] = obj
        for field in fields(obj):
            extract_type_registry(
                registry=registry,
                obj=resolve_forward_ref(field.type),
                caller_ns=caller_ns,
            )

    elif get_origin(obj) is list:
        item_type = resolve_forward_ref(get_args(obj)[0])
        extract_type_registry(
            registry=registry,
            obj=item_type,
            caller_ns=caller_ns,
        )

    elif get_origin(obj) is dict:
        _, value_type = get_args(obj)
        value_type = resolve_forward_ref(value_type)
        extract_type_registry(
            registry=registry,
            obj=value_type,
            caller_ns=caller_ns,
        )

    # There's nothing to do here, because by the time we've seen a ForwardRef,
    # we've already seen the class that it refers to
    elif isinstance(obj, ForwardRef):
        registry[obj.__forward_arg__] = obj._evaluate(
            caller_ns,
            caller_ns,
            frozenset(),
        )

    elif issubclass(obj, Enum):
        registry[obj.__name__] = obj

    return registry


def is_asset(arg_type: Type) -> bool:
    """Check if a type is an uploadable asset"""
    try:
        if issubclass(arg_type, (io.BytesIO, io.BufferedReader, Path)):
            return True
    # Union types do not work in issubclass, but it's ok
    except TypeError:
        pass

    return False
