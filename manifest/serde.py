from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import (
    Any,
    ForwardRef,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import jsonschema

from manifest.utils.types import resolve_forward_ref


def is_primitive_type(obj_type: Any):
    """Are we dealing with a simple type?"""
    return obj_type in (int, float, str, bool, type(None))


T = TypeVar("T")


def serialize(
    *,
    data_type: Type | UnionType | Optional[T],
    caller_ns: dict[str, Any],
) -> dict:
    definitions: dict[str, dict] = {}
    refs: set[str] = set()
    schema = _serialize(
        data_type=data_type,
        definitions=definitions,
        refs=refs,
        caller_ns=caller_ns,
    )

    # Only include schema definitions that were actually used as refs
    if refs:
        final_defs = {}
        for ref in refs:
            final_defs[ref] = deepcopy(definitions[ref])

        schema["$defs"] = final_defs

    return schema


def _serialize(
    *,
    data_type: Type | UnionType | Optional[T] | Enum,
    definitions: dict[str, dict],
    refs: set[str],
    caller_ns: dict[str, Any],
) -> dict:
    """Serializes a data type into its jsonschema.

    :param data_type: The type to serialize.
    """
    if data_type is str:
        return {"type": "string"}

    if data_type is int:
        return {"type": "integer"}

    if data_type is float:
        return {"type": "number"}

    if data_type is bool:
        return {"type": "boolean"}

    if data_type is None or data_type is type(None):
        return {"type": "null"}

    if is_dataclass(data_type):
        properties = {}
        required = []
        for field in fields(data_type):
            field_type = resolve_forward_ref(field.type)
            properties[field.name] = _serialize(
                data_type=field_type,
                definitions=definitions,
                refs=refs,
                caller_ns=caller_ns,
            )
            if field.default is field.default_factory or field.default is None:
                required.append(field.name)
        schema = {
            "type": "object",
            "properties": properties,
            "dataclassType": cast(type, data_type).__name__,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required

        ref = cast(type, data_type).__name__
        definitions[ref] = schema

        return schema

    if get_origin(data_type) is list:
        item_type = resolve_forward_ref(get_args(data_type)[0])
        return {
            "type": "array",
            "items": _serialize(
                data_type=item_type,
                definitions=definitions,
                refs=refs,
                caller_ns=caller_ns,
            ),
        }

    if get_origin(data_type) is tuple:
        item_types = get_args(data_type)
        return {
            "type": "array",
            "prefixItems": [
                _serialize(
                    data_type=resolve_forward_ref(t),
                    definitions=definitions,
                    refs=refs,
                    caller_ns=caller_ns,
                )
                for t in item_types
            ],
        }

    if get_origin(data_type) is dict:
        _, value_type = get_args(data_type)
        value_type = resolve_forward_ref(value_type)
        return {
            "type": "object",
            "additionalProperties": _serialize(
                data_type=value_type,
                definitions=definitions,
                refs=refs,
                caller_ns=caller_ns,
            ),
        }

    # This catches Optional as well
    if get_origin(data_type) in (Union, UnionType):
        types = get_args(data_type)
        schemas = [
            _serialize(
                data_type=t,
                definitions=definitions,
                refs=refs,
                caller_ns=caller_ns,
            )
            for t in types
        ]
        return {"anyOf": schemas}

    if isinstance(data_type, ForwardRef):
        ref = data_type.__forward_arg__
        refs.add(ref)

        if ref not in definitions:
            # prevents recursion, when we _serialize below
            definitions[ref] = {}

            concrete_type = data_type._evaluate(
                caller_ns,
                caller_ns,
                frozenset(),
            )
            definitions[ref] = _serialize(
                data_type=concrete_type,
                definitions=definitions,
                refs=refs,
                caller_ns=caller_ns,
            )

        return {"$ref": f"#/$defs/{ref}"}

    if isinstance(data_type, type):
        if issubclass(data_type, Enum):
            return {
                "type": "string",
                "enum": [e.name for e in data_type],
                "enumType": data_type.__name__,
            }

    raise TypeError(f"Unsupported type: {data_type}")


def deserialize(*, schema: dict, data: Any, registry: dict[str, Type]) -> Any:
    jsonschema.validate(data, schema)

    definitions = schema.get("$defs", {})
    return _deserialize(
        schema=schema,
        data=data,
        registry=registry,
        definitions=definitions,
    )


def _deserialize(
    *,
    schema: dict,
    data: Any,
    registry: dict[str, Type],
    definitions: dict[str, dict],
) -> Any:
    """
    Deserialize a JSON-friendly data structure into a dataclass instance.

    :param spec: The JSON-encodable data structure that describes the dataclass.
    :param data: The JSON-encodable data structure to deserialize.
    :param registry: A dictionary mapping type names to the corresponding types.

    :return: An instance of the dataclass.
    """

    type = schema.get("type")

    def dataclass_deserializer(*, class_name, fields_spec, data):
        dc = registry.get(class_name)
        if not dc:
            raise ValueError(f"Unknown dataclass: {class_name}")

        # Prepare fields to instantiate the dataclass
        kwargs = {}
        for field_name, field_spec in fields_spec.items():
            kwargs[field_name] = _deserialize(
                schema=field_spec,
                data=data[field_name],
                registry=registry,
                definitions=definitions,
            )

        return dc(**kwargs)

    if type == "object":
        if "dataclassType" in schema:
            class_name = schema["dataclassType"]
            fields_spec = schema["properties"]
            return dataclass_deserializer(
                class_name=class_name,
                fields_spec=fields_spec,
                data=data,
            )

        # dict container type
        if "additionalProperties" in schema:
            value_type = schema["additionalProperties"]
            return {
                k: _deserialize(
                    schema=value_type,
                    data=v,
                    registry=registry,
                    definitions=definitions,
                )
                for k, v in data.items()
            }

    # union type
    if "anyOf" in schema:
        # Try to find the first option that works
        for option in schema["anyOf"]:
            try:
                return _deserialize(
                    schema=option,
                    data=data,
                    registry=registry,
                    definitions=definitions,
                )
            except:  # noqa E722
                pass

    # list container type
    if type == "array":

        # If items is a collection, we have a tuple
        if "prefixItems" in schema:
            return tuple(
                _deserialize(
                    schema=schema["prefixItems"][i],
                    data=d,
                    registry=registry,
                    definitions=definitions,
                )
                for i, d in enumerate(data)
            )

        # Otherwise, we have a normal list
        return [
            _deserialize(
                schema=schema["items"],
                data=d,
                registry=registry,
                definitions=definitions,
            )
            for d in data
        ]

    if type == "string":

        if "enum" in schema:
            enum_cls = registry.get(schema["enumType"])
            if not enum_cls:
                raise ValueError(f"Unknown enum: {schema['enumType']}")

            return enum_cls[data]

        return data

    if type == "integer":
        return data

    if type == "number":
        return data

    if type == "boolean":
        return data

    if type == "null":
        return None

    if type is None:
        ref = schema.get("$ref")
        if ref:
            class_name = ref.split("/")[-1]
            fields_spec = definitions[class_name]["properties"]
            return dataclass_deserializer(
                class_name=class_name,
                fields_spec=fields_spec,
                data=data,
            )

    raise ValueError(f"Unsupported schema: {schema}")
