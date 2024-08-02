from dataclasses import fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import Any, Type, get_args, get_origin

import jsonschema


def is_primitive_type(obj_type: Any):
    """Are we dealing with a simple type?"""
    return obj_type in (int, float, str, bool, type(None))


def serialize(data_type: Type | UnionType) -> dict:
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
            field_type = field.type
            properties[field.name] = serialize(field_type)
            if field.default is field.default_factory or field.default is None:
                required.append(field.name)
        schema = {
            "type": "object",
            "properties": properties,
            "dataclassType": data_type.__name__,
        }
        if required:
            schema["required"] = required
        return schema

    if get_origin(data_type) is list:
        item_type = get_args(data_type)[0]
        return {
            "type": "array",
            "items": serialize(item_type),
        }

    if get_origin(data_type) is dict:
        _, value_type = get_args(data_type)
        return {
            "type": "object",
            "additionalProperties": serialize(value_type),
        }

    if isinstance(data_type, UnionType):
        types = get_args(data_type)
        schemas = [serialize(t) for t in types]
        return {"anyOf": schemas}

    if issubclass(data_type, Enum):
        return {
            "type": "string",
            "enum": [e.name for e in data_type],
            "enumType": data_type.__name__,
        }

    raise TypeError(f"Unsupported type: {data_type}")


def deserialize(
    *,
    schema: dict,
    data: Any,
    registry: dict[str, Any],
) -> Any:
    """
    Deserialize a JSON-friendly data structure into a dataclass instance.

    :param spec: The JSON-friendly data structure that describes the dataclass.
    :param data: The JSON-friendly data structure to deserialize.
    :param registry: A dictionary mapping type names to the corresponding types.

    :return: An instance of the dataclass.
    """
    jsonschema.validate(data, schema)

    type = schema.get("type")

    if type == "object":
        if "dataclassType" in schema:
            class_name = schema["dataclassType"]
            fields_spec = schema["properties"]

            dc = registry.get(class_name)
            if not dc:
                raise ValueError(f"Unknown dataclass: {class_name}")

            # Prepare fields to instantiate the dataclass
            kwargs = {}
            for field_name, field_spec in fields_spec.items():
                kwargs[field_name] = deserialize(
                    schema=field_spec,
                    data=data[field_name],
                    registry=registry,
                )

            return dc(**kwargs)

        # dict container type
        if "additionalProperties" in schema:
            value_type = schema["additionalProperties"]
            return {
                k: deserialize(
                    schema=value_type,
                    data=v,
                    registry=registry,
                )
                for k, v in data.items()
            }

    # union type
    if "anyOf" in schema:
        # Try to find the first option that works
        for option in schema["anyOf"]:
            try:
                return deserialize(
                    schema=option,
                    data=data,
                    registry=registry,
                )
            except:  # noqa E722
                pass

    # list container type
    if type == "array":
        return [
            deserialize(
                schema=schema["items"],
                data=d,
                registry=registry,
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

    raise ValueError(f"Unsupported schema: {schema}")
