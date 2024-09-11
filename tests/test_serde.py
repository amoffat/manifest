from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pytest

from manifest.serde import deserialize, serialize


def test_simple() -> None:
    schema = serialize(data_type=str, caller_ns=locals())
    expected = "hello"

    actual = deserialize(
        schema=schema,
        data=expected,
        registry={},
    )

    assert actual == expected


def test_tuple() -> None:
    schema = serialize(
        data_type=tuple[int, str, float],
        caller_ns=locals(),
    )
    expected = (123, "hello", 3.14)

    actual = deserialize(
        schema=schema,
        data=[123, "hello", 3.14],
        registry={},
    )

    assert actual == expected


def test_enum() -> None:
    class Shape(Enum):
        SPHERE = 1
        CUBE = 2
        CYLINDER = 3

    schema = serialize(data_type=Shape, caller_ns=locals())
    data = "CUBE"

    actual = deserialize(
        schema=schema,
        data=data,
        registry={"Shape": Shape},
    )

    expected = Shape.CUBE

    assert actual == expected


def test_dict() -> None:
    schema = serialize(data_type=dict[str, int], caller_ns=locals())
    expected = {
        "a": 1,
        "b": 2,
        "c": 3,
    }

    actual = deserialize(
        schema=schema,
        data=expected,
        registry={},
    )

    assert actual == expected


def test_none() -> None:
    schema = serialize(data_type=type(None), caller_ns=locals())
    expected = None

    actual = deserialize(
        schema=schema,
        data=expected,
        registry={},
    )

    assert actual == expected


@pytest.mark.parametrize("expected", [None, 123, "hello"])
def test_union(expected) -> None:
    schema = serialize(data_type=str | int | None, caller_ns=locals())

    actual = deserialize(
        schema=schema,
        data=expected,
        registry={},
    )

    assert actual == expected


def test_list() -> None:
    schema = serialize(data_type=list[int], caller_ns=locals())
    expected = [1, 2, 3]

    actual = deserialize(
        schema=schema,
        data=expected,
        registry={},
    )

    assert actual == expected


def test_list_of_dataclass() -> None:
    @dataclass
    class Color:
        r: int
        g: int
        b: int

    schema = serialize(data_type=list[Color], caller_ns=locals())
    data = [
        {
            "r": 125,
            "g": 200,
            "b": 255,
        },
        {
            "r": 0,
            "g": 0,
            "b": 0,
        },
    ]

    actual = deserialize(
        schema=schema,
        data=data,
        registry={"Color": Color},
    )

    expected = [
        Color(r=125, g=200, b=255),
        Color(r=0, g=0, b=0),
    ]

    assert actual == expected


def test_dataclass() -> None:
    @dataclass
    class Food:
        name: str
        shape: str
        weight: float
        cost: int

    expected = Food(
        name="apple",
        shape="sphere",
        weight=0.16,
        cost=50,
    )
    schema = serialize(data_type=expected.__class__, caller_ns=locals())

    data = {
        "name": "apple",
        "shape": "sphere",
        "weight": 0.16,
        "cost": 50,
    }

    actual = deserialize(
        schema=schema,
        data=data,
        registry={"Food": Food},
    )

    assert actual == expected


def test_complex() -> None:
    @dataclass
    class Color:
        r: int
        g: int
        b: int

    class Shape(Enum):
        SPHERE = 1
        CUBE = 2
        CYLINDER = 3

    @dataclass
    class RandomObject:
        name: str
        shape: Shape | None
        weight: float  # in kg
        dimensions: list[float]
        color: Color
        cost: int = 0  # in cents

    expected = RandomObject(
        name="toaster",
        shape=Shape.CUBE,
        weight=2.75,
        dimensions=[10.5, 7.8, 12.3],
        color=Color(r=125, g=200, b=255),
        cost=499,
    )
    schema = serialize(data_type=expected.__class__, caller_ns=locals())

    data = {
        "name": "toaster",
        "shape": "CUBE",
        "weight": 2.75,
        "dimensions": [10.5, 7.8, 12.3],
        "color": {
            "r": 125,
            "g": 200,
            "b": 255,
        },
        "cost": 499,
    }

    actual = deserialize(
        schema=schema,
        data=data,
        registry={
            "RandomObject": RandomObject,
            "Color": Color,
            "Shape": Shape,
        },
    )

    assert actual == expected


def test_optional() -> None:
    schema = serialize(data_type=Optional[str], caller_ns=locals())

    actual = deserialize(
        schema=schema,
        data=None,
        registry={},
    )

    assert actual is None

    actual = deserialize(
        schema=schema,
        data="hello",
        registry={},
    )

    assert actual == "hello"


def test_recursive() -> None:
    @dataclass
    class Node:
        value: int
        next: Optional["Node"]

    schema = serialize(data_type=Node, caller_ns=locals())

    data = {
        "value": 1,
        "next": {
            "value": 2,
            "next": {
                "value": 3,
                "next": None,
            },
        },
    }

    actual = deserialize(
        schema=schema,
        data=data,
        registry={"Node": Node},
    )

    expected = Node(
        value=1,
        next=Node(
            value=2,
            next=Node(
                value=3,
                next=None,
            ),
        ),
    )

    assert actual == expected
