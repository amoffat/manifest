from dataclasses import dataclass
from enum import Enum

import pytest

from manifest.serde import deserialize, serialize


def test_simple() -> None:
    schema = serialize(str)
    expected = "hello"

    actual = deserialize(
        schema=schema,
        data=expected,
        registry={},
    )

    assert actual == expected


def test_tuple() -> None:
    schema = serialize(tuple[int, str, float])
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

    schema = serialize(Shape)
    data = "CUBE"

    actual = deserialize(
        schema=schema,
        data=data,
        registry={"Shape": Shape},
    )

    expected = Shape.CUBE

    assert actual == expected


def test_dict() -> None:
    schema = serialize(dict[str, int])
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
    schema = serialize(type(None))
    expected = None

    actual = deserialize(
        schema=schema,
        data=expected,
        registry={},
    )

    assert actual == expected


@pytest.mark.parametrize("expected", [None, 123, "hello"])
def test_union(expected) -> None:
    schema = serialize(str | int | None)

    actual = deserialize(
        schema=schema,
        data=expected,
        registry={},
    )

    assert actual == expected


def test_list() -> None:
    schema = serialize(list[int])
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

    schema = serialize(list[Color])
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
    schema = serialize(expected.__class__)

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
    schema = serialize(expected.__class__)

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
