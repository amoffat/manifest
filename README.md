# Manifest ✨

Call an LLM by calling a function.

- Define a function name, arguments, return value, and docstring.
- Call your function as normal, passing in your values.
- For those values, an LLM will return a response that conforms to your return type.

# Installation

```
pip install manifest
```

Now make sure your OpenAI key is set:

```
export OPENAI_API_KEY="your_api_key_here"
```

# Examples

## Sentiment analysis

```python
from manifest import ai

@ai
def is_optimistic(text: str) -> bool:
    """Determines if the text is optimistic"""

print(is_optimistic("This is amazing!")) # Prints True
```

## Translation

```python
from manifest import ai

@ai
def translate(english_text: str, target_lang: str) -> str:
    """Translates text from english into a target language"""

print(translate("Hello", "fr")) # Prints "Bonjour"
```

## Image analysis

```python
from pathlib import Path
from manifest import ai

@ai
def breed_of_dog(image: Path) -> str:
    """Determines the breed of dog from a photo"""

image = Path("path/to/dog.jpg")
print(breed_of_dog(image)) # Prints "German Shepherd" (or whatever)
```

## Complex objects

```python
from dataclasses import dataclass
from manifest import ai

@dataclass
class Actor:
    name: str
    character: str

@dataclass
class Movie:
    title: str
    director: str
    year: int
    top_cast: list[Actor]

@ai
def similar_movie(movie: str, before_year: int | None=None) -> Movie:
    """Discovers a similar movie, before a certain year, if the year is
    provided."""

like_inception = similar_movie("Inception")
print(like_inception) # Prints a movie similar to inception

```

## Recursive types

It can handle self-referential types. For example, each `Character` has a `social_graph`, and each `SocialGraph` is composed of `Characters`.

```python
from dataclasses import dataclass
from pprint import pprint

from manifest import ai


@dataclass
class Character:
    name: str
    occupation: str
    social_graph: "SocialGraph"


@dataclass
class SocialGraph:
    friends: list[Character]
    enemies: list[Character]


@ai
def get_character_social_graph(character_name: str) -> SocialGraph:
    """For a given fictional character, return their social graph, resolving
    each friend and enemy's social graph recursively."""


graph = get_character_social_graph("Walter White")
pprint(graph)

```

```
SocialGraph(
    friends=[
        Character(
            name='Jesse Pinkman',
            occupation='Meth Manufacturer',
            social_graph=SocialGraph(
                friends=[Character(name='Walter White', occupation='Chemistry Teacher', social_graph=SocialGraph(friends=[], enemies=[]))],
                enemies=[Character(name='Hank Schrader', occupation='DEA Agent', social_graph=SocialGraph(friends=[], enemies=[]))]
            )
        ),
        Character(
            name='Saul Goodman',
            occupation='Lawyer',
            social_graph=SocialGraph(friends=[Character(name='Walter White', occupation='Chemistry Teacher', social_graph=SocialGraph(friends=[], enemies=[]))], enemies=[])
        )
    ],
    enemies=[
        Character(
            name='Hank Schrader',
            occupation='DEA Agent',
            social_graph=SocialGraph(
                friends=[Character(name='Marie Schrader', occupation='Radiologic Technologist', social_graph=SocialGraph(friends=[], enemies=[]))],
                enemies=[Character(name='Walter White', occupation='Meth Manufacturer', social_graph=SocialGraph(friends=[], enemies=[]))]
            )
        ),
        Character(
            name='Gus Fring',
            occupation='Businessman',
            social_graph=SocialGraph(
                friends=[Character(name='Mike Ehrmantraut', occupation='Fixer', social_graph=SocialGraph(friends=[], enemies=[]))],
                enemies=[Character(name='Walter White', occupation='Meth Manufacturer', social_graph=SocialGraph(friends=[], enemies=[]))]
            )
        )
    ]
)
```

# How does it work?

Manifest relies heavily on runtime metadata, such as a function's name,
docstring, arguments, and type hints. It uses all of these to compose a prompt
behind the scenes, then sends the prompt to an LLM. The LLM "executes" the
prompt, and returns a json-based format that we can safely parse back into the
appropriate object.

To get the most out the `@ai` decorator:

- Name your function well.
- Add type hints to your function.
- Add a high-value docstring to your function.

# Limitations

## REPL

Manifest doesn't work from the REPL, due to it needing access to the source code
of the functions it decorates.

## Types

You can only pass in and return the following types:

- Dataclasses
- `Enum` subclasses
- primitives (str, int, bool, None, etc)
- basic container types (list, dict, tuple)
- unions
- Any combination of the above

## Prompts

The prompt templates are also a little fiddly sometimes. They can be improved.

# Initialization

To make things super simple, manifest uses ambient LLM credentials, currently
just `OPENAI_API_KEY`. If environment credentials are not found, you will be
instructed to initialize the library yourself.
