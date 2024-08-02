# Manifest

```
man·i·fest [verb]

: to make something happen by imagining it and consciously thinking that it will happen
```

Manifest is an experiment in letting an LLM provide the return value for a
function. It allows you to manifest powerful behaviors into existence merely by
defining the function's facade.

Shout out to [@dstufft](https://github.com/dstufft) for gifting me the PyPI repo
name [`manifest`](https://pypi.org/project/manifest/) 🙏

```
pip install manifest
```

# Examples

## Sentiment analysis

```python
from manifest import ai

@ai
def is_optimistic(text: str) -> bool:
    """ Determines if the text is optimistic"""
    ...

assert is_optimistic("This is amazing!")
```

## Translation

```python
from manifest import ai

@ai
def translate(english_text: str, target_lang: str) -> str:
    """ Translates text from english into a target language """
    ...

assert translate("Hello", "fr") == "Bonjour"
```

## Image analysis

You can pass in bytes to make use of a model's multimodal abilities. COMING SOON

```python
import io
from manifest import ai

@ai
def breed_of_dog(image: io.BytesIO) -> str:
    """ Determines the breed of dog from a photo """
    ...

image = open("/path/to/terrier.jpg", "r")
print(breed_of_dog(image))
```

## Complex objects

Your function can use fairly complex data structures.

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
    ...

like_inception = similar_movie("Inception")
print(like_inception)

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

You can only pass in and return the following types:

- Dataclasses
- `Enum` subclasses
- primitives (str, int, bool, None, etc)
- basic container types (list, dict)
- unions
- Any combination of the above

The prompt templates are also a little fiddly sometimes. They can be improved.

# Initialization

To make things super simple, manifest uses ambient LLM credentials, currently
just `OPENAI_API_KEY`. If environment credentials are not found, you will be
instructed to initialize the library yourself.
