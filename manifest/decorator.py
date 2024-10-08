import inspect
import json
from functools import wraps
from io import BytesIO
from types import UnionType
from typing import Any, Callable, cast, get_origin, get_type_hints

from manifest import exc, initialize, parser, serde, tmpl
from manifest.llm.base import LLM
from manifest.types.service import Service
from manifest.utils.args import get_args_dict
from manifest.utils.asset import get_asset_data
from manifest.utils.types import extract_type_registry, is_asset

# Lazily-initialized singleton LLM client. This must be lazy because we don't
# know when the user will have initialized the client, either automatically or
# manually. We just know it must be initialized when a decorated function is
# executed.
llm: LLM | None = None


def ai(*decorator_args, **decorator_kwargs) -> Callable:
    """Decorates a function whose return value will be provided by an
    LLM.

    :param decorator_args: The function to be decorated, if called like @ai
    :param decorator_kwargs: Configuration options for the decorator, if called
        like @ai(some_arg=123). Currently unused.
    """

    def outer(fn: Callable) -> Callable:
        """Decorator that wraps a function and produces a function that, when
        executed, will call the LLM to provide the return value."""
        caller_frame = inspect.currentframe().f_back.f_back  # type: ignore
        caller_ns = cast(dict[str, Any], caller_frame.f_globals)  # type: ignore

        ants = get_type_hints(fn)
        name = fn.__name__
        fn_src: str = inspect.getsource(fn)

        # Populate the type registry. Data types in the LLM's responses will be
        # restricted to hydrating only these types.
        type_registry: dict[str, Any] = {}
        for value in ants.values():
            extract_type_registry(
                registry=type_registry,
                obj=value,
                caller_ns=caller_ns,
            )

        try:
            return_type = ants["return"]
        except KeyError:
            raise ValueError(f"Function '{name}' is missing a return type annotation")

        @wraps(fn)
        def inner(*exec_args, **exec_kwargs) -> Any:
            """The decorated function that will be called in place of the
            wrapped function."""

            # The LLM client must be initialized lazily, as it may depend on
            # either automatic or manual initialization.
            global llm
            if not llm:
                llm = initialize.make_llm()

            # We use the LLM object to serialize the return type to jsonschema,
            # instead of using serde.serialize directly. This is because the LLM
            # class may know about what the LLM service supports, for example,
            # if the type needs to be wrapped in an envelope. We do the same
            # with deserialization, later in the call.
            return_type_spec = llm.serialize(
                return_type=return_type,
                caller_ns=caller_ns,
            )
            return_type_spec_json = json.dumps(return_type_spec, indent=2)

            service: Service = llm.service()
            call_tmpl = tmpl.load(f"{service.value}/call.j2")
            system_tmpl = tmpl.load(f"{service.value}/system.j2")

            # Consolidate args and kwargs into a single dict
            call_args = get_args_dict(
                fn=fn,
                args=exec_args,
                kwargs=exec_kwargs,
            )

            images: list[BytesIO] = []

            # Aggregate our arguments into a format that the LLM can understand
            args = []
            for arg_name, arg_value in call_args.items():

                try:
                    arg_type = ants[arg_name]
                except KeyError:
                    raise ValueError(
                        f"Function '{name}' is missing an annotation for '{arg_name}'"
                    )

                if is_asset(arg_type):
                    images.append(get_asset_data(arg_value))

                    args.append(
                        {
                            "name": arg_name,
                            "value": "Uploaded image",
                        }
                    )
                    continue

                # TODO handle checking union types
                elif isinstance(arg_type, UnionType):
                    pass

                elif get_origin(arg_type) is not None:
                    if type(arg_value) is not get_origin(arg_type):
                        raise TypeError(
                            f"Argument '{arg_name}' is of type {type(arg_value)}, "
                            f"but should be of type {get_origin(arg_type)}"
                        )

                elif not isinstance(arg_value, arg_type):
                    raise TypeError(
                        f"Argument '{arg_name}' is of type {type(arg_value)}, "
                        f"but should be of type {arg_type}"
                    )

                # Collect the source code of the argument type, if it's not a
                # builtin type
                src = None
                try:
                    src = inspect.getsource(arg_type)
                except TypeError:
                    pass

                args.append(
                    {
                        "name": arg_name,
                        "schema": serde.serialize(
                            data_type=arg_type,
                            caller_ns=caller_ns,
                        ),
                        "value": arg_value,
                        "src": src,
                    }
                )

            ret_src = None
            try:
                ret_src = inspect.getsource(return_type)
            except TypeError:
                pass

            tmpl_params = {
                "fn": fn_src,
                "args": args,
                "return_type": {
                    "schema": return_type_spec_json,
                    "src": ret_src,
                },
            }
            prompt = call_tmpl.render(**tmpl_params)
            system_msg = system_tmpl.render()

            def complete():
                resp = llm.call(
                    prompt=prompt,
                    system_msg=system_msg,
                    response_schema=return_type_spec,
                    images=images,
                )

                try:
                    data_spec = parser.parse_return_value(resp)
                    ret_val = llm.deserialize(
                        schema=return_type_spec,
                        data=data_spec,
                        registry=type_registry,
                    )
                    return ret_val
                except Exception as e:
                    raise exc.DeserializationError(
                        f"Failed to deserialize LLM response: {e}"
                    )

            tries = max(decorator_kwargs.get("retry", 2), 0) + 1
            completed = False
            for _ in range(tries):
                try:
                    ret_val = complete()
                except exc.DeserializationError:
                    continue
                else:
                    completed = True
                    break

            if not completed:
                raise ValueError("Failed to complete the LLM prompt")
            return ret_val

        return inner

    # The decorator was called without parentheses, eg:
    #   @ai
    #   def my_function():
    #       ...
    if len(decorator_args) == 1 and callable(decorator_args[0]):
        return outer(decorator_args[0])

    # The decorator was called with parentheses, eg:
    #   @ai(some_arg=123)
    #   def my_function():
    #       ...
    else:
        return outer
