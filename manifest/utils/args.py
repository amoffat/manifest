import inspect


def get_args_dict(*, fn, args, kwargs):
    """Get the arguments of a function as a dictionary by consolidating the
    positional and keyword arguments, then baking in the values and defaults."""

    # Get the signature of the function
    signature = inspect.signature(fn)

    # Bind the arguments to the function's signature
    bound_args = signature.bind(*args, **kwargs)

    # Apply the defaults (if any)
    bound_args.apply_defaults()

    # Return the arguments as an ordered dictionary
    return dict(bound_args.arguments)
