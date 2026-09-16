"""Parameter validation shared by every test estimator.

Two styles live here side by side:

- ``validate_string`` -- the scikit-learn-flavoured check used by the newer
  estimators, which type their categorical hyperparameters as plain ``str`` and
  validate the value explicitly against an allowed set;
- ``_validate_params`` -- the annotation-driven decorator used by
  :class:`~bbttest.tests.bbt.bbt.BBTTest`, kept here so both packages share a
  single validation module rather than reaching across into each other.
"""

from __future__ import annotations

import sys
from functools import wraps
from typing import TYPE_CHECKING, Literal, get_args, get_origin

if TYPE_CHECKING:
    from collections.abc import Collection

if sys.version_info >= (3, 12):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType


def validate_string(value: str, allowed: Collection[str], name: str) -> str:
    """Validate that a string hyperparameter is one of the allowed options.

    This is the scikit-learn-style alternative to ``Literal`` typing: the
    parameter is annotated ``str`` and its value is checked explicitly, which
    keeps type hints simple and validation visible at the call site.

    Parameters
    ----------
    value : str
        The value to validate.
    allowed : Collection[str]
        The permitted values.
    name : str
        The parameter name, for the error message.

    Returns
    -------
    str
        The validated value, unchanged.

    Raises
    ------
    ValueError
        If ``value`` is not in ``allowed``.
    """
    if value not in allowed:
        raise ValueError(
            f"Invalid value '{value}' for parameter '{name}'. "
            f"Expected one of {tuple(allowed)}."
        )
    return value


def is_literal_value(value: object, typx: object) -> bool:
    """Return whether ``value`` is a permitted member of a ``Literal`` annotation."""
    if isinstance(typx, TypeAliasType):
        typx = typx.__value__
    if get_origin(typx) is Literal:
        return value in get_args(typx)
    return False


def _validate_params(func):
    """Decorate a function to validate its ``Literal``-annotated arguments."""
    from inspect import Parameter, signature

    sig = signature(func)
    accepts_kwargs = any(
        param.kind == Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not accepts_kwargs:
            for kwarg in kwargs:
                if kwarg not in sig.parameters:
                    raise ValueError(f"Unexpected keyword argument '{kwarg}'")
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for name, value in bound_args.arguments.items():
            param = sig.parameters[name]
            if param.kind == Parameter.VAR_KEYWORD:
                continue
            # If type annotation is a Literal, validate the value
            if param.annotation is not param.empty and is_literal_value(
                value, param.annotation
            ):
                continue  # Valid value, continue to next parameter
            elif (
                param.annotation is not param.empty
                and get_origin(param.annotation) is Literal
            ):
                raise ValueError(
                    f"Invalid value '{value}' for parameter '{name}'. "
                    f"Expected one of {get_args(param.annotation)}."
                )
        return func(*args, **kwargs)

    return wrapper
