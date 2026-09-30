"""Shared helper for the NumPy 2 ``__array__`` protocol."""

from typing import Any, Optional

import numpy as np


def array_view(source: np.ndarray, dtype: Any = None,
               copy: Optional[bool] = None) -> np.ndarray:
    """Implement the ``__array__(dtype, copy)`` contract for an array.

    ``source`` is normally the zero-copy view of an underlying samcore
    buffer.  A dtype conversion or ``copy=True`` returns a new array;
    ``copy=False`` raises when either would be required.
    """
    requested = None if dtype is None else np.dtype(dtype)
    if requested is not None and requested != source.dtype:
        if copy is False:
            raise ValueError(
                f"cannot avoid a copy when converting to {requested}")
        return source.astype(requested, copy=True)
    if copy:
        return source.copy()
    return source
