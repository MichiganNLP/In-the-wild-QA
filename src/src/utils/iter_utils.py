from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence

from typing import Any, Literal, TypeVar

T = TypeVar("T")


# See `grouper` in https://docs.python.org/3/library/itertools.html#itertools-recipes.
def batch_iter(iterable: Iterable[T], n: int, *, incomplete: Literal["fill", "ignore"] = "ignore",
               fill_value: Any | None = None) -> Iterator[Iterable[T]]:
    """Batches the data into non-overlapping fixed-length batches.

    Examples:

    grouper("ABCDEFGH", 3) --> ABC DEF
    grouper("ABCDEFGH", 3, incomplete="fill", fill_value="x") --> ABC DEF GHx
    """
    args = [iter(iterable)] * n
    if incomplete == "fill":
        return itertools.zip_longest(*args, fillvalue=fill_value)
    elif incomplete == "ignore":
        return zip(*args)
    else:
        raise ValueError(f"Expected 'fill' or 'ignore'; got '{incomplete}'")

# See https://stackoverflow.com/a/312464/1165181
def batch_sequence(seq: Sequence[T], n: int) -> Iterator[Sequence[T]]:
    """Yield successive n-sized chunks from `seq`."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
