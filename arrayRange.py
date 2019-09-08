"""
Numpy's `split` can split a multidimensional array into non-overlapping
sub-arrays. However, this is not a memory-efficient way of dealing with
non-overlapping partitions of an array because it effectively doubles
memory usage.

This module provides an iterable generator that produces tuples of slices,
each of which can be used to index into a Numpy array and obtain a small
view into it. It is very memory-efficient since no copy of the array is
ever created.

This all works because Numpy ndarrays can be indexed using a tuple of
slices: that is, `arr[a:b, c:d, e:f]` is equivalent to
`arr[(slice(a, b), slice(c, d), slice(e, f))]`.

This module doesn't import Numpy at all (except in the unit test when
`__name__` is '__main__') since it generates pure-Python slices.
"""

from itertools import product
from typing import List


def arrayRange(start: List[int], stop: List[int], step: List[int]):
  """
  Makes an iterable of non-overlapping slices, e.g., to partition an array

  Returns an iterable of tuples of slices, each of which can be used to
  index into a multidimensional array such as Numpy's ndarray.

  >> [arr[tup] for tup in arrayRange([0, 0], arr.shape, [5, 7])]
  
  where `arr` can be indexed with a tuple of slices (e.g., Numpy), will
  evaluate to a list of sub-arrays.

  Same arguments as `range` except all three arguments are required and
  expected to be list-like of same length. `start` indicates the indexes
  to start each dimension. `stop` indicates the stop index for each
  dimension. `step` is the size of the chunk in each dimension.
  """
  assert len(start) == len(stop)
  assert len(stop) == len(step)
  assert all(map(lambda x: x > 0, step))
  startRangesGen = map(lambda v: range(*v), zip(start, stop, step))
  startToSliceMapper = lambda multiStart: tuple(
      slice(i, min(i + step, stop)) for i, stop, step in zip(multiStart, stop, step))
  return map(startToSliceMapper, product(*startRangesGen))


def splitFor(func, arr, sizes):
  """
  For-iterable riff on Numpy's `split`.

  Given a function `func` that takes only an array, apply it to
  non-overlapping sub-arrays of `arr`. Each sub-array will have size
  equal to or less than `sizes`. `arr` may be multidimensional, so
  `sizes` has to be the same length as `arr.shape`.

  This function returns nothing. `func`'s return value is ignored.
  """
  for tup in arrayRange([0, 0], arr.shape, sizes):
    func(arr[tup])


if __name__ == '__main__':
  import numpy as np

  def test(shape, step):
    x = np.arange(np.prod(shape)).reshape(shape) + 1
    y = x.copy()
    neg = np.sum(x < 0)
    for tup in arrayRange([0, 0], x.shape, step):
      x[tup] *= -1
      newneg = np.sum(x < 0)
      assert newneg > neg
      neg = newneg
    assert np.array_equal(x, -y)

  test([4, 7], [2, 4])
  test([4, 7], [1, 1])
  test([4, 7], [5, 4])
  test([4, 7], [5, 9])
