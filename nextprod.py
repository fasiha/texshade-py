"""
Direct ports of Julia's `nextprod` [1], with helper function `nextpow` [2],
which are in Julia 1.2.0.

[1] https://github.com/JuliaLang/julia/blob/c6da87ff4bc7a855e217856757ad3413cf6d1f79/base/combinatorics.jl#L248-L262
[2] https://github.com/JuliaLang/julia/blob/c6da87ff4bc7a855e217856757ad3413cf6d1f79/base/intfuncs.jl#L334-L356

This derivative work is licensed under the MIT License, the same license as Julia Base.
"""

from typing import List
import math


def nextpow(a: float, x: float):
  """The smallest `a^n` not less than `x`, where `n` is a non-negative integer.
  
  `a` must be greater than 1, and `x` must be greater than 0.
  # Examples
  ```jldoctest
  julia> nextpow(2, 7)
  8
  julia> nextpow(2, 9)
  16
  julia> nextpow(5, 20)
  25
  julia> nextpow(4, 16)
  16
  ```
  """
  assert x > 0 and a > 1
  if x <= 1:
    return 1.0
  n = math.ceil(math.log(x, a))
  p = a**(n - 1)
  return p if p >= x else a**n


def nextprod(a: List[int], x: int):
  """Next integer greater than or equal to `n` that can be written as ``\\prod k_i^{p_i}`` for integers
  ``p_1``, ``p_2``, etc.
  # Examples
  ```jldoctest
  julia> nextprod([2, 3], 105)
  108
  julia> 2^2 * 3^3
  108
  ```
  """
  k = len(a)
  v = [1] * k  # current value of each counter
  mx = [nextpow(ai, x) for ai in a]  # maximum value of each counter
  v[0] = mx[0]  # start at first case that is >= x
  p = mx[0]  # initial value of product in this case
  best = p
  icarry = 1

  while v[-1] < mx[-1]:
    if p >= x:
      best = p if p < best else best  # keep the best found yet
      carrytest = True
      while carrytest:
        p = p // v[icarry - 1]
        v[icarry - 1] = 1
        icarry += 1
        p *= a[icarry - 1]
        v[icarry - 1] *= a[icarry - 1]
        carrytest = v[icarry - 1] > mx[icarry - 1] and icarry < k
      if p < x:
        icarry = 1
    else:
      while p < x:
        p *= a[0]
        v[0] *= a[0]
  return mx[-1] if mx[-1] < best else best
