
"""
    parity_sign(P)

Given an array or tuple of integers representing  
a permutation or a subset of a permutation, 
compute the parity sign defined as -1 for a 
permutation consisting of an odd number of swaps 
and +1 for an even number of swaps. This 
implementation uses an O(n^2) algorithm and is 
intended for small permutations only.
"""
function parity_sign(P)::Int
  L = length(P)
  s = +1
  for i in 1:L, j in (i + 1):L
    s *= sign(P[j] - P[i])
  end
  return s
end
