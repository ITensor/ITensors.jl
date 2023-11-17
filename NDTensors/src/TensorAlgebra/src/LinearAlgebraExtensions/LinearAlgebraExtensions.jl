module LinearAlgebraExtensions
using LinearAlgebra: LinearAlgebra, qr
using ..TensorAlgebra:
  TensorAlgebra,
  BipartitionedPermutation,
  bipartition,
  bipartitioned_permutations,
  matricize,
  unmatricize

include("qr.jl")
end
