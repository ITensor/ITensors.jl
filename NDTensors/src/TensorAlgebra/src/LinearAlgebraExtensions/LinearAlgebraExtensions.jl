module LinearAlgebraExtensions
using LinearAlgebra: LinearAlgebra, qr
using ..TensorAlgebra: TensorAlgebra, BipartitionedPermutation, bipartitioned_permutations, matricize

include("qr.jl")
end
