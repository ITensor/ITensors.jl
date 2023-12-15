module LinearAlgebraExtensions
using LinearAlgebra: LinearAlgebra, qr
using ..TensorAlgebra:
  TensorAlgebra,
  BlockedPermutation,
  blockedperms

include("qr.jl")
end
