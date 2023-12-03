using LinearAlgebra: LinearAlgebra

LinearAlgebra.norm(a::AbstractSparseArray, p::Real=2) = sparse_norm(a, p)
