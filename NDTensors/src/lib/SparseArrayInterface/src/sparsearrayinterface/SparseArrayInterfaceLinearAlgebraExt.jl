using LinearAlgebra: norm

sparse_norm(a::AbstractArray, p::Real=2) = norm(sparse_storage(a))
