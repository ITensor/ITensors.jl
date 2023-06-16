adapt_structure(to, x::Union{MPS,MPO}) = map(xᵢ -> adapt(to, xᵢ), x)
