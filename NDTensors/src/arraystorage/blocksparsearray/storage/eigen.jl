function LinearAlgebra.eigen(a::BlockSparseArray; kwargs...)
  return error("Not implemented")
end

function LinearAlgebra.eigen(
  a::Union{Hermitian{<:Real,<:BlockSparseArray},Hermitian{<:Complex,<:BlockSparseArray}};
  kwargs...,
)
  return error("Not implemented")
end
