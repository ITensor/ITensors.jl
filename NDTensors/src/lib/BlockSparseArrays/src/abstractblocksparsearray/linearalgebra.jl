using LinearAlgebra: Adjoint, Transpose

# Like: https://github.com/JuliaLang/julia/blob/v1.11.1/stdlib/LinearAlgebra/src/transpose.jl#L184
# but also takes the dual of the axes.
# Fixes an issue raised in:
# https://github.com/ITensor/ITensors.jl/issues/1336#issuecomment-2353434147
function Base.copy(a::Adjoint{T,<:AbstractBlockSparseMatrix{T}}) where {T}
  a_dest = similar(parent(a), axes(a))
  a_dest .= a
  return a_dest
end

# More efficient than the generic `LinearAlgebra` version.
function Base.copy(a::Transpose{T,<:AbstractBlockSparseMatrix{T}}) where {T}
  a_dest = similar(parent(a), axes(a))
  a_dest .= a
  return a_dest
end
