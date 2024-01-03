using Compat: allequal
using LinearAlgebra: LinearAlgebra, Diagonal
using ..AllocateData: AllocateData, to_dim

function LinearAlgebra.Diagonal{T}(
  ::AllocateData.UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange,2}}
) where {T}
  dims = to_dim.(axes)
  @assert allequal(dims)
  diag_dim = first(dims)
  if VERSION < v"1.7.0-DEV.986"
    # https://github.com/JuliaLang/julia/pull/38282
    return Diagonal(Vector{T}(Base.undef, diag_dim))
  end
  return Diagonal{T}(Base.undef, diag_dim)
end
