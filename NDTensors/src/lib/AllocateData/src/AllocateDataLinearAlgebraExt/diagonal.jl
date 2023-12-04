using Compat: allequal
using LinearAlgebra: LinearAlgebra, Diagonal
using ..AllocateData: AllocateData, to_dim

function LinearAlgebra.Diagonal{T}(
  ::AllocateData.UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange,2}}
) where {T}
  dims = to_dim.(axes)
  @assert allequal(dims)
  return Diagonal{T}(Base.undef, first(dims))
end
