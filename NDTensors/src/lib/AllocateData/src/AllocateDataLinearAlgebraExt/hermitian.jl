using LinearAlgebra: LinearAlgebra, Hermitian
using ..AllocateData: AllocateData, to_dim

function (arraytype::Type{<:LinearAlgebra.Hermitian})(
  ::AllocateData.UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange,2}}
)
  # TODO: Check the parent type of `arraytype`.
  a = Array{eltype(arraytype)}(AllocateData.undef, axes)
  return Hermitian(a)
end

## TODO: For some reason this is broken, and gives an error:
## ERROR: UndefVarError: `T` not defined
## function LinearAlgebra.Hermitian{T}(
##   ::AllocateData.UndefInitializer, axes::Tuple{Vararg{AbstractUnitRange,2}}
## ) where {T}
##   a = Array{T}(AllocateData.undef, axes)
##   return Hermitian(a)
## end
