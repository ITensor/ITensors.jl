"""
TODO: Use `Accessors.jl` notation:
```julia
@set eltype(arraytype) = eltype
```
"""
function set_eltype(arraytype::Type{<:AbstractArray}, eltype::Type)
  return error(
    "Setting the element type of the array type `$arraytype` (to `$eltype`) is not currently defined.",
  )
end

"""
TODO: Use `Accessors.jl` notation:
```julia
@set ndims(arraytype) = ndims
```
"""
function set_ndims(arraytype::Type{<:AbstractArray}, ndims)
  return error(
    "Setting the number dimensions of the array type `$arraytype` (to `$ndims`) is not currently defined.",
  )
end

"""
`set_indstype` should be overloaded for
types with structured dimensions,
like `OffsetArrays` or named indices
(such as ITensors).
"""
function set_indstype(arraytype::Type{<:AbstractArray}, dims::Tuple)
  return set_ndims(arraytype, length(dims))
end

set_properties_if_unspecified(arraytype::Type{<:AbstractArray{ElT, N}}, eltype::Type = default_eltype(), ndims::Integer = 1) where {ElT, N} = arraytype
set_properties_if_unspecified(arraytype::Type{<:AbstractArray{ElT}}, eltype::Type = default_eltype(), ndims::Integer = 1) where {ElT} = set_ndims(arraytype, ndims)
set_properties_if_unspecified(arraytype::Type{<:AbstractArray}, eltype::Type = default_eltype(), ndims::Integer = 1) = set_eltype(set_ndims(arraytype, ndims), eltype)