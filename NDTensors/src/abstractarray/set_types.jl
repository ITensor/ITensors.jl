"""
TODO: Use `Accessors.jl` notation:
```julia
@set eltype(arraytype) = eltype
```
"""
function set_eltype(arraytype::Type{<:AbstractArray}, eltype::Type)
  return set_parameters(arraytype, Position(1), eltype)
end

"""
TODO: Use `Accessors.jl` notation:
```julia
@set ndims(arraytype) = ndims
```
"""
function set_ndims(arraytype::Type{<:AbstractArray}, ndims)
  return set_parameters(arraytype, Position(2), ndims)
end

# This is for uniform `Diag` storage which uses
# a Number as the data type.
# TODO: Delete this when we change to using a
# `FillArray` instead. This is a stand-in
# to make things work with the current design.
function set_ndims(numbertype::Type{<:Number}, ndims)
  return numbertype
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

function set_eltype_if_unspecified(
  arraytype::Type{<:AbstractArray{T}}, eltype::Type=default_eltype()
) where {T}
  return arraytype
end

#TODO transition to set_eltype when working for wrapped types
function set_eltype_if_unspecified(
  arraytype::Type{<:AbstractArray}, eltype::Type=default_eltype()
)
  return similartype(arraytype, eltype)
end

function set_parameter_if_unspecified(
  arraytype::Type{<:AbstractArray{ElT,N}}, eltype::Type=default_eltype(), ndims::Integer=1
) where {ElT,N}
  return arraytype
end
function set_parameter_if_unspecified(
  arraytype::Type{<:AbstractArray{ElT}}, eltype::Type=default_eltype(), ndims::Integer=1
) where {ElT}
  return set_ndims(arraytype, ndims)
end
function set_parameter_if_unspecified(
  arraytype::Type{<:AbstractArray}, eltype::Type=default_eltype(), ndims::Integer=1
)
  return set_eltype(set_ndims(arraytype, ndims), eltype)
end
