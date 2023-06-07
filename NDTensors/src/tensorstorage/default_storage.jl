## This is a fil which specifies the default storage type provided some set of parameters
## The parameters are the element type and storage type
default_datatype(eltype::Type=default_eltype()) = Vector{eltype}
default_eltype() = Float64

## This is a system to determine if inds are blocked or unblocked used for default_storagetype
@traitdef is_blocked{T}
@traitimpl is_blocked{T} <- is_blocked(T)
function is_blocked(indsT::Type{<:Tuple})
  return any(map(i -> is_blocked(fieldtype(indsT, i)), length(indsT.parameters)))
end
function is_blocked(inds::Tuple)
  return is_blocked(typeof(inds))
end

is_blocked(::Type{<:Int}) = false
is_blocked(::Type{<:Vector{Int}}) = true

## TODO use multiple dispace to make this pick between dense and blocksparse
@traitfn function default_storagetype(
  datatype::Type{<:AbstractArray}, inds::IndsT
) where {IndsT; !is_blocked{IndsT}}
  datatype = set_parameter_if_unspecified(datatype)
  return Dense{eltype(datatype),datatype}
end

@traitfn function default_storagetype(
  datatype::Type{<:AbstractArray}, inds::IndsT
) where {IndsT <: Tuple; is_blocked{IndsT}}
  datatype = set_parameter_if_unspecified(datatype)
  return BlockSparse{eltype(datatype),datatype,1}
end

function default_storagetype(datatype::Type{<:AbstractArray})
  return default_storagetype(datatype, ())
end

default_storagetype(eltype::Type) = default_storagetype(default_datatype(eltype))
function default_storagetype(eltype::Type, inds::Tuple)
  return default_storagetype(default_datatype(eltype), inds)
end
default_storagetype() = default_storagetype(default_eltype())
