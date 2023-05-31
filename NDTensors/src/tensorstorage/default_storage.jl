## This is a fil which specifies the default storage type provided some set of parameters
## The parameters are the element type and storage type
default_datatype(eltype::Type=default_eltype()) = Vector{eltype}
default_eltype() = Float64

## This is a system to determine if inds are blocked or unblocked used for default_storagetype
@traitdef is_blocked_inds{indsT}
@traitimpl is_blocked_inds{indsT} <- is_blocked_inds(indsT)
function is_blocked_inds(indsT::Type{<:Tuple})
  return all(map(i -> is_blocked_ind(fieldtype(indsT, i)), length(indsT.parameters)))
end
function is_blocked_inds(inds::Tuple)
  return is_blocked_inds(typeof(inds))
end

@traitdef is_blocked_ind{IndT}
@traitimpl is_blocked_ind{IndT} <- is_blocked_ind(IndT)
function is_blocked_ind(ind)
  return is_blocked_ind(typeof(ind))
end
is_blocked_ind(::Type{<:Int}) = false
is_blocked_ind(::Type{<:Vector{Int}}) = true

## TODO use multiple dispace to make this pick between dense and blocksparse
@traitfn function default_storagetype(
  datatype::Type{<:AbstractArray}, inds::IndsT
) where {IndsT; !is_blocked_inds{IndsT}}
  datatype = set_parameter_if_unspecified(datatype)
  return Dense{eltype(datatype),datatype}
end

@traitfn function default_storagetype(
  datatype::Type{<:AbstractArray}, inds::IndsT
) where {IndsT <: Tuple; is_blocked_inds{IndsT}}
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
