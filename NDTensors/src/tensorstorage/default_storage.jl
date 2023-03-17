## This is a fil which specifies the default storage type provided some set of parameters
## The parameters are the element type and storage type
default_datatype(eltype::Type=default_eltype()) = Vector{eltype}
default_eltype() = Float64

## TODO use multiple dispace to make this pick between dense and blocksparse
function default_storagetype(datatype::Type{<:AbstractArray}, inds::Tuple)
  datatype = set_parameter_if_unspecified(datatype)
  return Dense{eltype(datatype),datatype}
end

function default_storagetype(datatype::Type{<:AbstractArray})
  return default_storagetype(datatype, ())
end

default_storagetype(eltype::Type) = default_storagetype(default_datatype(eltype))
function default_storagetype(eltype::Type, inds::Tuple)
  return default_storagetype(default_datatype(eltype), inds)
end
default_storagetype() = default_storagetype(default_eltype())
