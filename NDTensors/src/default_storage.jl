## This is a fil which specifies the default storage type provided some set of parameters
## The parameters are the element type and storage type
default_datatype(eltype::Type=default_eltype()) = Vector{eltype}
default_eltype() = Float64

## TODO use multiple dispace to make this pick between dense and blocksparse
function default_storagetype(datatype::Type{<:AbstractArray}, inds::Tuple)
  datatype = set_eltype_if_unspecified(datatype, default_eltype())
  if eltype(inds) == Integer || length(inds) == 0
    return Dense{eltype(datatype),datatype}
  else
    ## return sparsetype
    return BlockSparse{eltype(datatype),datatype}
  end
end

function default_storagetype(datatype::Type{<:AbstractArray})
  return default_storagetype(datatype, ())
end

default_storagetype(eltype::Type{<:Number}) = default_storagetype(default_datatype(eltype))
default_storagetype() = default_storagetype(default_eltype())
