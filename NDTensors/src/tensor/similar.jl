# NDTensors.similar
similar(tensor::Tensor) = setstorage(tensor, similar(storage(tensor)))

# NDTensors.similar
similar(tensor::Tensor, eltype::Type) = setstorage(tensor, similar(storage(tensor), eltype))

# NDTensors.similar
function similar(tensor::Tensor, dims::Tuple)
  return setinds(setstorage(tensor, similar(storage(tensor), dims)), dims)
end

# NDTensors.similar
function similar(tensor::Tensor, dims::Dims)
  return setinds(setstorage(tensor, similar(storage(tensor), dims)), dims)
end

# NDTensors.similar
function similar(tensortype::Type{<:Tensor}, dims::Tuple)
  # TODO: Is there a better constructor pattern for this?
  # Maybe use `setstorage(::Type{<:Tensor}, ...)` and
  # `setinds(::Type{<:Tensor}, ...)`?
  return similartype(tensortype, dims)(
    AllowAlias(), similar(storagetype(tensortype), dims), dims
  )
end

# NDTensors.similar
function similar(tensortype::Type{<:Tensor}, dims::Dims)
  # TODO: Is there a better constructor pattern for this?
  # Maybe use `setstorage(::Type{<:Tensor}, ...)` and
  # `setinds(::Type{<:Tensor}, ...)`?
  return similartype(tensortype, dims)(
    AllowAlias(), similar(storagetype(tensortype), dims), dims
  )
end

# NDTensors.similar
function similar(tensor::Tensor, eltype::Type, dims::Tuple)
  return setinds(setstorage(tensor, similar(storage(tensor), eltype, dims)), dims)
end

# NDTensors.similar
function similar(tensor::Tensor, eltype::Type, dims::Dims)
  return setinds(setstorage(tensor, similar(storage(tensor), eltype, dims)), dims)
end

# Base overloads
Base.similar(tensor::Tensor) = NDTensors.similar(tensor)
Base.similar(tensor::Tensor, eltype::Type) = NDTensors.similar(tensor, eltype)
Base.similar(tensor::Tensor, dims::Tuple) = NDTensors.similar(tensor, dims)
Base.similar(tensor::Tensor, dims::Dims) = NDTensors.similar(tensor, dims)
function Base.similar(tensor::Tensor, eltype::Type, dims::Tuple)
  return NDTensors.similar(tensor, eltype, dims)
end
function Base.similar(tensor::Tensor, eltype::Type, dims::Dims)
  return NDTensors.similar(tensor, eltype, dims)
end

function similartype(tensortype::Type{<:Tensor}, eltype::Type)
  return set_storagetype(tensortype, similartype(storagetype(tensortype), eltype))
end

function similartype(tensortype::Type{<:Tensor}, dims::Tuple)
  tensortype_new_inds = set_indstype(tensortype, dims)
  # Need to pass `dims` in case that information is needed to make a storage type,
  # for example `BlockSparse` needs the number of dimensions.
  storagetype_new_inds = similartype(storagetype(tensortype_new_inds), dims)
  return set_storagetype(tensortype_new_inds, storagetype_new_inds)
end
