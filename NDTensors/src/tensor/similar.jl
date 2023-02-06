# NDTensors.similar
similar(tensor::Tensor) = setstorage(tensor, similar(storage(tensor)))

# NDTensors.similar
similar(tensor::Tensor, eltype::Type) = setstorage(tensor, similar(storage(tensor), eltype))

# NDTensors.similar
similar(tensor::Tensor, dims::Tuple) = setstorage(tensor, similar(storage(tensor), dims))
similar(tensor::Tensor, dims::Dims) = setstorage(tensor, similar(storage(tensor), dims))

# NDTensors.similar
function similar(tensor::Tensor, eltype::Type, dims::Tuple)
  return setstorage(tensor, similar(storage(tensor), eltype, dims))
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
  return set_storagetype(tensortype_new_inds, similartype(storagetype(tensortype_new_inds)))
end
