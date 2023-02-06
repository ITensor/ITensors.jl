# NDTensors.similar
similar(storage::TensorStorage) = setdata(storage, NDTensors.similar(data(storage)))

# NDTensors.similar
similar(storage::TensorStorage, eltype::Type) = setdata(storage, NDTensors.similar(data(storage), eltype))

# NDTensors.similar
similar(storage::TensorStorage, dims::Tuple) = setdata(storage, NDTensors.similar(data(storage), dims))

# NDTensors.similar
function similar(storagetype::Type{<:TensorStorage}, eltype::Type, dims::Tuple)
  return setdata(storagetype, NDTensors.similar(datatype(storagetype), eltype, dims))
end
# NDTensors.similar
function similar(storagetype::Type{<:TensorStorage}, eltype::Type)
  return error("Must specify dimensions.")
end
# NDTensors.similar
function similar(storagetype::Type{<:TensorStorage}, dims::Tuple)
  return NDTensors.similar(storagetype, eltype(storagetype), dims)
end

# NDTensors.similar
function similar(storagetype::Type{<:TensorStorage}, dims::DimOrInd...)
  return NDTensors.similar(storagetype, eltype(storagetype), dims)
end

# NDTensors.similar
function similar(storagetype::Type{<:TensorStorage}, dims::Dims)
  return NDTensors.similar(storagetype, eltype(storagetype), dims)
end

function similartype(storagetype::Type{<:TensorStorage}, eltype::Type)
  # TODO: Don't convert to an `AbstractVector` with `set_ndims(datatype, 1)`, once we support
  # more general data types.
  # return set_datatype(storagetype, NDTensors.similartype(datatype(storagetype), eltype))
  return set_datatype(storagetype, set_ndims(NDTensors.similartype(datatype(storagetype), eltype), 1))
end

## # Define Base.similar in terms of NDTensors.similar
## Base.similar(t::TensorStorage) = NDTensors.similar(t)
## Base.similar(t::TensorStorage, eltype::Type) = NDTensors.similar(t, eltype)
## Base.similar(t::TensorStorage, dims::Tuple) = NDTensors.similar(t, dims)
## Base.similar(t::TensorStorage, dims::DimOrInd...) = NDTensors.similar(t, dims)
