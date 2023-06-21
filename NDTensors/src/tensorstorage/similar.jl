# NDTensors.similar
similar(storage::TensorStorage) = setdata(storage, NDTensors.similar(data(storage)))

# NDTensors.similar
function similar(storage::TensorStorage, eltype::Type)
  return setdata(storage, NDTensors.similar(data(storage), eltype))
end

# NDTensors.similar
function similar(storage::TensorStorage, dims::Tuple)
  # TODO: Don't convert to an `AbstractVector` with `vec`, once we support
  # more general data types.
  # return setdata(storage, NDTensors.similar(data(storage), dims))
  return setdata(storage, vec(NDTensors.similar(data(storage), dims)))
end

# NDTensors.similar
function similar(storage::TensorStorage, eltype::Type, dims::Tuple)
  # TODO: Don't convert to an `AbstractVector` with `vec`, once we support
  # more general data types.
  # return setdata(storage, NDTensors.similar(data(storage), eltype, dims))
  return setdata(storage, vec(NDTensors.similar(data(storage), eltype, dims)))
end

# NDTensors.similar
function similar(storagetype::Type{<:TensorStorage}, eltype::Type, dims::Tuple)
  return similar(similartype(storagetype, eltype), dims)
end

# NDTensors.similar
function similar(storagetype::Type{<:TensorStorage}, eltype::Type)
  return error("Must specify dimensions.")
end

# NDTensors.similar
function similar(storagetype::Type{<:TensorStorage}, dims::Tuple)
  # TODO: Don't convert to an `AbstractVector` with `vec`, once we support
  # more general data types.
  # return setdata(storagetype, NDTensors.similar(datatype(storagetype), dims))
  return setdata(storagetype, vec(NDTensors.similar(datatype(storagetype), dims)))
end

# NDTensors.similar
function similar(storagetype::Type{<:TensorStorage}, dims::Dims)
  # TODO: Don't convert to an `AbstractVector` with `prod`, once we support
  # more general data types.
  # return setdata(storagetype, NDTensors.similar(datatype(storagetype), dims))
  return setdata(storagetype, NDTensors.similar(datatype(storagetype), prod(dims)))
end

# NDTensors.similar
function similar(storagetype::Type{<:TensorStorage}, dims::DimOrInd...)
  return similar(storagetype, NDTensors.to_shape(dims))
end

# Define Base.similar in terms of NDTensors.similar
Base.similar(storage::TensorStorage) = NDTensors.similar(storage)
Base.similar(storage::TensorStorage, eltype::Type) = NDTensors.similar(storage, eltype)
## TODO: Are these methods needed?
## Base.similar(storage::TensorStorage, dims::Tuple) = NDTensors.similar(storage, dims)
## Base.similar(storage::TensorStorage, dims::Dims...) = NDTensors.similar(storage, dims...)
## Base.similar(storage::TensorStorage, dims::DimOrInd...) = NDTensors.similar(storage, dims...)

function similartype(storagetype::Type{<:TensorStorage}, eltype::Type)
  # TODO: Don't convert to an `AbstractVector` with `set_ndims(datatype, 1)`, once we support
  # more general data types.
  # return set_datatype(storagetype, NDTensors.similartype(datatype(storagetype), eltype))
  return set_datatype(storagetype, set_ndims(similartype(datatype(storagetype), eltype), 1))
end

function similartype(storagetype::Type{<:TensorStorage}, dims::Tuple)
  # TODO: In the future, set the dimensions of the data type based on `dims`, once
  # more general data types beyond `AbstractVector` are supported.
  # `similartype` unwraps any wrapped data.
  return set_ndims(
    set_datatype(storagetype, set_ndims(similartype(datatype(storagetype)), 1)),
    length(dims),
  )
end
