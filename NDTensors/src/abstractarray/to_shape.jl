# Like `Base.to_shape` but more general, can return
# `Index`, etc. Customize for an array/tensor
# with custom index types.
# NDTensors.to_shape
function to_shape(arraytype::Type{<:AbstractArray}, dims::Tuple)
  return NDTensors.to_shape(dims)
end
# NDTensors.to_shape
to_shape(dims) = Base.to_shape(dims)

# NDTensors.to_shape overoads for block dimensions.
to_shape(dims::Tuple{Vararg{Vector{<:Integer}}}) = map(to_shape, dims)
# Each dimension.
# NDTensors.to_shape
to_shape(i::Vector{<:Integer}) = sum(to_shape, i)
