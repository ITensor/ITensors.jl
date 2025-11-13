using Base: DimOrInd, Dims, OneTo
using .Vendored.TypeParameterAccessors: IsWrappedArray, unwrap_array_type, set_eltype, similartype

## Custom `NDTensors.similar` implementation.
## More extensive than `Base.similar`.

# This function actually allocates the data.
# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, dims::Tuple)
    shape = NDTensors.to_shape(arraytype, dims)
    return similartype(arraytype, shape)(undef, NDTensors.to_shape(arraytype, shape))
end

# This function actually allocates the data.
# Catches conversions of dimensions specified by ranges
# dimensions specified by integers with `Base.to_shape`.
# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, dims::Dims)
    return similartype(arraytype, dims)(undef, dims)
end

# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, dims::DimOrInd...)
    return similar(arraytype, NDTensors.to_shape(dims))
end

# Handles range inputs, `Base.to_shape` converts them to integer dimensions.
# See Julia's `base/abstractarray.jl`.
# NDTensors.similar
function similar(
        arraytype::Type{<:AbstractArray},
        shape::Tuple{Union{Integer, OneTo}, Vararg{Union{Integer, OneTo}}},
    )
    return NDTensors.similar(arraytype, NDTensors.to_shape(shape))
end

# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, eltype::Type, dims::Tuple)
    return NDTensors.similar(similartype(arraytype, eltype, dims), dims)
end

# TODO: Add an input `structure` which can store things like the nonzero
# structure of a sparse/block sparse tensor.
# NDTensors.similar
# function similar(arraytype::Type{<:AbstractArray}, structure)
#   return NDTensors.similar(similartype(arraytype, structure), structure)
# end

# TODO: Add an input `structure` which can store things like the nonzero
# structure of a sparse/block sparse tensor.
# NDTensors.similar
# function similar(arraytype::Type{<:AbstractArray}, eltype::Type, structure)
#   return NDTensors.similar(similartype(arraytype, eltype, structure), structure)
# end

# TODO: Add an input `structure` which can store things like the nonzero
# structure of a sparse/block sparse tensor.
# NDTensors.similar
# function similar(arraytype::Type{<:AbstractArray}, structure, dims::Tuple)
#   return NDTensors.similar(similartype(arraytype, structure, dims), structure, dims)
# end

# TODO: Add an input `structure` which can store things like the nonzero
# structure of a sparse/block sparse tensor.
# NDTensors.similar
# function similar(arraytype::Type{<:AbstractArray}, eltype::Type, structure, dims::Tuple)
#   return NDTensors.similar(similartype(arraytype, eltype, structure, dims), structure, dims)
# end

# TODO: Maybe makes an empty array, i.e. `similartype(arraytype, eltype)()`?
# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, eltype::Type)
    return error("Must specify dimensions.")
end

## NDTensors.similar for instances

# NDTensors.similar
function similar(array::AbstractArray, eltype::Type, dims::Tuple)
    return NDTensors.similar(similartype(typeof(array), eltype), dims)
end

# NDTensors.similar
function similar(array::AbstractArray, eltype::Type, dims::Int)
    return NDTensors.similar(similartype(typeof(array), eltype), dims)
end

# NDTensors.similar
similar(array::AbstractArray, dims::Tuple) = NDTensors.similar(typeof(array), dims)

# Use the `size` to determine the dimensions
# NDTensors.similar
function similar(array::AbstractArray, eltype::Type)
    return NDTensors.similar(typeof(array), eltype, size(array))
end

# Use the `size` to determine the dimensions
# NDTensors.similar
similar(array::AbstractArray) = NDTensors.similar(typeof(array), size(array))
