using Base: DimOrInd, Dims, OneTo
using SimpleTraits: @traitfn
using TypeParameterAccessors: TypeParameterAccessors, IsWrappedArray, NDims, set_eltype,
    set_ndims, similartype, unwrap_array_type

# Wrapper-aware `similartype` for arbitrary `AbstractArray`s. Used at NDTensors
# call sites that recurse into a generic `AbstractArray` type (potentially a
# wrapper like `SubArray`, `ReshapedArray`, `Adjoint`) where upstream
# `TypeParameterAccessors.similartype` either gives `Any` (because `Base.similar`
# isn't type-stable for the wrapper) or recurses to `Union{}` (because NDTensors
# overloads `Base.similar` for the type). NDTensors-owned types use
# `TypeParameterAccessors.similartype` directly via their own overloads.
@traitfn function array_similartype(
        arraytype::Type{ArrT}
    ) where {{ArrT; !IsWrappedArray{ArrT}}}
    return arraytype
end
@traitfn function array_similartype(
        arraytype::Type{ArrT}, eltype::Type
    ) where {{ArrT; !IsWrappedArray{ArrT}}}
    return set_eltype(arraytype, eltype)
end
@traitfn function array_similartype(
        arraytype::Type{ArrT}, dims::Tuple
    ) where {{ArrT; !IsWrappedArray{ArrT}}}
    return set_ndims(arraytype, length(dims))
end
@traitfn function array_similartype(
        arraytype::Type{ArrT}, ndims::NDims
    ) where {{ArrT; !IsWrappedArray{ArrT}}}
    return set_ndims(arraytype, ndims)
end
@traitfn function array_similartype(
        arraytype::Type{ArrT}
    ) where {{ArrT; IsWrappedArray{ArrT}}}
    return array_similartype(unwrap_array_type(arraytype), NDims(arraytype))
end
@traitfn function array_similartype(
        arraytype::Type{ArrT}, eltype::Type
    ) where {{ArrT; IsWrappedArray{ArrT}}}
    return array_similartype(unwrap_array_type(arraytype), eltype, NDims(arraytype))
end
@traitfn function array_similartype(
        arraytype::Type{ArrT}, dims::Tuple
    ) where {{ArrT; IsWrappedArray{ArrT}}}
    return array_similartype(unwrap_array_type(arraytype), dims)
end
function array_similartype(arraytype::Type{<:AbstractArray}, eltype::Type, ndims::NDims)
    return array_similartype(array_similartype(arraytype, eltype), ndims)
end
function array_similartype(arraytype::Type{<:AbstractArray}, eltype::Type, dims::Tuple)
    return array_similartype(array_similartype(arraytype, eltype), dims)
end

## Custom `NDTensors.similar` implementation.
## More extensive than `Base.similar`.

# This function actually allocates the data.
# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, dims::Tuple)
    shape = NDTensors.to_shape(arraytype, dims)
    return array_similartype(arraytype, shape)(undef, NDTensors.to_shape(arraytype, shape))
end

# This function actually allocates the data.
# Catches conversions of dimensions specified by ranges
# dimensions specified by integers with `Base.to_shape`.
# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, dims::Dims)
    return array_similartype(arraytype, dims)(undef, dims)
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
        shape::Tuple{Union{Integer, OneTo}, Vararg{Union{Integer, OneTo}}}
    )
    return NDTensors.similar(arraytype, NDTensors.to_shape(shape))
end

# NDTensors.similar
function similar(arraytype::Type{<:AbstractArray}, eltype::Type, dims::Tuple)
    return NDTensors.similar(array_similartype(arraytype, eltype, dims), dims)
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
