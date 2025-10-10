module NDTensorsMappedArraysExt
using MappedArrays: AbstractMappedArray
using NDTensors: NDTensors
function NDTensors.similar(arraytype::Type{<:AbstractMappedArray}, dims::Tuple{Vararg{Int}})
    return similar(Array{eltype(arraytype)}, dims)
end
function NDTensors.similartype(storagetype::Type{<:AbstractMappedArray})
    return Array{eltype(storagetype), ndims(storagetype)}
end
function NDTensors.similartype(
        storagetype::Type{<:AbstractMappedArray}, dims::Tuple{Vararg{Int}}
    )
    return Array{eltype(storagetype), length(dims)}
end

using MappedArrays: ReadonlyMappedArray
using NDTensors: AllowAlias
# It is a bit unfortunate that we have to define this, it fixes an ambiguity
# error with MappedArrays.
function (arraytype::Type{ReadonlyMappedArray{T, N, A, F}} where {T, N, A <: AbstractArray, F})(
        ::AllowAlias, a::AbstractArray
    )
    return a
end
end
