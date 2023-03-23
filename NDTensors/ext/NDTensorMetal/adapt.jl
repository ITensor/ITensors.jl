#
# Used to adapt `EmptyStorage` types
#

NDTensors.to_vector_type(arraytype::Type{MtlArray}) = MtlVector
NDTensors.to_vector_type(arraytype::Type{MtlArray{T}}) where {T} = MtlVector{T}

@inline NDTensors.mtl(xs; unified::Bool=false) = NDTensors.adapt_structure(MtlArray, xs)
