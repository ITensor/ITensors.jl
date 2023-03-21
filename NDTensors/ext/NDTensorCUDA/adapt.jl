#
# Used to adapt `EmptyStorage` types
#

to_vector_type(arraytype::Type{CuArray}) = CuVector
to_vector_type(arraytype::Type{CuArray{T}}) where {T} = CuVector{T}
