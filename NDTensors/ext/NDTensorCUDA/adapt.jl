#
# Used to adapt `EmptyStorage` types
#

to_vector_type(arraytype::Type{CuArray}) = CuVector
to_vector_type(arraytype::Type{CuArray{T}}) where {T} = CuVector{T}

Adapt.adapt_storage(::CuArrayAdaptor{B}, xs::AbstractArray{T,N}) where {T<:AbstractFloat,N,B} =
  CUDA.isbits(xs) ? xs : CuArray{T,N,B}(xs)

Adapt.adapt_storage(::CuArrayAdaptor{B}, xs::AbstractArray{T,N}) where {T<:Complex{<:AbstractFloat},N,B} =
  CUDA.isbits(xs) ? xs : CuArray{T,N,B}(xs)