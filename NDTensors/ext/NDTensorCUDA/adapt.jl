#
# Used to adapt `EmptyStorage` types
#

to_vector_type(arraytype::Type{CuArray}) = CuVector
to_vector_type(arraytype::Type{CuArray{T}}) where {T} = CuVector{T}

function Adapt.adapt_storage(
  ::CuArrayAdaptor{B}, xs::AbstractArray{T,N}
) where {T<:AbstractFloat,N,B}
  return CUDA.isbits(xs) ? xs : CuArray{T,N,B}(xs)
end

function Adapt.adapt_storage(
  ::CuArrayAdaptor{B}, xs::AbstractArray{T,N}
) where {T<:Complex{<:AbstractFloat},N,B}
  return CUDA.isbits(xs) ? xs : CuArray{T,N,B}(xs)
end
