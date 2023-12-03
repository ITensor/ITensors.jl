Base.convert(type::Type{<:AbstractSparseArray}, a::AbstractArray) = type(a)

Base.convert(::Type{T}, a::T) where {T<:AbstractSparseArray} = a

function (::Type{T})(a::T) where {T<:AbstractSparseArray}
  return copy(a)
end
