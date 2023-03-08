ndims(x) = Base.ndims(x)

if VERSION < v"1.7"
  ndims(x::AbstractArray) = ndims(typeof(x))

  ndims(x::Type{<:AbstractArray{<:Any, N::Number}}) = N
end