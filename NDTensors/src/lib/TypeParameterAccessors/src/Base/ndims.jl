## TypeParameterAccessors.ndims (not imported from Base)

ndims(array) = ndims(typeof(array))
ndims(type::Type) = parameter(type, ndims)

## In house patch to deal issue of calling ndims with an Array of unspecified eltype
## https://github.com/JuliaLang/julia/pull/40682
if VERSION < v"1.7"
  ndims(::Type{<:AbstractArray{<:Any,N}}) where {N} = N
end
