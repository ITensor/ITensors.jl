## NDTensors.ndims (not imported from Base)

ndims(array::AbstractArray) = Base.ndims(array)
ndims(arrayT::Type{<:AbstractArray}) = Base.ndims(arrayT)

## In house patch to deal issue of calling ndims with an Array of unspecified eltype
## https://github.com/JuliaLang/julia/pull/40682
if VERSION < v"1.7"
  ndims(array::Type{<:AbstractArray{<:Any,N}}) where {N} = N
end
