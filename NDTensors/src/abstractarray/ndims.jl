## NDTensors.ndims (not imported from Base)
using .TypeParameterAccessors:
  TypeParameterAccessors, Self, type_parameter, set_ndims, set_parameter

ndims((array::AbstractArray)) = ndims(typeof(array))
ndims(arraytype::Type{<:AbstractArray}) = type_parameter(arraytype, Base.ndims)

## TODO for now have `NDTensors.set_ndims` call `TypeParameterAccessors.set_ndims`
NDTensors.set_ndims(type::Type, length) = TypeParameterAccessors.set_ndims(type, length)

TypeParameterAccessors.position(::Type{<:Number}, ::typeof(ndims)) = Self()

"""
`set_indstype` should be overloaded for
types with structured dimensions,
like `OffsetArrays` or named indices
(such as ITensors).
"""
function set_indstype(arraytype::Type{<:AbstractArray}, dims::Tuple)
  return set_ndims(arraytype, length(dims))
end

# ndims(array::AbstractArray) = Base.ndims(array)
# ndims(arraytype::Type{<:AbstractArray}) = Base.ndims(arraytype)

# ## In house patch to deal issue of calling ndims with an Array of unspecified eltype
# ## https://github.com/JuliaLang/julia/pull/40682
# if VERSION < v"1.7"
#   ndims(::Type{<:AbstractArray{<:Any,N}}) where {N} = N
# end
