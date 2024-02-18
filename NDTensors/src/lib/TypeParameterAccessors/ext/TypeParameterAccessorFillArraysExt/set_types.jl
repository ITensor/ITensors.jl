using FillArrays: AbstractFill, Fill, Zeros
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position
using NDTensors.UnspecifiedTypes: UnspecifiedZero

axestype(::Type{<:AbstractFill}) = nothing

# ## default parameters
function TypeParameterAccessors.default_parameter(::Type{<:AbstractFill}, ::typeof(eltype))
  return UnspecifiedZero
end
TypeParameterAccessors.default_parameter(::Type{<:AbstractFill}, ::typeof(ndims)) = 0
#TypeParameterAccessors.default_parameter(::Type{<:AbstractFill}, ::typeof(axestype)) = Tuple{}

#TypeParameterAccessors.parameter_function(::Type{<:AbstractFill}, ::Position{3}) = axestype
