using FillArrays: AbstractFill, Fill, Zeros
using NDTensors.TypeParameterAccessors: Position, default_parameter
using NDTensors.UnspecifiedTypes: UnspecifiedZero
using NDTensors.UnallocatedArrays: axestype

# ## default parameters
function default_parameter(::Type{<:AbstractFill}, ::typeof(eltype))
  return UnspecifiedZero
end
default_parameter(::Type{<:AbstractFill}, ::typeof(ndims)) = 0
default_parameter(::Type{<:AbstractFill}, ::typeof(axestype)) = Tuple{}
