using FillArrays: AbstractFill, Fill, Zeros
using NDTensors.TypeParameterAccessors: Position, default_parameter
using NDTensors.UnspecifiedTypes: UnspecifiedZero

# ## default parameters
function default_parameter(::Type{<:AbstractFill}, ::Position{1})
  return UnspecifiedZero
end
default_parameter(::Type{<:AbstractFill}, ::Position{2}) = 0
default_parameter(::Type{<:AbstractFill}, ::Position{3}) = Tuple{}
