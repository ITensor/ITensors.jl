# TypeParameterAccessors definitions
using Dagger: Dagger, DArray
using NDTensors.TypeParameterAccessors:
  TypeParameterAccessors, Position, default_type_parameters, parameter, position

blocktype(darray::DArray) = blocktype(typeof(darray))
blocktype(darrayT::Type{<:DArray}) = parameter(darrayT, position(darrayT, blocktype))

function TypeParameterAccessors.position(::Type{<:DArray}, ::typeof(blocktype))
  return Position(3)
end

concattype(darray::DArray) = concattype(typeof(darray))
concattype(darrayT::Type{<:DArray}) = parameter(darrayT, position(darrayT, concattype))

function TypeParameterAccessors.position(::Type{<:DArray}, ::typeof(concattype))
  return Position(4)
end

## TODO use autoblock
function TypeParameterAccessors.default_type_parameters(::Type{<:DArray})
  return (default_type_parameters(AbstractArray)..., Blocks{2}, typeof(cat))
end
