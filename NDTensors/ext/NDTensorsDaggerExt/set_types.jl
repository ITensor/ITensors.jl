# TypeParameterAccessors definitions
using Dagger: Dagger, Blocks, DArray
using NDTensors: NDTensors
using NDTensors.TypeParameterAccessors:
  TypeParameterAccessors, Position, default_type_parameters, parameter, position, set_type_parameters

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

## TODO need to make this work. Need to specify 
function NDTensors.set_ndims(type::Type{<:DArray}, param)
  return set_type_parameters(type, (ndims, blocktype), (param, Blocks{param}))
end
