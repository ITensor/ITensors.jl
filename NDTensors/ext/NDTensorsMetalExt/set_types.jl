using Metal: Metal, MtlArray
# `TypeParameterAccessors.jl` definitions.

using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position
using NDTensors.GPUArraysCoreExtensions: storagemode

## TODO remove TypeParameterAccessors when SetParameters is removed
function TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(eltype))
  return Position(1)
end
function TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(ndims))
  return Position(2)
end
function TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(storagemode))
  return Position(3)
end

function TypeParameterAccessors.default_type_parameters(::Type{<:MtlArray})
  return (Float32, 1, Metal.DefaultStorageMode)
end
