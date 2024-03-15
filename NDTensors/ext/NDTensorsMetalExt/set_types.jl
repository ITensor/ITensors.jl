# `TypeParameterAccessors.jl` definitions.

using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position, set_type_parameter
using NDTensors.GPUArraysCoreExtensions: storagemode
# Metal-specific type parameter setting
function set_storagemode(arraytype::Type{<:MtlArray}, param)
  return TypeParameterAccessors.set_type_parameter(arraytype, storagemode, param)
end

## TODO remove TypeParameterAccessors when SetParameters is removed
function TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(eltype))
  return Position(1)
end
function TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(Base.ndims))
  return Position(2)
end
function TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(storagemode))
  return Position(3)
end

function TypeParameterAccessors.default_type_parameters(::Type{<:MtlArray})
  return (Float32, 1, Metal.DefaultStorageMode)
end
