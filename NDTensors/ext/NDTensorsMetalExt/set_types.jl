using Metal: MtlArray
using NDTensors.GPUArraysCoreExtensions: storagemode
using TypeParameterAccessors: TypeParameterAccessors, Position

function TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(storagemode))
  return Position(3)
end
