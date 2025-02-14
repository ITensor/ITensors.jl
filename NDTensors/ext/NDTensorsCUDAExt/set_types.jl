using CUDA: CuArray
using NDTensors.GPUArraysCoreExtensions: storagemode
using TypeParameterAccessors: TypeParameterAccessors, Position

function TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(storagemode))
  return Position(3)
end
