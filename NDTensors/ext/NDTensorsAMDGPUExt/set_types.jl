using AMDGPU: ROCArray
using NDTensors.GPUArraysCoreExtensions: storagemode
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position

TypeParameterAccessors.position(::Type{<:ROCArray}, ::typeof(storagemode)) = Position(3)
