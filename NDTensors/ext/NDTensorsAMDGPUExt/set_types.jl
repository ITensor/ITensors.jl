using AMDGPU: ROCArray
using NDTensors.GPUArraysCoreExtensions: storagemode
using NDTensors.Vendored.TypeParameterAccessors: Position, TypeParameterAccessors

TypeParameterAccessors.position(::Type{<:ROCArray}, ::typeof(storagemode)) = Position(3)
