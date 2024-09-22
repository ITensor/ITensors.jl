# TypeParameterAccessors definitions
using NDTensors.TypeParameterAccessors:
  TypeParameterAccessors, Position, default_type_parameters
using NDTensors.GPUArraysCoreExtensions: storagemode
using AMDGPU: AMDGPU, ROCArray
using GPUArraysCore: AbstractGPUArray

function TypeParameterAccessors.default_type_parameters(::Type{<:ROCArray})
  return (default_type_parameters(AbstractGPUArray)..., AMDGPU.Mem.HIPBuffer)
end

TypeParameterAccessors.position(::Type{<:ROCArray}, ::typeof(storagemode)) = Position(3)
