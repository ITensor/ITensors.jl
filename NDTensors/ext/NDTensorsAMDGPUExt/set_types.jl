# TypeParameterAccessors definitions
using NDTensors.TypeParameterAccessors:
  TypeParameterAccessors, Position, default_type_parameters
using NDTensors.GPUArraysCoreExtensions: storagemode
using AMDGPU: AMDGPU, ROCArray

function TypeParameterAccessors.default_type_parameters(::Type{<:ROCArray})
  return (default_type_parameters(AbstractArray)..., AMDGPU.Mem.HIPBuffer)
end

TypeParameterAccessors.position(::Type{<:ROCArray}, ::typeof(storagemode)) = Position(3)
