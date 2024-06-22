# TypeParameterAccessors definitions
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position
using NDTensors.GPUArraysCoreExtensions: storagemode
using AMDGPU: AMDGPU, ROCArray

function TypeParameterAccessors.default_type_parameters(::Type{<:ROCArray})
  return (Float64, 1, AMDGPU.Mem.HIPBuffer)
end

TypeParameterAccessors.position(::Type{<:ROCArray}, ::typeof(storagemode)) = Position(3)
