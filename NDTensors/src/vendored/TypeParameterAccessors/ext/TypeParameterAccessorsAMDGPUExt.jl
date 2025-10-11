module TypeParameterAccessorsAMDGPUExt

using AMDGPU: AMDGPU, ROCArray
using NDTensors.Vendored.TypeParameterAccessors: TypeParameterAccessors, Position

TypeParameterAccessors.position(::Type{<:ROCArray}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{<:ROCArray}, ::typeof(ndims)) = Position(2)
function TypeParameterAccessors.default_type_parameters(::Type{<:ROCArray})
    return (Float64, 1, AMDGPU.Mem.HIPBuffer)
end

end
