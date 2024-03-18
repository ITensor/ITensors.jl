# TypeParameterAccessors definitions
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position
using NDTensors.GPUArraysCoreExtensions: storagemode

TypeParameterAccessors.default_type_parameters(::Type{<:ROCArray}) = (Float64, 1, Mem.HIPBuffer)
TypeParameterAccessors.position(::Type{<:ROCArray}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{<:ROCArray}, ::typeof(ndims)) = Position(2)
TypeParameterAccessors.position(::Type{<:ROCArray}, ::typeof(storagemode)) = Position(3)
