module TypeParameterAccessorsCUDAExt

using CUDA: CUDA, CuArray
using NDTensors.Vendored.TypeParameterAccessors: Position, TypeParameterAccessors

# CUDA.jl v6.0 moved `default_memory` from `CUDA` to `CUDA.CUDACore`.
const default_memory = if isdefined(CUDA, :default_memory)
    CUDA.default_memory
else
    CUDA.CUDACore.default_memory
end

TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(ndims)) = Position(2)
function TypeParameterAccessors.default_type_parameters(::Type{<:CuArray})
    return (Float64, 1, default_memory)
end

end
