module TypeParameterAccessorsCUDAExt

using CUDA: CUDA, CuArray
using NDTensors.Vendored.TypeParameterAccessors: TypeParameterAccessors, Position

TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(ndims)) = Position(2)
function TypeParameterAccessors.default_type_parameters(::Type{<:CuArray})
    return (Float64, 1, CUDA.default_memory)
end

end
