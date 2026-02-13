using CUDA: CuArray
using NDTensors.GPUArraysCoreExtensions: storagemode
using NDTensors.Vendored.TypeParameterAccessors: Position, TypeParameterAccessors

function TypeParameterAccessors.position(::Type{<:CuArray}, ::typeof(storagemode))
    return Position(3)
end
