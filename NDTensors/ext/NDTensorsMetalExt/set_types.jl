using Metal: MtlArray
using NDTensors.GPUArraysCoreExtensions: storagemode
using NDTensors.Vendored.TypeParameterAccessors: Position, TypeParameterAccessors

function TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(storagemode))
    return Position(3)
end
