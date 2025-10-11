using Metal: MtlArray
using NDTensors.GPUArraysCoreExtensions: storagemode
using NDTensors.TypeParameterAccessors: TypeParameterAccessors, Position

function TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(storagemode))
    return Position(3)
end
