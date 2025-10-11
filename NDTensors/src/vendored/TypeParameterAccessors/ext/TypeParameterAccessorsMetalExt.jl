module TypeParameterAccessorsMetalExt

using Metal: Metal, MtlArray
using NDTensors.Vendored.TypeParameterAccessors: TypeParameterAccessors, Position

TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{<:MtlArray}, ::typeof(ndims)) = Position(2)
function TypeParameterAccessors.default_type_parameters(::Type{<:MtlArray})
    return (Float64, 1, Metal.DefaultStorageMode)
end

end
