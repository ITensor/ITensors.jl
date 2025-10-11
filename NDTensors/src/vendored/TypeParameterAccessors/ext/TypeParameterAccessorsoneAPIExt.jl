module TypeParameterAccessorsoneAPIExt

using oneAPI: oneAPI, oneArray
using NDTensors.Vendored.TypeParameterAccessors: TypeParameterAccessors, Position

TypeParameterAccessors.position(::Type{<:oneArray}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{<:oneArray}, ::typeof(ndims)) = Position(2)
function TypeParameterAccessors.default_type_parameters(::Type{<:oneAPI})
    return (Float64, 1, oneAPI.oneL0.DeviceBuffer)
end

end
