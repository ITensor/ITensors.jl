using .Vendored.TypeParameterAccessors: TypeParameterAccessors, Position, parenttype

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractVector})
    return Dense{eltype(datatype), datatype}
end

function set_datatype(storagetype::Type{<:Dense}, datatype::Type{<:AbstractArray})
    return error(
        "Setting the `datatype` of the storage type `$storagetype` to a $(ndims(datatype))-dimsional array of type `$datatype` is not currently supported, use an `AbstractVector` instead.",
    )
end

TypeParameterAccessors.default_type_parameters(::Type{<:Dense}) = (Float64, Vector)
TypeParameterAccessors.position(::Type{<:Dense}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{<:Dense}, ::typeof(parenttype)) = Position(2)
