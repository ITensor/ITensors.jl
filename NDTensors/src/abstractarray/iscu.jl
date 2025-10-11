using .Vendored.TypeParameterAccessors: unwrap_array_type
# TODO: Make `isgpu`, `ismtl`, etc.
# For `isgpu`, will require a `NDTensorsGPUArrayCoreExt`.
iscu(A::AbstractArray) = iscu(typeof(A))
function iscu(A::Type{<:AbstractArray})
    return (unwrap_array_type(A) == A ? false : iscu(unwrap_array_type(A)))
end
