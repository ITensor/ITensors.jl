using TypeParameterAccessors: TypeParameterAccessors, Position, parenttype
function TypeParameterAccessors.set_ndims(arraytype::Type{<:Tensor}, ndims)
    # TODO: Implement something like:
    # ```julia
    # return set_storagetype(arraytype, set_ndims(storagetype(arraytype), ndims))
    # ```
    # However, we will also need to define `set_ndims(indstype(arraytype), ndims)`
    # and use `set_indstype(arraytype, set_ndims(indstype(arraytype), ndims))`.
    return error(
        "Setting the number dimensions of the array type `$arraytype` (to `$ndims`) is not currently defined."
    )
end

function set_storagetype(tensortype::Type{<:Tensor}, storagetype)
    return Tensor{eltype(tensortype), ndims(tensortype), storagetype, indstype(tensortype)}
end

TypeParameterAccessors.parenttype(tensortype::Type{<:Tensor}) = storagetype(tensortype)
function TypeParameterAccessors.parenttype(storagetype::Type{<:TensorStorage})
    return datatype(storagetype)
end

function TypeParameterAccessors.position(::Type{<:Tensor}, ::typeof(parenttype))
    return Position(3)
end
