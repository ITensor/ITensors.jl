using .TypeParameterAccessors:
  TypeParameterAccessors,
  Position,
  parenttype,
  set_parenttype,
  set_type_parameter,
  set_type_parameters
function TypeParameterAccessors.set_ndims(arraytype::Type{<:Tensor}, ndims)
  # TODO: Implement something like:
  # ```julia
  # return set_storagetype(arraytype, set_ndims(storagetype(arraytype), ndims))
  # ```
  # However, we will also need to define `set_ndims(indstype(arraytype), ndims)`
  # and use `set_indstype(arraytype, set_ndims(indstype(arraytype), ndims))`.
  return error(
    "Setting the number dimensions of the array type `$arraytype` (to `$ndims`) is not currently defined.",
  )
end

function set_storagetype(tensortype::Type{<:Tensor}, storagetype)
  return Tensor{eltype(tensortype),ndims(tensortype),storagetype,indstype(tensortype)}
end

# TODO: Modify the `storagetype` according to `inds`, such as the dimensions?
# TODO: Make a version that accepts `indstype::Type`?
function set_indstype(tensortype::Type{<:Tensor}, inds::Tuple)
  N = length(inds)
  return set_type_parameters(tensortype, (Base.ndims, indstype), (N, typeof(inds)))
end

TypeParameterAccessors.position(::Type{<:Tensor}, ::typeof(eltype)) = Position(1)
TypeParameterAccessors.position(::Type{<:Tensor}, ::typeof(Base.ndims)) = Position(2)
TypeParameterAccessors.position(::Type{<:Tensor}, ::typeof(parenttype)) = Position(3)
TypeParameterAccessors.position(::Type{<:Tensor}, ::typeof(indstype)) = Position(4)
