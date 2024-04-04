using .TypeParameterAccessors:
  unwrap_array_type, specify_default_type_parameters, type_parameter

## Warning to use these functions it is necessary to define `TypeParameterAccessors.position(::Type{<:YourArrayType}, ::typeof(ndims)))`
# Implementation, catches if `ndims(arraytype) != length(dims)`.
function generic_randn(
  arraytype::Type{<:AbstractArray}, dims::Tuple; rng=Random.default_rng()
)
  return generic_randn(arraytype, dims...; rng=rng)
end

function generic_randn(arraytype::Type{<:AbstractArray}, dims...; rng=Random.default_rng())
  arraytype_specified = specify_type_parameter(
    unwrap_array_type(arraytype), ndims, length(dims)
  )
  arraytype_specified = specify_default_type_parameters(arraytype_specified)
  @assert length(dims) == type_parameter(arraytype_specified, ndims)
  data = similar(arraytype_specified, dims...)
  return randn!(rng, data)
end

# Implementation, catches if `ndims(arraytype) != length(dims)`.
function generic_zeros(arraytype::Type{<:AbstractArray}, dims::Tuple)
  return generic_zeros(arraytype, dims...)
end

function generic_zeros(arraytype::Type{<:AbstractArray}, dims...)
  arraytype_specified = specify_type_parameter(
    unwrap_array_type(arraytype), ndims, length(dims)
  )
  arraytype_specified = specify_default_type_parameters(arraytype_specified)
  @assert length(dims) == type_parameter(arraytype_specified, ndims)
  ElT = eltype(arraytype_specified)
  return fill!(similar(arraytype_specified, dims...), zero(ElT))
end
