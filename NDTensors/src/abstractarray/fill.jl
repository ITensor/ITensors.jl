using .TypeParameterAccessors:
  unwrap_array_type, specify_default_type_parameters, type_parameter

## Warning to use these functions it is necessary to define `TypeParameterAccessors.position(::Type{<:YourArrayType}, ::typeof(ndims)))`
## Creating these functions to help flatten tuples e.g. generic_randn(Array, (2,3)) -> generic_randn(Array, 2,3)
## So that its possible to take the length of dims and check it agains the given arraytype ndims
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

function generic_zeros(arraytype::Type{<:AbstractArray}, dims::Tuple)
  return generic_zeros(arraytype, dims...)
end

function generic_zeros(arraytype::Type{<:AbstractArray}, dims...)
  ## Right now I have to call specify first because its not possible to 
  ## call specify_default_type_parameter and overwrite a default parameter.
  ## So if arraytypes ndim is unspecified, specify it to the length of dims...
  arraytype_specified = specify_type_parameter(
    unwrap_array_type(arraytype), ndims, length(dims)
  )
  arraytype_specified = specify_default_type_parameters(arraytype_specified)
  @assert length(dims) == type_parameter(arraytype_specified, ndims)
  ElT = eltype(arraytype_specified)
  return fill!(similar(arraytype_specified, dims...), zero(ElT))
end
