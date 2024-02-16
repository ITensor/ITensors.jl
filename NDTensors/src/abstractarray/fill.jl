using .TypeParameterAccessors: specify_defaults
using .Unwrap: unwrap_array_type

function generic_randn(
  arraytype::Type{<:AbstractArray}, dim::Integer=0; rng=Random.default_rng()
)
  arraytype_specified = specify_default_parameters(unwrap_array_type(arraytype))
  data = similar(arraytype_specified, dim)
  return randn!(rng, data)
end

function generic_zeros(arraytype::Type{<:AbstractArray}, dims...)
  arraytype_specified = specify_default_parameters(unwrap_array_type(arraytype))
  ElT = eltype(arraytype_specified)
  return fill!(similar(arraytype_specified, dims...), zero(ElT))
end
