using .TypeParameterAccessors: DefaultParameters #specify_parameters
using .Unwrap: unwrap_array_type

function generic_randn(
  arraytype::Type{<:AbstractArray}, dim::Integer=0; rng=Random.default_rng()
)
  arraytype_specified = nothing#specify_parameters(unwrap_array_type(arraytype), DefaultParameters())
  data = similar(arraytype_specified, dim)
  return randn!(rng, data)
end

function generic_zeros(arraytype::Type{<:AbstractArray}, dims...)
  arraytype_specified = nothing#specify_parameters(unwrap_array_type(arraytype), DefaultParameters())
  ElT = eltype(arraytype_specified)
  return fill!(similar(arraytype_specified, dims...), zero(ElT))
end
