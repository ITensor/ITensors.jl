using .TypeParameterAccessors: unwrap_array_type, specify_default_type_parameters

function generic_randn(
  arraytype::Type{<:AbstractArray}, dim::Integer=0; rng=Random.default_rng()
)
  arraytype_specified = specify_default_type_parameters(
    unwrap_array_type(arraytype)
  )
  data = similar(arraytype_specified, dim)
  return randn!(rng, data)
end

function generic_zeros(arraytype::Type{<:AbstractArray}, dims...)
  arraytype_specified = specify_default_type_parameters(
    unwrap_array_type(arraytype)
  )
  ElT = eltype(arraytype_specified)
  return fill!(similar(arraytype_specified, dims...), zero(ElT))
end
