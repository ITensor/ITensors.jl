function generic_randn(
  arraytype::Type{<:AbstractArray}, dim::Integer=0; rng=Random.default_rng()
)
  arraytype_specified = set_unspecified_parameters(
    leaf_parenttype(arraytype), DefaultParameters()
  )
  data = similar(arraytype_specified, dim)
  return randn!(rng, data)
end

function generic_zeros(arraytype::Type{<:AbstractArray}, dims...)
  arraytype_specified = set_unspecified_parameters(
    leaf_parenttype(arraytype), DefaultParameters()
  )
  ElT = eltype(arraytype_specified)
  return fill!(similar(arraytype_specified, dims...), zero(ElT))
end
