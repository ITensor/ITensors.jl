function generic_randn(
  arraytype::Type{<:AbstractArray}, dim::Integer=0; rng=Random.default_rng()
)
  arraytype_specified = set_unspecified_parameters(
    unwrap_type(arraytype), DefaultParameters()
  )
  data = similar(arraytype_specified, dim)
  return randn!(rng, data)
end

function generic_zeros(arraytype::Type{<:AbstractArray}, dim::Integer=0)
  arraytype_specified = set_unspecified_parameters(
    unwrap_type(arraytype), DefaultParameters()
  )
  ElT = eltype(arraytype_specified)
  return fill!(similar(arraytype_specified, dim), zero(ElT))
end
