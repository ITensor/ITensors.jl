function generic_randn(arraytype::Type{<:AbstractArray}, dim::Integer=0)
  arraytype_specified = set_unspecified_parameters(
    leaf_parenttype(arraytype), DefaultParameters()
  )
  data = similar(arraytype_specified, dim)
  ElT = eltype(data)
  for i in 1:length(data)
    data[i] = randn(ElT)
  end
  return data
end

function generic_zeros(arraytype::Type{<:AbstractArray}, dim::Integer=0)
  arraytype_specified = set_unspecified_parameters(
    leaf_parenttype(arraytype), DefaultParameters()
  )
  ElT = eltype(arraytype_specified)
  return fill!(similar(arraytype_specified, dim), zero(ElT))
end
