function generic_randn(DataT::Type{<:AbstractArray}, dim::Integer=0)
  DataT = set_eltype_if_unspecified(DataT)
  data = similar(DataT, dim)
  ElT = eltype(DataT)
  for i in 1:length(data)
    data[i] = randn(ElT)
  end
  return data
end

function generic_zeros(DataT::Type{<:AbstractArray}, dim::Integer=0)
  DataT = set_eltype_if_unspecified(DataT)
  ElT = eltype(DataT)
  return fill!(similar(DataT, dim), zero(ElT))
end
