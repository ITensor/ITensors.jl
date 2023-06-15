function NDTensors.generic_randn(DataT::Type{<:CuArray}, dim::Integer=0)
  DataT = set_buffertype_if_unspecified(NDTensors.set_eltype_if_unspecified(DataT))
  data = similar(DataT, dim)
  ElT = eltype(DataT)
  for i in 1:length(data)
    data[i] = randn(ElT)
  end
  return data
end

function NDTensors.generic_zeros(DataT::Type{<:CuArray}, dim::Integer=0)
  DataT = set_buffertype_if_unspecified(NDTensors.set_eltype_if_unspecified(DataT))
  ElT = eltype(DataT)
  return fill!(similar(DataT, dim), zero(ElT))
end
