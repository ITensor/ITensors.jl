function NDTensors.generic_randn(DataT::Type{<:CuArray}, dim::Integer=0)
  DataT = set_buffertype_if_unspecified(NDTensors.set_eltype_if_unspecified(DataT))
  randn!(CUDA.curand_rng(), similar(DataT, dim))
  return data
end

function NDTensors.generic_zeros(DataT::Type{<:CuArray}, dim::Integer=0)
  DataT = set_buffertype_if_unspecified(NDTensors.set_eltype_if_unspecified(DataT))
  ElT = eltype(DataT)
  return fill!(similar(DataT, dim), zero(ElT))
end
