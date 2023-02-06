# NDTensors.similar
similar(tensor::Tensor) = setstorage(tensor, similar(storage(tensor)))

# NDTensors.similar
similar(tensor::Tensor, eltype::Type) = setstorage(tensor, similar(storage(tensor), eltype))

function similartype(tensortype::Type{<:Tensor}, eltype::Type)
  return set_storagetype(tensortype, similartype(storagetype(tensortype), eltype))
end

function similartype(tensortype::Type{<:Tensor}, dims::Tuple)
  tensortype_new_inds = set_indstype(tensortype, dims)
  return set_storagetype(tensortype_new_inds, similartype(storagetype(tensortype_new_inds)))
end
