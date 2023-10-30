function zeros(tensortype::Type{<:ArrayStorageTensor}, inds)
  return tensor(generic_zeros(storagetype(tensortype), dims(inds)), inds)
end
