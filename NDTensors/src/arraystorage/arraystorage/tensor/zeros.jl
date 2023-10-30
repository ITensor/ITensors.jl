function _zeros(tensortype::Type{<:ArrayStorageTensor}, inds)
  return tensor(generic_zeros(storagetype(tensortype), dims(inds)), inds)
end

# To resolve ambiguity error with `Base.zeros`.
function zeros(tensortype::Type{<:ArrayStorageTensor}, inds)
  return _zeros(tensortype, inds)
end

# To resolve ambiguity error with `Base.zeros`.
function zeros(tensortype::Type{<:ArrayStorageTensor}, inds::Tuple{Vararg{Integer}})
  return _zeros(tensortype, inds)
end
