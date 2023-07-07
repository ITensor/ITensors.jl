function adapt_structure(to, x::EmptyStorage)
  return adapt_storagetype(to, typeof(x))()
end

function adapt_storagetype(to::Type{<:AbstractArray}, x::Type{<:EmptyStorage})
  d = datatype(storagetype(x))
  return emptytype(adapt_storagetype(adapt(to, d), fulltype(x)))
end
