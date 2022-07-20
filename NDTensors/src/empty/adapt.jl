function adapt_structure(to, x::EmptyStorage)
  return adapt_storagetype(to, typeof(x))()
end

function adapt_storagetype(to::Type{<:AbstractArray}, x::Type{<:EmptyStorage})
  return emptytype(adapt_storagetype(to, fulltype(x)))
end
