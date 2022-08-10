function set_datatype(storagetype::Type{<:BlockSparse}, datatype::Type{<:AbstractVector})
  return BlockSparse{eltype(datatype),datatype,ndims(storagetype)}
end
