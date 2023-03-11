function set_eltype(arraytype::Type{<:MtlArray}, eltype::Type)
  return MtlArray{eltype,ndims(arraytype)}
end

function set_ndims(arraytype::Type{<:MtlArray}, ndims)
  return MtlArray{eltype(arraytype),ndims}
end