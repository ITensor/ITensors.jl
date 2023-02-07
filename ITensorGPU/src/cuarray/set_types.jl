buffertype(::Type{<:CuArray{<:Any,<:Any,B}}) where {B} = B

function set_eltype(arraytype::Type{<:CuArray}, eltype::Type)
  return CuArray{eltype,ndims(arraytype),buffertype(arraytype)}
end

function set_ndims(arraytype::Type{<:CuArray}, ndims)
  return CuArray{eltype(arraytype),ndims,buffertype(arraytype)}
end
