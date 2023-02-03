buffertype(::Type{<:CuArray{<:Any,<:Any,B}}) where {B} = B

function similartype(arraytype::Type{<:CuArray}, eltype::Type)
  return CuArray{eltype,ndims(arraytype),buffertype(arraytype)}
end

function similartype(arraytype::Type{<:CuArray}, dims::Tuple)
  return CuArray{eltype(arraytype),length(dims),buffertype(arraytype)}
end
