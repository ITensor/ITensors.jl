buffertype(datatype::Type{<:CuArray{<:Any,<:Any,B}}) where {B} = B
function buffertype(datatype::Type{<:CuArray})
  println(
    "CuArray definitions require a CUDA.Mem buffer try $(datatype{default_buffertype()})"
  )
  throw(TypeError)
end

default_buffertype() = CUDA.Mem.DeviceBuffer

function set_eltype(arraytype::Type{<:CuArray}, eltype::Type)
  return CuArray{eltype,ndims(arraytype),buffertype(arraytype)}
end

function set_ndims(arraytype::Type{<:CuArray}, ndims)
  return CuArray{eltype(arraytype),ndims,buffertype(arraytype)}
end
