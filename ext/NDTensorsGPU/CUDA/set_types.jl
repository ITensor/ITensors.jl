using NDTensors

buffertype(datatype::Type{<:CUDA.CuArray{<:Any,<:Any,B}}) where {B} = B

function buffertype(datatype::Type{<:CUDA.CuArray})
  println(
    "CuArray definitions require a CUDA.Mem buffer try $(datatype{default_buffertype()})"
  )
  throw(TypeError)
end

default_buffertype() = CUDA.Mem.DeviceBuffer

function set_eltype(arraytype::Type{<:CUDA.CuArray}, eltype::Type)
  return CUDA.CuArray{eltype,ndims(arraytype),buffertype(arraytype)}
end

function set_ndims(arraytype::Type{<:CUDA.CuArray{T, <:Any, <:Any}}, ndims) where {T}
  return CUDA.CuArray{eltype(arraytype),ndims,buffertype(arraytype)}
end

function set_ndims(arraytype::Type{<:CUDA.CuArray{T}}, ndims) where {T}
  return CUDA.CuArray{eltype(arraytype), ndims, default_buffertype()}
end

function set_ndims(arraytype::Type{<:CUDA.CuArray}, ndims)
  return CUDA.CuArray{NDTensors.default_eltype(), ndims, default_buffertype()}
end