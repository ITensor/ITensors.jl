buffertype(datatype::Type{<:CuArray{<:Any,<:Any,B}}) where {B} = B
function buffertype(datatype::Type{<:CuArray})
  return error("No buffer type specified in type $(datatype)")
end

default_buffertype() = CUDA.Mem.DeviceBuffer

function set_eltype(arraytype::Type{<:CuArray}, eltype::Type)
  return CuArray{eltype,NDTensors.ndims(arraytype),buffertype(arraytype)}
end

function set_ndims(arraytype::Type{<:CuArray{T,<:Any,<:Any}}, ndims) where {T}
  return CuArray{eltype(arraytype),ndims,buffertype(arraytype)}
end

function set_ndims(arraytype::Type{<:CuArray{T}}, ndims) where {T}
  return CuArray{eltype(arraytype),ndims,default_buffertype()}
end

function set_ndims(arraytype::Type{<:CuArray}, ndims)
  return CuArray{NDTensors.default_eltype(),ndims,default_buffertype()}
end

function NDTensors.set_eltype_if_unspecified(
  arraytype::Type{CuVector{T}}, eltype::Type
) where {T}
  return arraytype
end
function NDTensors.set_eltype_if_unspecified(arraytype::Type{CuVector}, eltype::Type)
  return CuVector{eltype}
end
